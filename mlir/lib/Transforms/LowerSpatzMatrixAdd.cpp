//===- LowerSpatzMatrixAdd.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Spatz/SpatzDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include <limits>

using namespace mlir;

namespace {

static func::FuncOp getOrInsertPrivateFunction(PatternRewriter &rewriter, ModuleOp module, Location loc, StringRef name, FunctionType type) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name)) return fn;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto fn = rewriter.create<func::FuncOp>(loc, name, type);
  fn.setPrivate();
  return fn;
}

static Value createMinIndex(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
  return builder.create<arith::SelectOp>(loc, cond, lhs, rhs);
}

static Value createSubview1D(OpBuilder &builder, Location loc, Value source, Value offset, Value size, Type elemType, Attribute memorySpace) {
  SmallVector<OpFoldResult> offsets = {offset};
  SmallVector<OpFoldResult> sizes = {size};
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1)};
  return builder.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
}

static Value createSubview2D(OpBuilder &builder, Location loc, Value source, Value rowOffset, Value colOffset, Value rows, Value cols, Type elemType, Attribute memorySpace) {
  SmallVector<OpFoldResult> offsets = {rowOffset, colOffset};
  SmallVector<OpFoldResult> sizes = {rows, cols};
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
  return builder.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
}

static Value createSubview3DTo2D(OpBuilder &builder, Location loc, Value source, Value batchOffset, Value rowOffset, Value hiddenOffset, Value rows, Value hiddenSize) {
  SmallVector<OpFoldResult, 3> offsets = {batchOffset, rowOffset, hiddenOffset};
  SmallVector<OpFoldResult, 3> sizes = {builder.getIndexAttr(1), rows, hiddenSize};
  SmallVector<OpFoldResult, 3> strides = { builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
  auto sourceType = source.getType().cast<MemRefType>();
  auto resultType = memref::SubViewOp::inferRankReducedResultType( 2, sourceType, offsets, sizes, strides).cast<MemRefType>();
  return builder.create<memref::SubViewOp>(loc, resultType, source, offsets, sizes,strides)
      .getResult();
}

template <typename ArithOp>
static ArithOp getSingleArithmeticOp(linalg::GenericOp op) {
  if (!op) return nullptr;
  ArithOp result = nullptr;
  for (Operation &nested : op.getRegion().front().without_terminator()) {
    auto candidate = dyn_cast<ArithOp>(&nested);
    if (!candidate || result) return nullptr;
    result = candidate;
  }
  return result;
}

static bool hasRank(Value value, int64_t rank) {
  auto type = value.getType().dyn_cast<MemRefType>();
  return type && type.getRank() == rank;
}

static bool writesTo(linalg::GenericOp op, Value value) {
  for (Value output : op.outputs()) if (output == value) return true;
  return false;
}

static linalg::GenericOp getGenericProducer(Value value) {
  linalg::GenericOp writer = nullptr;
  for (Operation *user : value.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic || !writesTo(generic, value)) continue;
    if (writer) return nullptr;
    writer = generic;
  }
  return writer;
}

struct LayerNormMatch {
  Value input;
  Value gamma;
  Value beta;
  Value output;
  float epsilon;
  SmallVector<Operation *> opsToErase;
};

static FailureOr<LayerNormMatch> matchLayerNorm(linalg::GenericOp finalAdd) {
  LayerNormMatch match;
  match.epsilon = 1.0e-12f;

  if (!getSingleArithmeticOp<arith::AddFOp>(finalAdd) || finalAdd.getNumInputs() != 2 || finalAdd.getNumOutputs() != 1)
    return failure();

  Value scaled, beta;
  if (hasRank(finalAdd.inputs()[0], 3) && hasRank(finalAdd.inputs()[1], 1)) {
    scaled = finalAdd.inputs()[0];
    beta = finalAdd.inputs()[1];
  } else if (hasRank(finalAdd.inputs()[1], 3) && hasRank(finalAdd.inputs()[0], 1)) {
    scaled = finalAdd.inputs()[1];
    beta = finalAdd.inputs()[0];
  } else return failure();

  auto scaleOp = getGenericProducer(scaled);
  if (!scaleOp || !getSingleArithmeticOp<arith::MulFOp>(scaleOp) || scaleOp.getNumInputs() != 2)
    return failure();

  Value normalized, gamma;
  if (hasRank(scaleOp.inputs()[0], 3) && hasRank(scaleOp.inputs()[1], 1)) {
    normalized = scaleOp.inputs()[0];
    gamma = scaleOp.inputs()[1];
  } else if (hasRank(scaleOp.inputs()[1], 3) && hasRank(scaleOp.inputs()[0], 1)) {
    normalized = scaleOp.inputs()[1];
    gamma = scaleOp.inputs()[0];
  } else return failure();

  auto normalizeOp = getGenericProducer(normalized);
  if (!normalizeOp || !getSingleArithmeticOp<arith::MulFOp>(normalizeOp) || normalizeOp.getNumInputs() != 2)
    return failure();

  Value centered = nullptr;
  Value invStdBroadcast = nullptr;
  for (Value input : normalizeOp.inputs()) {
    auto producer = getGenericProducer(input);
    if (producer && getSingleArithmeticOp<arith::SubFOp>(producer)) centered = input;
    else invStdBroadcast = input;
  }
  if (!centered || !invStdBroadcast) return failure();

  auto centeredOp = getGenericProducer(centered);
  if (!centeredOp || !getSingleArithmeticOp<arith::SubFOp>(centeredOp) || centeredOp.getNumInputs() != 2)
    return failure();

  Value input = centeredOp.inputs()[0];
  Value meanBroadcast = centeredOp.inputs()[1];
  if (!hasRank(input, 3) || !hasRank(meanBroadcast, 3)) return failure();

  linalg::GenericOp squareOp = nullptr;
  Value squareBuffer = nullptr;
  for (Operation *user : centered.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic || generic.getNumInputs() != 1 || generic.inputs()[0] != centered) continue;

    auto mul = getSingleArithmeticOp<arith::MulFOp>(generic);
    if (!mul) continue;

    Value arg = generic.getRegion().front().getArgument(0);
    if (mul.getLhs() == arg && mul.getRhs() == arg) {
      squareOp = generic;
      squareBuffer = generic.outputs()[0];
      break;
    }
  }
  if (!squareOp) return failure();

  auto meanBroadcastOp = getGenericProducer(meanBroadcast);
  auto invStdBroadcastOp = getGenericProducer(invStdBroadcast);
  if (!meanBroadcastOp || !invStdBroadcastOp) return failure();

  auto meanCollapse = meanBroadcastOp.inputs()[0].getDefiningOp<memref::CollapseShapeOp>();
  auto invStdCollapse = invStdBroadcastOp.inputs()[0].getDefiningOp<memref::CollapseShapeOp>();
  if (!meanCollapse || !invStdCollapse) return failure();

  Value meanBuffer = meanCollapse.src();
  Value invStdBuffer = invStdCollapse.src();

  auto meanDivOp = getGenericProducer(meanBuffer);
  auto rsqrtOp = getGenericProducer(invStdBuffer);
  if (!meanDivOp || !rsqrtOp) return failure();

  Value meanSumBuffer = meanDivOp.inputs()[0];
  auto meanReduceOp = getGenericProducer(meanSumBuffer);
  if (!meanReduceOp) return failure();

  memref::CopyOp meanInitCopy = nullptr;
  Value zeroBuffer = nullptr;
  for (Operation *user : meanSumBuffer.getUsers()) {
    auto copy = dyn_cast<memref::CopyOp>(user);
    if (copy && copy.getTarget() == meanSumBuffer) { meanInitCopy = copy; zeroBuffer = copy.getSource(); break; }
  }
  if (!meanInitCopy) return failure();

  Value varianceEpsBuffer = rsqrtOp.inputs()[0];
  auto epsilonAddOp = getGenericProducer(varianceEpsBuffer);
  if (!epsilonAddOp) return failure();

  Value varianceBuffer = epsilonAddOp.inputs()[0];
  auto varianceDivOp = getGenericProducer(varianceBuffer);
  if (!varianceDivOp) return failure();

  Value varianceSumBuffer = varianceDivOp.inputs()[0];
  auto varianceReduceOp = getGenericProducer(varianceSumBuffer);
  if (!varianceReduceOp || varianceReduceOp.inputs()[0] != squareBuffer) return failure();

  memref::CopyOp varianceInitCopy = nullptr;
  for (Operation *user : varianceSumBuffer.getUsers()) {
    auto copy = dyn_cast<memref::CopyOp>(user);
    if (copy && copy.getTarget() == varianceSumBuffer) {
      varianceInitCopy = copy;
      if (zeroBuffer && copy.getSource() != zeroBuffer) return failure();
      zeroBuffer = copy.getSource();
      break;
    }
  }
  if (!varianceInitCopy) return failure();

  linalg::FillOp zeroFill = nullptr;
  if (zeroBuffer) {
    for (Operation *user : zeroBuffer.getUsers()) {
      auto fill = dyn_cast<linalg::FillOp>(user);
      if (!fill) continue;
      for (Value output : fill.outputs()) if (output == zeroBuffer) { zeroFill = fill; break; }
      if (zeroFill) break;
    }
  }

  match.input = input;
  match.gamma = gamma;
  match.beta = beta;
  match.output = finalAdd.outputs()[0];

  match.opsToErase.push_back(finalAdd);
  match.opsToErase.push_back(scaleOp);
  match.opsToErase.push_back(normalizeOp);
  match.opsToErase.push_back(invStdBroadcastOp);
  match.opsToErase.push_back(invStdCollapse);
  match.opsToErase.push_back(rsqrtOp);
  match.opsToErase.push_back(epsilonAddOp);
  match.opsToErase.push_back(varianceDivOp);
  match.opsToErase.push_back(varianceReduceOp);
  match.opsToErase.push_back(squareOp);
  match.opsToErase.push_back(centeredOp);
  match.opsToErase.push_back(meanBroadcastOp);
  match.opsToErase.push_back(meanCollapse);
  match.opsToErase.push_back(meanDivOp);
  match.opsToErase.push_back(meanReduceOp);
  match.opsToErase.push_back(meanInitCopy);
  match.opsToErase.push_back(varianceInitCopy);
  if (zeroFill) match.opsToErase.push_back(zeroFill);

  return match;
}

struct LinalgLayerNormFusionPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto match = matchLayerNorm(op);
    if (failed(match)) return failure();
    auto func = op->getParentOfType<func::FuncOp>();
    rewriter.create<spatz::LayerNormOp>(op.getLoc(), match->input, match->gamma, match->beta, match->output, rewriter.getF32FloatAttr(match->epsilon), rewriter.getI64IntegerAttr(64), rewriter.getI64IntegerAttr(128));
    for (Operation *eraseOp : match->opsToErase) if (eraseOp && eraseOp->getBlock()) rewriter.eraseOp(eraseOp);
    SmallVector<memref::AllocOp> deadAllocs;
    func.walk([&](memref::AllocOp alloc) { if (alloc.getResult().use_empty()) deadAllocs.push_back(alloc); });
    for (memref::AllocOp alloc : deadAllocs) if (alloc->getBlock()) rewriter.eraseOp(alloc);
    return success();
  }
};

struct LayerNormLowering : public OpRewritePattern<spatz::LayerNormOp> {
  using OpRewritePattern<spatz::LayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::LayerNormOp op, PatternRewriter &rewriter) const override {
    auto inputType = op.input().getType().dyn_cast<MemRefType>();
    auto gammaType = op.gamma().getType().dyn_cast<MemRefType>();
    auto betaType = op.beta().getType().dyn_cast<MemRefType>();
    auto outputType = op.output().getType().dyn_cast<MemRefType>();
    if (!inputType || !gammaType || !betaType || !outputType) return failure();
    if (inputType.getRank() != 3 || outputType.getRank() != 3 || gammaType.getRank() != 1 || betaType.getRank() != 1) return failure();

    Type elemType = inputType.getElementType();
    if (!elemType.isF32()) return failure();

    int64_t tileRowsVal = op.tileRows();
    int64_t maxHiddenVal = op.maxHidden();
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type i32Type = rewriter.getI32Type();
    FloatType f32Type = rewriter.getF32Type();

    auto coreIdxFn = getOrInsertPrivateFunction(rewriter, module, loc, "snrt_cluster_core_idx", rewriter.getFunctionType({}, {i32Type}));
    auto barrierFn = getOrInsertPrivateFunction(rewriter, module, loc, "snrt_cluster_hw_barrier", rewriter.getFunctionType({}, {}));
    auto l1ResetFn = getOrInsertPrivateFunction(rewriter, module, loc, "snrt_l1alloc_reset", rewriter.getFunctionType({}, {}));
    auto rsqrtFn = getOrInsertPrivateFunction(rewriter, module, loc, "baremetal_rsqrt", rewriter.getFunctionType({f32Type}, {f32Type}));

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value tileRows = rewriter.create<arith::ConstantIndexOp>(loc, tileRowsVal);
    Value vectorChunk = rewriter.create<arith::ConstantIndexOp>(loc, 64);
    Value c0I32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value zeroF32 = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), f32Type);
    Value epsilon = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(op.epsilon().convertToFloat()), f32Type);
    Value batchDim = rewriter.create<memref::DimOp>(loc, op.input(), 0);
    Value sequenceDim = rewriter.create<memref::DimOp>(loc, op.input(), 1);
    Value hiddenDim = rewriter.create<memref::DimOp>(loc, op.input(), 2);
    auto l1Space = IntegerAttr::get(rewriter.getI64Type(), 1);
    auto tileType = MemRefType::get({tileRowsVal, maxHiddenVal}, elemType, MemRefLayoutAttrInterface{}, l1Space);
    auto paramType = MemRefType::get({maxHiddenVal}, elemType, MemRefLayoutAttrInterface{}, l1Space);
    Value tileL1 = rewriter.create<memref::AllocOp>(loc, tileType);
    Value gammaL1 = rewriter.create<memref::AllocOp>(loc, paramType);
    Value betaL1 = rewriter.create<memref::AllocOp>(loc, paramType);
    Value coreId = rewriter.create<func::CallOp>(loc, coreIdxFn, ValueRange{}).getResult(0);
    Value isCore0 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, coreId, c0I32);

    rewriter.create<scf::IfOp>(loc, isCore0, [&](OpBuilder &b0, Location l0) {
      auto tagType = MemRefType::get({1}, b0.getI32Type());
      Value tagGamma = b0.create<memref::AllocaOp>(l0, tagType);
      Value tagBeta = b0.create<memref::AllocaOp>(l0, tagType);
      Value tagInput = b0.create<memref::AllocaOp>(l0, tagType);
      Value tagOutput = b0.create<memref::AllocaOp>(l0, tagType);

      Value gammaSrc = createSubview1D(b0, l0, op.gamma(), c0, hiddenDim, elemType, gammaType.getMemorySpace());
      Value betaSrc = createSubview1D(b0, l0, op.beta(), c0, hiddenDim, elemType, betaType.getMemorySpace());
      Value gammaDst = createSubview1D(b0, l0, gammaL1, c0, hiddenDim, elemType, l1Space);
      Value betaDst = createSubview1D(b0, l0, betaL1, c0, hiddenDim, elemType, l1Space);
      b0.create<memref::DmaStartOp>(l0, gammaSrc, ValueRange{c0}, gammaDst, ValueRange{c0}, hiddenDim, tagGamma, ValueRange{c0});
      b0.create<memref::DmaStartOp>(l0, betaSrc, ValueRange{c0}, betaDst, ValueRange{c0}, hiddenDim, tagBeta, ValueRange{c0});
      b0.create<memref::DmaWaitOp>(l0, tagGamma, ValueRange{c0}, hiddenDim);
      b0.create<memref::DmaWaitOp>(l0, tagBeta, ValueRange{c0}, hiddenDim);

      b0.create<scf::ForOp>(l0, c0, batchDim, c1, ValueRange{}, [&](OpBuilder &bb, Location bl, Value batchIdx, ValueRange) {
        bb.create<scf::ForOp>(bl, c0, sequenceDim, tileRows, ValueRange{}, [&](OpBuilder &tb, Location tl, Value rowOff, ValueRange) {
          Value remRows = tb.create<arith::SubIOp>(tl, sequenceDim, rowOff);
          Value rows = createMinIndex(tb, tl, remRows, tileRows);
          Value flatInput = createSubview3DTo2D(
              tb, tl, op.input(), batchIdx, rowOff, c0, rows, hiddenDim);
          Value tile2D = createSubview2D(tb, tl, tileL1, c0, c0, rows,
                                         hiddenDim, elemType, l1Space);
          Value tileElems = tb.create<arith::MulIOp>(tl, rows, hiddenDim);
          tb.create<memref::DmaStartOp>(tl, flatInput, ValueRange{c0, c0}, tile2D, ValueRange{c0, c0}, tileElems, tagInput, ValueRange{c0});
          tb.create<memref::DmaWaitOp>(tl, tagInput, ValueRange{c0}, tileElems);

          tb.create<scf::ForOp>(tl, c0, rows, c1, ValueRange{}, [&](OpBuilder &rb, Location rl, Value row, ValueRange) {
            auto vecType = VectorType::get({8}, elemType, {true});
            auto redType = VectorType::get({2}, elemType, {true});
            Value vlOne = rb.create<arith::ConstantIntOp>(rl, 1, 32);
            Value hiddenI32 = rb.create<arith::IndexCastOp>(rl, i32Type, hiddenDim);
            Value hiddenF32 = rb.create<arith::UIToFPOp>(rl, f32Type, hiddenI32);
            auto sumLoop = rb.create<scf::ForOp>(rl, c0, hiddenDim, vectorChunk, ValueRange{zeroF32}, [&](OpBuilder &sb, Location sl, Value col, ValueRange args) {
              Value rem = sb.create<arith::SubIOp>(sl, hiddenDim, col);
              Value vlIdx = createMinIndex(sb, sl, rem, vectorChunk);
              Value vl = sb.create<arith::IndexCastOp>(sl, i32Type, vlIdx);
              Value seedScalar = sb.create<memref::LoadOp>(sl, tileL1, ValueRange{row, col});
              seedScalar = sb.create<arith::SubFOp>(sl, seedScalar, seedScalar);
              Value seed = sb.create<spatz::VFMvVFOp>(sl, redType, sb.create<LLVM::UndefOp>(sl, redType), seedScalar, vlOne);
              Value x = sb.create<spatz::VLEOp>(sl, vecType, tileL1, ValueRange{row, col}, vl);
              Value reduced = sb.create<spatz::VFredUSumOp>(sl, redType, sb.create<LLVM::UndefOp>(sl, redType), x, seed, vl);
              Value partial = sb.create<spatz::VFMvFSOp>(sl, f32Type, reduced, vlOne);
              Value accumulated = sb.create<arith::AddFOp>(sl, args[0], partial);
              sb.create<scf::YieldOp>(sl, accumulated);
            });
            Value mean = rb.create<arith::DivFOp>(rl, sumLoop.getResult(0), hiddenF32);
            auto varianceLoop = rb.create<scf::ForOp>(rl, c0, hiddenDim, vectorChunk, ValueRange{zeroF32}, [&](OpBuilder &vb, Location vlLoc, Value col, ValueRange args) {
              Value rem = vb.create<arith::SubIOp>(vlLoc, hiddenDim, col);
              Value vlIdx = createMinIndex(vb, vlLoc, rem, vectorChunk);
              Value vl = vb.create<arith::IndexCastOp>(vlLoc, i32Type, vlIdx);
              Value seedScalar = vb.create<memref::LoadOp>(vlLoc, tileL1, ValueRange{row, col});
              seedScalar = vb.create<arith::SubFOp>(vlLoc, seedScalar, seedScalar);
              Value seed = vb.create<spatz::VFMvVFOp>(vlLoc, redType, vb.create<LLVM::UndefOp>(vlLoc, redType), seedScalar, vlOne);
              Value undef = vb.create<LLVM::UndefOp>(vlLoc, vecType);
              Value x = vb.create<spatz::VLEOp>(vlLoc, vecType, tileL1, ValueRange{row, col}, vl);
              Value centered = vb.create<spatz::VFSubVFOp>(vlLoc, vecType, undef, x, mean, vl);
              vb.create<spatz::VSEOp>(vlLoc, centered, tileL1, ValueRange{row, col}, vl);
              Value square = vb.create<spatz::VFMulVVOp>(vlLoc, vecType, undef, centered, centered, vl);
              Value reduced = vb.create<spatz::VFredUSumOp>(vlLoc, redType, vb.create<LLVM::UndefOp>(vlLoc, redType), square, seed, vl);
              Value partial = vb.create<spatz::VFMvFSOp>(vlLoc, f32Type, reduced, vlOne);
              Value accumulated = vb.create<arith::AddFOp>(vlLoc, args[0], partial);
              vb.create<scf::YieldOp>(vlLoc, accumulated);
            });
            Value variance = rb.create<arith::DivFOp>(rl, varianceLoop.getResult(0), hiddenF32);
            Value varianceEps = rb.create<arith::AddFOp>(rl, variance, epsilon);
            Value invStd = rb.create<func::CallOp>(rl, rsqrtFn, ValueRange{varianceEps}).getResult(0);
            rb.create<scf::ForOp>(rl, c0, hiddenDim, vectorChunk, ValueRange{}, [&](OpBuilder &ob, Location ol, Value col, ValueRange) {
              Value rem = ob.create<arith::SubIOp>(ol, hiddenDim, col);
              Value vlIdx = createMinIndex(ob, ol, rem, vectorChunk);
              Value vl = ob.create<arith::IndexCastOp>(ol, i32Type, vlIdx);
              Value undef = ob.create<LLVM::UndefOp>(ol, vecType);
              Value centered = ob.create<spatz::VLEOp>(ol, vecType, tileL1, ValueRange{row, col}, vl);
              Value gamma = ob.create<spatz::VLEOp>(ol, vecType, gammaL1, ValueRange{col}, vl);
              Value beta = ob.create<spatz::VLEOp>(ol, vecType, betaL1, ValueRange{col}, vl);
              Value normalized = ob.create<spatz::VFMulVFOp>(ol, vecType, undef, centered, invStd, vl);
              Value scaled = ob.create<spatz::VFMulVVOp>(ol, vecType, undef, normalized, gamma, vl);
              Value result = ob.create<spatz::VFAddVVOp>(ol, vecType, undef, scaled, beta, vl);
              ob.create<spatz::VSEOp>(ol, result, tileL1, ValueRange{row, col}, vl);
              ob.create<scf::YieldOp>(ol);
            });
            rb.create<scf::YieldOp>(rl);
          });

          Value flatOutput = createSubview3DTo2D(
              tb, tl, op.output(), batchIdx, rowOff, c0, rows, hiddenDim);
          tb.create<memref::DmaStartOp>(tl, tile2D, ValueRange{c0, c0}, flatOutput, ValueRange{c0, c0}, tileElems, tagOutput, ValueRange{c0});
          tb.create<memref::DmaWaitOp>(tl, tagOutput, ValueRange{c0}, tileElems);
          tb.create<scf::YieldOp>(tl);
        });
        bb.create<scf::YieldOp>(bl);
      });

      b0.create<func::CallOp>(l0, l1ResetFn, ValueRange{});
      b0.create<scf::YieldOp>(l0);
    });

    rewriter.create<func::CallOp>(loc, barrierFn, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixAddLowering : public OpRewritePattern<spatz::MatrixAddOp> {
  using OpRewritePattern<spatz::MatrixAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixAddOp op, PatternRewriter &rewriter) const override {
    auto accType = op.accMatrix().getType().dyn_cast<MemRefType>();
    if (!accType) return failure();

    auto elemType = accType.getElementType();
    Type computeElemType = elemType;
    bool isFloat = false;

    if (auto dtAttr = op->getAttrOfType<IntegerAttr>("dataType")) {
      int32_t dtCode = dtAttr.getInt();
      if (dtCode == 5) { computeElemType = rewriter.getF16Type(); isFloat = true; }
      else if (dtCode == 6) { computeElemType = rewriter.getF32Type(); isFloat = true; }
      else if (dtCode == 13) { computeElemType = rewriter.getBF16Type(); isFloat = true; }
    } else {
      if (elemType.isa<FloatType>()) isFloat = true;
    }

    auto vecLenAttr = op->getAttrOfType<IntegerAttr>("vecLen");
    if (!vecLenAttr || vecLenAttr.getInt() <= 0) return failure();
    
    int64_t baseElements = 8;
    auto memVectorType = VectorType::get({baseElements}, elemType, {true});
    auto computeVectorType = VectorType::get({baseElements}, computeElemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Type i32Type = rewriter.getI32Type();
    Value vlI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.cols());

    rewriter.create<scf::ForOp>(loc, c0, op.rows(), c1, ValueRange{}, [&](OpBuilder &builder, Location bodyLoc, Value rowIdx, ValueRange) {
          Value indices[] = {rowIdx, c0};
          Value accVec = builder.create<spatz::VLEOp>(bodyLoc, memVectorType, op.accMatrix(), indices, vlI32);
          Value tmpVec = builder.create<spatz::VLEOp>(bodyLoc, memVectorType, op.tmpMatrix(), indices, vlI32);
          
          Value accComputeVec = accVec;
          Value tmpComputeVec = tmpVec;
          if (memVectorType != computeVectorType) {
            accComputeVec = builder.create<arith::BitcastOp>(bodyLoc, computeVectorType, accVec);
            tmpComputeVec = builder.create<arith::BitcastOp>(bodyLoc, computeVectorType, tmpVec);
          }

          Value undefVec = builder.create<LLVM::UndefOp>(bodyLoc, computeVectorType);
          Value sumComputeVec;
          if (isFloat) sumComputeVec = builder.create<spatz::VFAddVVOp>(bodyLoc, computeVectorType, undefVec, accComputeVec, tmpComputeVec, vlI32);
          else sumComputeVec = builder.create<spatz::VAddVVOp>(bodyLoc, computeVectorType, undefVec, accComputeVec, tmpComputeVec, vlI32);

          Value sumMemVec = sumComputeVec;
          if (memVectorType != computeVectorType) sumMemVec = builder.create<arith::BitcastOp>(bodyLoc, memVectorType, sumComputeVec);

          builder.create<spatz::VSEOp>(bodyLoc, sumMemVec, op.accMatrix(), indices, vlI32);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixVectorAddLowering : public OpRewritePattern<spatz::MatrixVectorAddOp> {
  using OpRewritePattern<spatz::MatrixVectorAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixVectorAddOp op, PatternRewriter &rewriter) const override {
    auto accType = op.accMatrix().getType().dyn_cast<MemRefType>();
    if (!accType) return failure();

    auto elemType = accType.getElementType();
    Type computeElemType = elemType;
    bool isFloat = false;

    if (auto dtAttr = op->getAttrOfType<IntegerAttr>("dataType")) {
      int32_t dtCode = dtAttr.getInt();
      if (dtCode == 5) { computeElemType = rewriter.getF16Type(); isFloat = true; }
      else if (dtCode == 6) { computeElemType = rewriter.getF32Type(); isFloat = true; }
      else if (dtCode == 13) { computeElemType = rewriter.getBF16Type(); isFloat = true; }
    } else {
      if (elemType.isa<FloatType>()) isFloat = true;
    }

    auto vecLenAttr = op->getAttrOfType<IntegerAttr>("vecLen");
    if (!vecLenAttr || vecLenAttr.getInt() <= 0) return failure();
    
    int64_t baseElements = 8;
    auto memVectorType = VectorType::get({baseElements}, elemType, {true});
    auto computeVectorType = VectorType::get({baseElements}, computeElemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Type i32Type = rewriter.getI32Type();
    Value vlI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.cols());

    rewriter.create<scf::ForOp>(loc, c0, op.rows(), c1, ValueRange{}, [&](OpBuilder &builder, Location bodyLoc, Value rowIdx, ValueRange) {
          Value matIndices[] = {rowIdx, c0};
          Value vecIndices[] = {c0};
          
          Value accVec = builder.create<spatz::VLEOp>(bodyLoc, memVectorType, op.accMatrix(), matIndices, vlI32);
          Value addVec = builder.create<spatz::VLEOp>(bodyLoc, memVectorType, op.addVector(), vecIndices, vlI32);
          
          Value accComputeVec = accVec;
          Value addComputeVec = addVec;
          if (memVectorType != computeVectorType) {
            accComputeVec = builder.create<arith::BitcastOp>(bodyLoc, computeVectorType, accVec);
            addComputeVec = builder.create<arith::BitcastOp>(bodyLoc, computeVectorType, addVec);
          }

          Value undefVec = builder.create<LLVM::UndefOp>(bodyLoc, computeVectorType);
          Value sumComputeVec;
          if (isFloat) sumComputeVec = builder.create<spatz::VFAddVVOp>(bodyLoc, computeVectorType, undefVec, accComputeVec, addComputeVec, vlI32);
          else sumComputeVec = builder.create<spatz::VAddVVOp>(bodyLoc, computeVectorType, undefVec, accComputeVec, addComputeVec, vlI32);

          Value sumMemVec = sumComputeVec;
          if (memVectorType != computeVectorType) sumMemVec = builder.create<arith::BitcastOp>(bodyLoc, memVectorType, sumComputeVec);

          builder.create<spatz::VSEOp>(bodyLoc, sumMemVec, op.accMatrix(), matIndices, vlI32);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixScalarMulLowering : public OpRewritePattern<spatz::MatrixScalarMulOp> {
  using OpRewritePattern<spatz::MatrixScalarMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixScalarMulOp op, PatternRewriter &rewriter) const override {
    auto accType = op.accMatrix().getType().dyn_cast<MemRefType>();
    if (!accType) return failure();

    auto elemType = accType.getElementType();
    Type computeElemType = elemType;
    bool isFloat = false;

    if (auto dtAttr = op->getAttrOfType<IntegerAttr>("dataType")) {
      int32_t dtCode = dtAttr.getInt();
      if (dtCode == 5) { computeElemType = rewriter.getF16Type(); isFloat = true; }
      else if (dtCode == 6) { computeElemType = rewriter.getF32Type(); isFloat = true; }
      else if (dtCode == 13) { computeElemType = rewriter.getBF16Type(); isFloat = true; }
    } else {
      if (elemType.isa<FloatType>()) isFloat = true;
    }

    auto vecLenAttr = op->getAttrOfType<IntegerAttr>("vecLen");
    if (!vecLenAttr || vecLenAttr.getInt() <= 0) return failure();
    
    int64_t baseElements = 8;
    auto memVectorType = VectorType::get({baseElements}, elemType, {true});
    auto computeVectorType = VectorType::get({baseElements}, computeElemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Type i32Type = rewriter.getI32Type();
    Value vlI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.cols());

    rewriter.create<scf::ForOp>(loc, c0, op.rows(), c1, ValueRange{}, [&](OpBuilder &builder, Location bodyLoc, Value rowIdx, ValueRange) {
          Value matIndices[] = {rowIdx, c0};
          
          Value accVec = builder.create<spatz::VLEOp>(bodyLoc, memVectorType, op.accMatrix(), matIndices, vlI32);
          
          Value accComputeVec = accVec;
          if (memVectorType != computeVectorType) {
            accComputeVec = builder.create<arith::BitcastOp>(bodyLoc, computeVectorType, accVec);
          }

          Value undefVec = builder.create<LLVM::UndefOp>(bodyLoc, computeVectorType);
          Value mulComputeVec;
          if (isFloat) mulComputeVec = builder.create<spatz::VFMulVFOp>(bodyLoc, computeVectorType, undefVec, accComputeVec, op.scalar(), vlI32);
          else mulComputeVec = builder.create<spatz::VMulVFOp>(bodyLoc, computeVectorType, undefVec, accComputeVec, op.scalar(), vlI32);

          Value mulMemVec = mulComputeVec;
          if (memVectorType != computeVectorType) mulMemVec = builder.create<arith::BitcastOp>(bodyLoc, memVectorType, mulComputeVec);

          builder.create<spatz::VSEOp>(bodyLoc, mulMemVec, op.accMatrix(), matIndices, vlI32);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixSubLowering : public OpRewritePattern<spatz::MatrixSubOp> {
  using OpRewritePattern<spatz::MatrixSubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixSubOp op, PatternRewriter &rewriter) const override {
    auto accType = op.accMatrix().getType().dyn_cast<MemRefType>();
    if (!accType) return failure();

    auto elemType = accType.getElementType();
    Type computeElemType = elemType;
    bool isFloat = false;

    if (auto dtAttr = op->getAttrOfType<IntegerAttr>("dataType")) {
      int32_t dtCode = dtAttr.getInt();
      if (dtCode == 5) { computeElemType = rewriter.getF16Type(); isFloat = true; }
      else if (dtCode == 6) { computeElemType = rewriter.getF32Type(); isFloat = true; }
      else if (dtCode == 13) { computeElemType = rewriter.getBF16Type(); isFloat = true; }
    } else {
      if (elemType.isa<FloatType>()) isFloat = true;
    }

    auto vecLenAttr = op->getAttrOfType<IntegerAttr>("vecLen");
    if (!vecLenAttr || vecLenAttr.getInt() <= 0) return failure();
    
    int64_t baseElements = 8;
    auto memVectorType = VectorType::get({baseElements}, elemType, {true});
    auto computeVectorType = VectorType::get({baseElements}, computeElemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Type i32Type = rewriter.getI32Type();
    Value vlI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.cols());

    rewriter.create<scf::ForOp>(loc, c0, op.rows(), c1, ValueRange{}, [&](OpBuilder &builder, Location bodyLoc, Value rowIdx, ValueRange) {
          Value indices[] = {rowIdx, c0};
          Value accVec = builder.create<spatz::VLEOp>(bodyLoc, memVectorType, op.accMatrix(), indices, vlI32);
          Value tmpVec = builder.create<spatz::VLEOp>(bodyLoc, memVectorType, op.tmpMatrix(), indices, vlI32);
          
          Value accComputeVec = accVec;
          Value tmpComputeVec = tmpVec;
          if (memVectorType != computeVectorType) {
            accComputeVec = builder.create<arith::BitcastOp>(bodyLoc, computeVectorType, accVec);
            tmpComputeVec = builder.create<arith::BitcastOp>(bodyLoc, computeVectorType, tmpVec);
          }

          Value undefVec = builder.create<LLVM::UndefOp>(bodyLoc, computeVectorType);
          Value subComputeVec;
          
          if (isFloat) subComputeVec = builder.create<spatz::VFSubVVOp>(bodyLoc, computeVectorType, undefVec, accComputeVec, tmpComputeVec, vlI32);
          else return;

          Value subMemVec = subComputeVec;
          if (memVectorType != computeVectorType) subMemVec = builder.create<arith::BitcastOp>(bodyLoc, memVectorType, subComputeVec);

          builder.create<spatz::VSEOp>(bodyLoc, subMemVec, op.accMatrix(), indices, vlI32);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixExpReduceLowering : public OpRewritePattern<spatz::MatrixExpReduceOp> {
  using OpRewritePattern<spatz::MatrixExpReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixExpReduceOp op, PatternRewriter &rewriter) const override {
    auto sType = op.s_buf().getType().dyn_cast<MemRefType>();
    if (!sType || !sType.getElementType().isa<FloatType>()) return failure();

    auto elemType = sType.getElementType();
    int64_t baseElements = 8;
    auto vecType = VectorType::get({baseElements}, elemType, {true});
    auto intVecType = VectorType::get({baseElements}, rewriter.getI32Type(), {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Type i32Type = rewriter.getI32Type();
    
    Value vlI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.mE());
    Value zeroIdx[] = {c0};

    auto singleElemMemType = MemRefType::get({1}, elemType);
    Value memB = rewriter.create<memref::AllocaOp>(loc, singleElemMemType);
    Value memC = rewriter.create<memref::AllocaOp>(loc, singleElemMemType);
    
    Value constB = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1064866805.0f));
    Value constC = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 12102203.0f));
    
    rewriter.create<memref::StoreOp>(loc, constB, memB, zeroIdx);
    rewriter.create<memref::StoreOp>(loc, constC, memC, zeroIdx);

    Value zeroFloat = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0f));
    rewriter.create<scf::ForOp>(loc, c0, op.mE(), c1, ValueRange{}, 
        [&](OpBuilder &builder, Location bodyLoc, Value colIdx, ValueRange loopArgs) {
          Value storeIndices[] = {colIdx};
          builder.create<memref::StoreOp>(bodyLoc, zeroFloat, op.sum_arr(), storeIndices);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    Value initialSumVec = rewriter.create<spatz::VLEOp>(loc, vecType, op.sum_arr(), zeroIdx, vlI32);
    Value mNewVec = rewriter.create<spatz::VLEOp>(loc, vecType, op.m_val(), zeroIdx, vlI32);

    auto forOp = rewriter.create<scf::ForOp>(loc, c0, op.nE(), c1, ValueRange{initialSumVec}, [&](OpBuilder &builder, Location bodyLoc, Value rIdx, ValueRange loopArgs) {
          
          Value currentSumVec = loopArgs[0];

          Value scalarB = builder.create<memref::LoadOp>(bodyLoc, memB, zeroIdx);
          Value scalarC = builder.create<memref::LoadOp>(bodyLoc, memC, zeroIdx);
          Value bVec = builder.create<spatz::VFMvVFOp>(bodyLoc, vecType, builder.create<LLVM::UndefOp>(bodyLoc, vecType), scalarB, vlI32);

          Value matrixIndices[] = {rIdx, c0};
          Value sVec = builder.create<spatz::VLEOp>(bodyLoc, vecType, op.s_buf(), matrixIndices, vlI32);

          if (op.bias()) {
              Value biasScalar = builder.create<memref::LoadOp>(bodyLoc, op.bias(), ValueRange{rIdx});
              Value undef = builder.create<LLVM::UndefOp>(bodyLoc, vecType);
              Value biasVec = builder.create<spatz::VFMvVFOp>(bodyLoc, vecType, undef, biasScalar, vlI32);
              sVec = builder.create<spatz::VFAddVVOp>(bodyLoc, vecType, undef, sVec, biasVec, vlI32);
          }

          Value undefVec = builder.create<LLVM::UndefOp>(bodyLoc, vecType);
          Value subVec = builder.create<spatz::VFSubVVOp>(bodyLoc, vecType, undefVec, sVec, mNewVec, vlI32);
          Value expFloat = builder.create<spatz::VFMaccVFOp>(bodyLoc, vecType, bVec, subVec, scalarC, vlI32);
          Value intVec = builder.create<spatz::VFCvtRtzXUFVOp>(bodyLoc, intVecType, builder.create<LLVM::UndefOp>(bodyLoc, intVecType), expFloat, vlI32);
          Value expVec = builder.create<spatz::VBitcastOp>(bodyLoc, vecType, intVec);

          builder.create<spatz::VSEOp>(bodyLoc, expVec, op.p_T(), matrixIndices, vlI32);

          Value newSumVec = builder.create<spatz::VFAddVVOp>(bodyLoc, vecType, undefVec, currentSumVec, expVec, vlI32);

          builder.create<scf::YieldOp>(bodyLoc, ValueRange{newSumVec});
        });

    Value finalSumVec = forOp.getResult(0);
    rewriter.create<spatz::VSEOp>(loc, finalSumVec, op.sum_arr(), zeroIdx, vlI32);

    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixColumnMaxLowering : public OpRewritePattern<spatz::MatrixColumnMaxOp> {
  using OpRewritePattern<spatz::MatrixColumnMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixColumnMaxOp op, PatternRewriter &rewriter) const override {
    auto matrixType = op.matrix().getType().dyn_cast<MemRefType>();
    if (!matrixType || !matrixType.getElementType().isa<FloatType>()) return failure();

    auto elemType = matrixType.getElementType();
    
    int64_t baseElements = 8;
    auto computeVectorType = VectorType::get({baseElements}, elemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Type i32Type = rewriter.getI32Type();
    Value vlI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.cols());

    Value negInf = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getLargest(elemType.cast<FloatType>().getFloatSemantics(), true), elemType.cast<FloatType>());

    rewriter.create<scf::ForOp>(loc, c0, op.cols(), c1, ValueRange{}, 
        [&](OpBuilder &builder, Location bodyLoc, Value colIdx, ValueRange loopArgs) {
          Value storeIndices[] = {colIdx};
          builder.create<memref::StoreOp>(bodyLoc, negInf, op.maxVector(), storeIndices);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    Value zeroIndex[] = {c0};
    
    Value initialMaxVec = rewriter.create<spatz::VLEOp>(loc, computeVectorType, op.maxVector(), zeroIndex, vlI32);

    auto forOp = rewriter.create<scf::ForOp>(loc, c0, op.rows(), c1, ValueRange{initialMaxVec}, 
        [&](OpBuilder &builder, Location bodyLoc, Value rowIdx, ValueRange loopArgs) {
          
          Value currentMaxVec = loopArgs[0];
          Value matrixIndices[] = {rowIdx, c0};
          Value rowVec = builder.create<spatz::VLEOp>(bodyLoc, computeVectorType, op.matrix(), matrixIndices, vlI32);
          Value undefVec = builder.create<LLVM::UndefOp>(bodyLoc, computeVectorType);
          Value newMaxVec = builder.create<spatz::VFMaxVVOp>(bodyLoc, computeVectorType, undefVec, currentMaxVec, rowVec, vlI32);
          builder.create<scf::YieldOp>(bodyLoc, newMaxVec);

        });

    Value finalMaxVec = forOp.getResult(0);
    rewriter.create<spatz::VSEOp>(loc, finalMaxVec, op.maxVector(), zeroIndex, vlI32);

    rewriter.eraseOp(op);
    return success();
  }
};

struct VectorMaxUpdateLowering : public OpRewritePattern<spatz::VectorMaxUpdateOp> {
  using OpRewritePattern<spatz::VectorMaxUpdateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::VectorMaxUpdateOp op, PatternRewriter &rewriter) const override {
    auto elemType = op.vec_prev().getType().cast<MemRefType>().getElementType();
    if (!elemType.isa<FloatType>()) return failure();

    int64_t baseElements = 16;
    auto vecType = VectorType::get({baseElements}, elemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, baseElements);
    Value len = op.len();

    rewriter.create<scf::ForOp>(loc, c0, len, step, ValueRange{}, 
      [&](OpBuilder &builder, Location bodyLoc, Value idx, ValueRange) {
        
        Value rem = builder.create<arith::SubIOp>(bodyLoc, len, idx);
        Value cond = builder.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::slt, rem, step);
        Value vlIndex = builder.create<arith::SelectOp>(bodyLoc, cond, rem, step);
        Value vl = builder.create<arith::IndexCastOp>(bodyLoc, rewriter.getI32Type(), vlIndex);

        Value indices[] = {idx};
        Value vPrev = builder.create<spatz::VLEOp>(bodyLoc, vecType, op.vec_prev(), indices, vl);
        Value vNew = builder.create<spatz::VLEOp>(bodyLoc, vecType, op.vec_new(), indices, vl);
        
        Value undef = builder.create<LLVM::UndefOp>(bodyLoc, vecType);
        
        Value vMax = builder.create<spatz::VFMaxVVOp>(bodyLoc, vecType, undef, vPrev, vNew, vl);
        
        Value vDiff = builder.create<spatz::VFSubVVOp>(bodyLoc, vecType, undef, vPrev, vMax, vl);

        builder.create<spatz::VSEOp>(bodyLoc, vMax, op.vec_prev(), indices, vl);
        builder.create<spatz::VSEOp>(bodyLoc, vDiff, op.vec_diff(), indices, vl);
        builder.create<scf::YieldOp>(bodyLoc);
    });

    rewriter.eraseOp(op);
    return success();
  }
};

struct VectorExpLowering : public OpRewritePattern<spatz::VectorExpOp> {
  using OpRewritePattern<spatz::VectorExpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::VectorExpOp op, PatternRewriter &rewriter) const override {
    
    auto inputType = op.vec_in().getType().dyn_cast<MemRefType>();
    auto outputType = op.vec_out().getType().dyn_cast<MemRefType>();

    if (!inputType || !outputType) return failure();

    Type elementType = inputType.getElementType();

    if (!elementType.isF32() || outputType.getElementType() != elementType) return failure();

    Location loc = op.getLoc();

    FloatType f32Type = rewriter.getF32Type();
    IntegerType i32Type = rewriter.getI32Type();

    constexpr int64_t scalableElements = 2;
    constexpr int64_t elementsPerIteration = 16;

    auto floatVecType = VectorType::get({scalableElements}, f32Type, {true});
    auto integerVecType = VectorType::get({scalableElements}, i32Type, {true});
    
    Value begin = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, elementsPerIteration);
    Value fullVL = rewriter.create<arith::ConstantIntOp>(loc, elementsPerIteration, 32);

    Value lowerBoundConstant = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(-87.0f), f32Type);
    Value zeroConstant = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), f32Type);
    Value approximationBConstant = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(1064866805.0f), f32Type);
    Value approximationCConstant = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(12102203.0f), f32Type);

    Value len = op.len();

    rewriter.create<scf::ForOp>( loc,begin,len,step,ValueRange{},
        [&](OpBuilder &builder,Location bodyLoc,Value idx,ValueRange) {
          
            Value remaining = builder.create<arith::SubIOp>(bodyLoc, len, idx);
            Value hasTail = builder.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::slt, remaining, step);
            Value remainingI32 = builder.create<arith::IndexCastOp>(bodyLoc, i32Type, remaining);
            Value vl = builder.create<arith::SelectOp>(bodyLoc, hasTail, remainingI32, fullVL);

            SmallVector<Value, 1> indices = {idx};

            Value undefFloat = builder.create<LLVM::UndefOp>(bodyLoc, floatVecType);
            Value undefInteger = builder.create<LLVM::UndefOp>(bodyLoc, integerVecType);

            Value lenI32 = builder.create<arith::IndexCastOp>(bodyLoc, i32Type, len);
            Value lenF32 = builder.create<arith::SIToFPOp>(bodyLoc, f32Type, lenI32);
            Value runtimeZero = builder.create<arith::SubFOp>(bodyLoc, lenF32, lenF32);

            Value localLowerBound = builder.create<arith::AddFOp>(bodyLoc, lowerBoundConstant, runtimeZero);
            Value localZero = builder.create<arith::AddFOp>(bodyLoc, zeroConstant, runtimeZero);
            Value localApproximationB = builder.create<arith::AddFOp>(bodyLoc, approximationBConstant, runtimeZero);
            Value localApproximationC = builder.create<arith::AddFOp>(bodyLoc, approximationCConstant, runtimeZero);

            Value x = builder.create<spatz::VLEOp>(bodyLoc, floatVecType, op.vec_in(), indices, vl);

            Value lowerBoundVec = builder.create<spatz::VFMvVFOp>(bodyLoc, floatVecType, undefFloat, localLowerBound, vl);
            Value zeroVec = builder.create<spatz::VFMvVFOp>(bodyLoc, floatVecType, undefFloat, localZero, vl);
            Value bVec = builder.create<spatz::VFMvVFOp>(bodyLoc, floatVecType, undefFloat, localApproximationB, vl);

            Value xLower = builder.create<spatz::VFMaxVVOp>(bodyLoc, floatVecType, x, x, lowerBoundVec, vl);
            Value negX = builder.create<spatz::VFSubVVOp>(bodyLoc, floatVecType, zeroVec, zeroVec, xLower, vl);
            Value negPositive = builder.create<spatz::VFMaxVVOp>(bodyLoc, floatVecType, negX, negX, zeroVec, vl);
            Value xSafe = builder.create<spatz::VFSubVVOp>(bodyLoc, floatVecType, zeroVec, zeroVec, negPositive, vl);

            Value approximateBitsFloat = builder.create<spatz::VFMaccVFOp>(bodyLoc, floatVecType, bVec, xSafe, localApproximationC, vl);
            Value approximateBitsInteger = builder.create<spatz::VFCvtRtzXUFVOp>(bodyLoc, integerVecType, undefInteger, approximateBitsFloat, vl);

            Value approximateExp = builder.create<spatz::VBitcastOp>(bodyLoc, floatVecType, approximateBitsInteger);

            builder.create<spatz::VSEOp>(bodyLoc, approximateExp, op.vec_out(), indices, vl);

            builder.create<scf::YieldOp>(bodyLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpTy, bool IsAdd>
struct VectorElementwiseLowering : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    auto elemType = op.vec_a().getType().template cast<MemRefType>().getElementType();
    int64_t baseElements = 16;
    auto vecType = VectorType::get({baseElements}, elemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, baseElements);
    Value len = op.len();

    rewriter.create<scf::ForOp>(loc, c0, len, step, ValueRange{}, 
      [&](OpBuilder &builder, Location bodyLoc, Value idx, ValueRange) {
        
        Value rem = builder.create<arith::SubIOp>(bodyLoc, len, idx);
        Value cond = builder.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::slt, rem, step);
        Value vlIndex = builder.create<arith::SelectOp>(bodyLoc, cond, rem, step);
        Value vl = builder.create<arith::IndexCastOp>(bodyLoc, rewriter.getI32Type(), vlIndex);

        Value indices[] = {idx};
        Value vA = builder.create<spatz::VLEOp>(bodyLoc, vecType, op.vec_a(), indices, vl);
        Value vB = builder.create<spatz::VLEOp>(bodyLoc, vecType, op.vec_b(), indices, vl);
        Value undef = builder.create<LLVM::UndefOp>(bodyLoc, vecType);
        
        Value vRes;
        if constexpr (IsAdd) {
            vRes = builder.create<spatz::VFAddVVOp>(bodyLoc, vecType, undef, vA, vB, vl);
        } else {
            vRes = builder.create<spatz::VFMulVVOp>(bodyLoc, vecType, undef, vA, vB, vl);
        }

        builder.create<spatz::VSEOp>(bodyLoc, vRes, op.vec_a(), indices, vl);
        builder.create<scf::YieldOp>(bodyLoc);
    });

    rewriter.eraseOp(op);
    return success();
  }
};
using VectorAddLowering = VectorElementwiseLowering<spatz::VectorAddOp, true>;
using VectorMulLowering = VectorElementwiseLowering<spatz::VectorMulOp, false>;

template <typename OpTy, bool IsScale>
struct MatrixRowMathLowering : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(OpTy op,
                  PatternRewriter &rewriter) const override {
    auto matrixType =
        op.matrix().getType().template dyn_cast<MemRefType>();

    if (!matrixType)
      return failure();

    Type elemType = matrixType.getElementType();

    if (!elemType.isF32())
      return failure();

    Location loc = op.getLoc();

    auto f32Type = rewriter.getF32Type();
    auto i32Type = rewriter.getI32Type();

    constexpr int64_t baseElements = 8;

    auto vecType =
        VectorType::get(
            {baseElements},
            elemType,
            {true});

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, baseElements);
    Value oneF32 = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(1.0f), f32Type);
    Value rows = op.rows();
    Value cols = op.cols();
    Value scalarArray;

    if constexpr (IsScale)
      scalarArray = op.scaleVec();
    else
      scalarArray = op.divVec();

    rewriter.create<scf::ForOp>(loc,c0,rows,c1,ValueRange{},
        [&](OpBuilder &rowBuilder,Location rowLoc,Value rowIndex,ValueRange) {

          Value rowScalar = rowBuilder.create<memref::LoadOp>(rowLoc,scalarArray,ValueRange{rowIndex});
          Value rowMultiplier;

          if constexpr (IsScale) {
            rowMultiplier = rowScalar;
          } else {
            rowMultiplier =rowBuilder.create<arith::DivFOp>(rowLoc,oneF32,rowScalar);
          }

          rowBuilder.create<scf::ForOp>(rowLoc,c0,cols,step,ValueRange{},
              [&](OpBuilder &innerBuilder,Location innerLoc,Value columnIndex,ValueRange) {

                Value remaining = innerBuilder.create<arith::SubIOp>(innerLoc, cols, columnIndex);
                Value hasTail = innerBuilder.create<arith::CmpIOp>(innerLoc, arith::CmpIPredicate::slt, remaining, step);
                Value vlIndex = innerBuilder.create<arith::SelectOp>(innerLoc, hasTail, remaining, step);
                Value vl = innerBuilder.create<arith::IndexCastOp>(innerLoc, i32Type, vlIndex);

                SmallVector<Value, 2> matrixIndices = {rowIndex, columnIndex};

                Value undef = innerBuilder.create<LLVM::UndefOp>(innerLoc, vecType);

                Value matrixVector = innerBuilder.create<spatz::VLEOp>(innerLoc, vecType, op.matrix(), matrixIndices, vl);
                Value multiplierVector = innerBuilder.create<spatz::VFMvVFOp>(innerLoc, vecType, undef, rowMultiplier, vl);
                Value resultVector = innerBuilder.create<spatz::VFMulVVOp>(innerLoc, vecType, undef, matrixVector, multiplierVector, vl);

                innerBuilder.create<spatz::VSEOp>(innerLoc, resultVector, op.matrix(), matrixIndices, vl);
                innerBuilder.create<scf::YieldOp>(innerLoc);

              });

          rowBuilder.create<scf::YieldOp>(rowLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};
using MatrixRowScaleLowering = MatrixRowMathLowering<spatz::MatrixRowScaleOp, true>;
using MatrixRowDivLowering = MatrixRowMathLowering<spatz::MatrixRowDivOp, false>;


struct LowerSpatzMatrixAddPass : public LowerSpatzMatrixAddBase<LowerSpatzMatrixAddPass> {
  LowerSpatzMatrixAddPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgLayerNormFusionPattern, LayerNormLowering, MatrixAddLowering, MatrixVectorAddLowering, MatrixScalarMulLowering, MatrixSubLowering,
    MatrixExpReduceLowering, MatrixColumnMaxLowering, VectorMaxUpdateLowering, VectorExpLowering, VectorAddLowering,
    VectorMulLowering, MatrixRowScaleLowering, MatrixRowDivLowering>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createLowerSpatzMatrixAddPass() {
  return std::make_unique<LowerSpatzMatrixAddPass>();
}
} // namespace mlir