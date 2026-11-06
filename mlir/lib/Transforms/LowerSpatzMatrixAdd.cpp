#include "PassDetail.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Matchers.h"
#include <string>
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

static LogicalResult lowerF32MatrixAdd(spatz::MatrixAddOp op,
                                      PatternRewriter &b) {
  auto at = op.accMatrix().getType().cast<MemRefType>();
  auto pt = op.tmpMatrix().getType().cast<MemRefType>();
  SmallVector<int64_t> as, ps;
  int64_t ao, po;
  if (at.getRank() != 2 || pt.getRank() != 2 ||
      !at.getElementType().isF32() || !pt.getElementType().isF32() ||
      failed(getStridesAndOffset(at, as, ao)) ||
      failed(getStridesAndOffset(pt, ps, po)) || as[1] != 1 || ps[1] != 1)
    return failure();
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return failure();
  Location loc = op.getLoc();
  DataLayoutAnalysis analysis(module);
  LowerToLLVMOptions options(b.getContext(),
                             analysis.getAtOrAbove(module.getOperation()));
  LLVMTypeConverter converter(b.getContext(), options, &analysis);
  auto i32 = b.getI32Type();
  auto c32 = [&](int64_t n) -> Value {
    return b.create<LLVM::ConstantOp>(loc, i32, b.getI32IntegerAttr(n));
  };
  auto narrow = [&](Value v) -> Value {
    unsigned bits = v.getType().cast<IntegerType>().getWidth();
    if (bits > 32) return b.create<LLVM::TruncOp>(loc, i32, v);
    if (bits < 32) return b.create<LLVM::ZExtOp>(loc, i32, v);
    return v;
  };
  auto descriptor = [&](Value ref) -> Value {
    Type ty = converter.convertType(ref.getType());
    return b.create<UnrealizedConversionCastOp>(
        loc, TypeRange{ty}, ValueRange{ref}).getResult(0);
  };
  auto field = [&](Value desc, ArrayRef<int64_t> pos) -> Value {
    auto fields = desc.getType().cast<LLVM::LLVMStructType>().getBody();
    Type ty = fields[pos[0]];
    if (pos.size() == 2) ty = ty.cast<LLVM::LLVMArrayType>().getElementType();
    return b.create<LLVM::ExtractValueOp>(loc, ty, desc, b.getI64ArrayAttr(pos));
  };
  auto address = [&](Value desc) -> Value {
    Value ptr = field(desc, {1});
    Value offset = narrow(field(desc, {2}));
    Value bytes = b.create<LLVM::MulOp>(loc, offset, c32(4));
    Value base = b.create<LLVM::PtrToIntOp>(loc, i32, ptr);
    return b.create<LLVM::AddOp>(loc, base, bytes);
  };
  Value ad = descriptor(op.accMatrix()), pd = descriptor(op.tmpMatrix());
  Value ap = address(ad), pp = address(pd);
  auto pitch = [&](Value desc, int64_t stride) -> Value {
    if (!ShapedType::isDynamicStrideOrOffset(stride)) return c32(stride * 4);
    return b.create<LLVM::MulOp>(loc, narrow(field(desc, {4, 0})), c32(4));
  };
  Value ab = pitch(ad, as[0]), pb = pitch(pd, ps[0]);
  APInt rowConstant, colConstant;
  bool staticRows = matchPattern(op.rows(), m_ConstantInt(&rowConstant));
  bool staticCols = matchPattern(op.cols(), m_ConstantInt(&colConstant));

  bool flat = staticRows && staticCols && rowConstant.isNonNegative() &&
      colConstant.isNonNegative() && rowConstant.getLimitedValue() <= 65535 &&
      colConstant.getLimitedValue() <= 65535 &&
      as[0] == colConstant.getSExtValue() && ps[0] == as[0];
  Value rows = b.create<arith::IndexCastOp>(loc, i32, op.rows());
  Value cols = b.create<arith::IndexCastOp>(loc, i32, op.cols());
  if (flat) {
    rows = c32(1);
    cols = c32(rowConstant.getSExtValue() * colConstant.getSExtValue());
  }
  std::string code =
      "mv t0, $0\n\tmv t1, $1\n\tmv t2, $2\n\t"
      "csrr zero, fcsr\n\tbeqz t2, 9f\n\tbeqz $3, 9f\n\t"

      "vsetvli t4, $3, e32, m2, ta, ma\n\tbne t4, $3, 4f\n\t"
      "1:\n\tvle32.v v8, (t0)\n\tcsrr zero, fcsr\n\t"
      "vle32.v v10, (t1)\n\tcsrr zero, fcsr\n\t"
      "vfadd.vv v8, v8, v10\n\tvse32.v v8, (t0)\n\tcsrr zero, fcsr\n\t"
      "add t0, t0, $4\n\tadd t1, t1, $5\n\t"
      "addi t2, t2, -1\n\tbnez t2, 1b\n\tj 9f\n\t"

      "4:\n\tmv t3, $3\n\t"
      "5:\n\tvsetvli t4, t3, e32, m2, ta, ma\n\t"
      "vle32.v v8, (t0)\n\tcsrr zero, fcsr\n\t"
      "vle32.v v10, (t1)\n\tcsrr zero, fcsr\n\t"
      "vfadd.vv v8, v8, v10\n\tvse32.v v8, (t0)\n\tcsrr zero, fcsr\n\t"
      "slli t5, t4, 2\n\tadd t0, t0, t5\n\tadd t1, t1, t5\n\t"
      "sub t3, t3, t4\n\tbnez t3, 5b\n\t"
      "slli t5, $3, 2\n\tsub t3, $4, t5\n\tadd t0, t0, t3\n\t"
      "sub t3, $5, t5\n\tadd t1, t1, t3\n\t"
      "addi t2, t2, -1\n\tbnez t2, 4b\n\t"
      "9:\n\tcsrr zero, fcsr\n\t";
  std::string constraints = "r,r,r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},";
  for (int reg = 8; reg < 24; ++reg)
    constraints += "~{v" + std::to_string(reg) + "},";
  constraints += "~{vl},~{vtype},~{memory}";
  OperationState state(loc, LLVM::InlineAsmOp::getOperationName());
  state.addOperands(SmallVector<Value, 6>{ap, pp, rows, cols, ab, pb});
  state.addAttribute("asm_string", b.getStringAttr(code));
  state.addAttribute("constraints", b.getStringAttr(constraints));
  state.addAttribute("has_side_effects", b.getUnitAttr());
  b.create(state);
  b.eraseOp(op);
  return success();
}

struct MatrixAddLowering : public OpRewritePattern<spatz::MatrixAddOp> {
  using OpRewritePattern<spatz::MatrixAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixAddOp op, PatternRewriter &rewriter) const override {
    auto accType = op.accMatrix().getType().dyn_cast<MemRefType>();
    if (!accType) return failure();
    auto dt = op->getAttrOfType<IntegerAttr>("dataType");
    if (accType.getElementType().isF32() && dt && dt.getInt() == 6 &&
        succeeded(lowerF32MatrixAdd(op, rewriter)))
      return success();

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

    Type i32Type = rewriter.getI32Type();
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value oneI32 = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value rowsI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.rows());
    Value vlI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.cols());

    auto loop = rewriter.create<scf::WhileOp>(
        loc, TypeRange{i32Type}, ValueRange{zeroI32});

    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *before = rewriter.createBlock(
          &loop.getBefore(), loop.getBefore().end(), TypeRange{i32Type},
          SmallVector<Location>{loc});
      rewriter.setInsertionPointToEnd(before);
      Value rowI32 = before->getArgument(0);
      Value keepGoing = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, rowI32, rowsI32);
      rewriter.create<scf::ConditionOp>(
          loc, keepGoing, ValueRange{rowI32});

      Block *after = rewriter.createBlock(
          &loop.getAfter(), loop.getAfter().end(), TypeRange{i32Type},
          SmallVector<Location>{loc});
      rewriter.setInsertionPointToEnd(after);
      rowI32 = after->getArgument(0);
      Value rowIndex = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), rowI32);
      Value indices[] = {rowIndex, zeroIndex};

      Value accVec = rewriter.create<spatz::VLEOp>(
          loc, memVectorType, op.accMatrix(), indices, vlI32);
      Value tmpVec = rewriter.create<spatz::VLEOp>(
          loc, memVectorType, op.tmpMatrix(), indices, vlI32);

      Value accComputeVec = accVec;
      Value tmpComputeVec = tmpVec;
      if (memVectorType != computeVectorType) {
        accComputeVec = rewriter.create<arith::BitcastOp>(
            loc, computeVectorType, accVec);
        tmpComputeVec = rewriter.create<arith::BitcastOp>(
            loc, computeVectorType, tmpVec);
      }

      Value undefVec = rewriter.create<LLVM::UndefOp>(loc, computeVectorType);
      Value sumComputeVec;
      if (isFloat)
        sumComputeVec = rewriter.create<spatz::VFAddVVOp>(
            loc, computeVectorType, undefVec, accComputeVec, tmpComputeVec,
            vlI32);
      else
        sumComputeVec = rewriter.create<spatz::VAddVVOp>(
            loc, computeVectorType, undefVec, accComputeVec, tmpComputeVec,
            vlI32);

      Value sumMemVec = sumComputeVec;
      if (memVectorType != computeVectorType)
        sumMemVec = rewriter.create<arith::BitcastOp>(
            loc, memVectorType, sumComputeVec);
      rewriter.create<spatz::VSEOp>(
          loc, sumMemVec, op.accMatrix(), indices, vlI32);

      Value nextRow = rewriter.create<arith::AddIOp>(
          loc, rowI32, oneI32);
      rewriter.create<scf::YieldOp>(loc, ValueRange{nextRow});
    }

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

    auto parentFn = op->getParentOfType<func::FuncOp>();
    if (!parentFn || parentFn.isExternal()) return failure();
    Value memB, memC;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&parentFn.getBody().front());
      memB = rewriter.create<memref::AllocaOp>(loc, singleElemMemType);
      memC = rewriter.create<memref::AllocaOp>(loc, singleElemMemType);
    }

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

    constexpr int64_t scalableElements = 8;
    constexpr int64_t elementsPerIteration = 8;

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

static Value getSpatzF32Address(OpBuilder &ib, Location loc, ModuleOp module, Value ref) {
  DataLayoutAnalysis layoutAnalysis(module);
  LowerToLLVMOptions options(ib.getContext(), layoutAnalysis.getAtOrAbove(module.getOperation()));
  LLVMTypeConverter converter(ib.getContext(), options, &layoutAnalysis);
  auto addr32 = [&](Value ref) -> Value {
    Type descTy = converter.convertType(ref.getType());
    auto fields = descTy.cast<LLVM::LLVMStructType>().getBody();
    Value desc = ib.create<UnrealizedConversionCastOp>(loc, TypeRange{descTy}, ValueRange{ref}).getResult(0);
    Value ptr = ib.create<LLVM::ExtractValueOp>(loc, fields[1], desc, ib.getI64ArrayAttr({1}));
    Value off = ib.create<LLVM::ExtractValueOp>(loc, fields[2], desc, ib.getI64ArrayAttr({2}));
    unsigned bits = off.getType().cast<IntegerType>().getWidth();
    if (bits > 32) off = ib.create<LLVM::TruncOp>(loc, ib.getI32Type(), off);
    else if (bits < 32) off = ib.create<LLVM::ZExtOp>(loc, ib.getI32Type(), off);
    Value four = ib.create<LLVM::ConstantOp>(loc, ib.getI32Type(), ib.getI32IntegerAttr(4));
    Value bytes = ib.create<LLVM::MulOp>(loc, off, four);
    Value base = ib.create<LLVM::PtrToIntOp>(loc, ib.getI32Type(), ptr);
    return ib.create<LLVM::AddOp>(loc, base, bytes);
  };
  return addr32(ref);
}

struct AttentionSoftmaxUpdateLowering : public OpRewritePattern<spatz::AttentionSoftmaxUpdateOp> {
  using OpRewritePattern<spatz::AttentionSoftmaxUpdateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::AttentionSoftmaxUpdateOp op, PatternRewriter &rewriter) const override {
    OpBuilder &ib = rewriter;
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value scorePtr = getSpatzF32Address(ib, loc, module, op.scores()), maxPtr = getSpatzF32Address(ib, loc, module, op.maxState());
    Value alphaPtr = getSpatzF32Address(ib, loc, module, op.alpha()), denPtr = getSpatzF32Address(ib, loc, module, op.denom());
    Value queries = ib.create<arith::IndexCastOp>(loc, ib.getI32Type(), op.rows());
    Value keys = ib.create<arith::IndexCastOp>(loc, ib.getI32Type(), ib.create<arith::ConstantIndexOp>(loc, op.keyRows()));
    bool fullKeys = op.keyRows() == 32;
    std::string code;
    if (fullKeys) {

      code =
          "mv t0, $0\n\tmv t1, $1\n\tmv t2, $2\n\tmv t3, $3\n\tmv t4, $4\n\t"
          "beqz t4, 5f\n\tbeqz $5, 5f\n\t"
          "li t6, 0x3e000000\n\tfmv.w.x ft0, t6\n\t"
          "li t6, 0x4b38aa3b\n\tfmv.w.x ft1, t6\n\t"
          "li t6, 0x4e7de250\n\tfmv.w.x ft2, t6\n\t"
          "li t6, 0xc2ae0000\n\tfmv.w.x ft3, t6\n\t"
          "fmv.w.x ft4, zero\n\tli t6, -8388609\n\tfmv.w.x ft5, t6\n\t"
          "1:\n\tvsetvli t5, t4, e32, m2, ta, ma\n\t"
          "li t6, -8388609\n\tfmv.w.x ft5, t6\n\tvfmv.v.f v24, ft5\n\tvfmv.v.f v28, ft5\n\tmv t6, t0\n\t";
      for (int row = 0; row < 12; ++row) {
        std::string v = "v" + std::to_string(2*row);
        std::string mx = (row & 1) ? "v28" : "v24";
        code += "vle32.v " + v + ", (t6)\n\tvfmax.vv " + mx + ", " + mx + ", " + v + "\n\taddi t6, t6, 256\n\t";

        if ((row & 1) == 1) code += "csrr s1, fcsr\n\t";
      }
      code += "li s1, 5\n\t2:\n\t";
      for (int row = 0; row < 4; ++row) {
        std::string mx = (row & 1) ? "v28" : "v24";
        code += "vle32.v v26, (t6)\n\tvfmax.vv " + mx + ", " + mx + ", v26\n\taddi t6, t6, 256\n\t";

        if ((row & 1) == 1) code += "csrr zero, fcsr\n\t";
      }
      code += "addi s1, s1, -1\n\tbnez s1, 2b\n\tvfmax.vv v24, v24, v28\n\tvfmul.vf v24, v24, ft0\n\t";
      if (!op.firstBlock())
        code += "vle32.v v28, (t1)\n\tvfmax.vv v24, v28, v24\n\t"
                "vfsub.vv v28, v28, v24\n\tvfmax.vf v28, v28, ft3\n\t"
                "vfmin.vf v28, v28, ft4\n\tvfmul.vf v28, v28, ft1\n\t"
                "vfadd.vf v28, v28, ft2\n\tvfcvt.rtz.xu.f.v v28, v28\n\t"
                "vse32.v v28, (t2)\n\t";

      code += "vse32.v v24, (t1)\n\tvmv.v.i v26, 0\n\t"
              "fmul.s ft5, ft0, ft1\n\tvfmv.v.f v30, ft2\n\t"
              "vfnmsac.vf v30, ft1, v24\n\tvmv.v.i v24, 0\n\tmv t6, t0\n\t";
      auto expRow = [&](int reg, bool odd) -> std::string {
        std::string v = "v" + std::to_string(reg);
        std::string sum = odd ? "v24" : "v26";
        return "vfmadd.vf " + v + ", ft5, v30\n\t"
               "vfcvt.rtz.xu.f.v " + v + ", " + v + "\n\t"
               "vse32.v " + v + ", (t6)\n\t"
               "vfadd.vv " + sum + ", " + sum + ", " + v + "\n\taddi t6, t6, 256\n\t";
      };

      for (int row = 0; row < 12; ++row) code += expRow(2*row, row & 1);
      code += "li s1, 5\n\t3:\n\t";
      for (int row = 0; row < 4; ++row)
        code += "vle32.v v0, (t6)\n\t" + expRow(0, row & 1);
      code += "addi s1, s1, -1\n\tbnez s1, 3b\n\tvfadd.vv v26, v26, v24\n\t";
      if (!op.firstBlock())
        code += "vle32.v v0, (t3)\n\tvfmacc.vv v26, v0, v28\n\t";
      code += "vse32.v v26, (t3)\n\tsub t4, t4, t5\n\tslli t6, t5, 2\n\t"
              "add t0, t0, t6\n\tadd t1, t1, t6\n\tadd t2, t2, t6\n\tadd t3, t3, t6\n\t"
              "bnez t4, 1b\n\t5:\n\tcsrr t6, fcsr\n\t";
    } else {
    code =
        "mv t0, $0\n\tmv t1, $1\n\tmv t2, $2\n\tmv t3, $3\n\tmv t4, $4\n\t"
        "beqz t4, 5f\n\tbeqz $5, 5f\n\t"
        "li t6, 0x3e000000\n\tfmv.w.x ft0, t6\n\t"
        "li t6, 0x4b38aa3b\n\tfmv.w.x ft1, t6\n\t"
        "li t6, 0x4e7de250\n\tfmv.w.x ft2, t6\n\t"
        "li t6, 0xc2ae0000\n\tfmv.w.x ft3, t6\n\t"
        "fmv.w.x ft4, zero\n\t"
        "li t6, -8388609\n\tfmv.w.x ft5, t6\n\t"
        "1:\n\tvsetvli t5, t4, e32, m2, ta, ma\n\t"
        "li t6, -8388609\n\tfmv.w.x ft5, t6\n\tvfmv.v.f v10, ft5\n\tmv t6, t0\n\t";
    code += fullKeys ? "li s1, 8\n\t" : "mv s1, $5\n\t";
    code += "2:\n\t";
    for (int row = 0; row < (fullKeys ? 4 : 1); ++row)
      code += "vle32.v v8, (t6)\n\tvfmax.vv v10, v10, v8\n\taddi t6, t6, 256\n\t";
    code += "addi s1, s1, -1\n\tbnez s1, 2b\n\t"
            "vfmul.vf v10, v10, ft0\n\t";
    if (!op.firstBlock()) {
      code += "vle32.v v12, (t1)\n\tvfmax.vv v10, v12, v10\n\t"
              "vfsub.vv v16, v12, v10\n\t"
              "vfmax.vf v16, v16, ft3\n\tvfmin.vf v16, v16, ft4\n\t"
              "vfmul.vf v16, v16, ft1\n\tvfadd.vf v16, v16, ft2\n\t"
              "vfcvt.rtz.xu.f.v v16, v16\n\tvse32.v v16, (t2)\n\t";
    }
    code += "vse32.v v10, (t1)\n\tvmv.v.i v12, 0\n\t"
            "fmul.s ft5, ft0, ft1\n\tvfmv.v.f v14, ft2\n\t"
            "vfnmsac.vf v14, ft1, v10\n\tmv t6, t0\n\t";
    code += fullKeys ? "li s1, 8\n\t" : "mv s1, $5\n\t";
    code += "3:\n\t";
    for (int row = 0; row < (fullKeys ? 4 : 1); ++row) {
      code += "vle32.v v8, (t6)\n\tvfmadd.vf v8, ft5, v14\n\t"
              "vfcvt.rtz.xu.f.v v8, v8\n\tvse32.v v8, (t6)\n\t"
              "vfadd.vv v12, v12, v8\n\taddi t6, t6, 256\n\t";
    }
    code += "addi s1, s1, -1\n\tbnez s1, 3b\n\t";
    if (!op.firstBlock())
      code += "vle32.v v18, (t3)\n\tvfmacc.vv v12, v18, v16\n\t";

    code += "vse32.v v12, (t3)\n\tsub t4, t4, t5\n\tslli t6, t5, 2\n\t"
            "add t0, t0, t6\n\tadd t1, t1, t6\n\tadd t2, t2, t6\n\tadd t3, t3, t6\n\t"
            "bnez t4, 1b\n\t5:\n\tcsrr t6, fcsr\n\t";
    }
    if (op.qkDim() != 64) {
      const std::string oldScale = "0x3e000000", newScale = op.qkDim() == 128 ? "0x3db504f3" : "0x3d800000";
      size_t pos = code.find(oldScale);
      assert(pos != std::string::npos && "missing fused-attention scale");
      code.replace(pos, oldScale.size(), newScale);
    }
    OperationState state(loc, LLVM::InlineAsmOp::getOperationName());
    state.addOperands(SmallVector<Value,6>{scorePtr, maxPtr, alphaPtr, denPtr, queries, keys});
    state.addAttribute("asm_string", ib.getStringAttr(code));
    std::string constraints =
        "r,r,r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{s1},"
        "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},";
    if (fullKeys) {
      for (int reg = 0; reg < 32; ++reg) constraints += "~{v" + std::to_string(reg) + "},";
    } else {
      for (int reg = 8; reg < 20; ++reg) constraints += "~{v" + std::to_string(reg) + "},";
    }
    constraints += "~{vl},~{vtype},~{memory}";
    state.addAttribute("constraints", ib.getStringAttr(constraints));
    state.addAttribute("has_side_effects", ib.getUnitAttr());
    ib.create(state);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixUpdateLowering : public OpRewritePattern<spatz::MatrixUpdateOp> {
  using OpRewritePattern<spatz::MatrixUpdateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixUpdateOp op, PatternRewriter &rewriter) const override {
    OpBuilder &ib = rewriter;
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value ap = getSpatzF32Address(ib, loc, module, op.accMatrix()), dp = getSpatzF32Address(ib, loc, module, op.source()), xp = getSpatzF32Address(ib, loc, module, op.scaleVec());
    Value rows32 = ib.create<arith::IndexCastOp>(loc, ib.getI32Type(), op.rows());

    auto emit = [&](OpBuilder &ab, bool fullTile) {
      std::string code =
          "mv t0, $0\n\tmv t1, $1\n\tmv t2, $2\n\t"
          "mv t3, $3\n\tbeqz t3, 5f\n\tli t4, 64\n\t"
          "vsetvli t5, t4, e32, m2, ta, ma\n\t"
          "bne t5, t4, 2f\n\t";
      auto data = [&]() -> std::string {
        std::string c = op.fromSource() ? "vle32.v v8, (t1)\n\t" : "vle32.v v8, (t0)\n\t";
        if (op.add()) c += "vle32.v v10, (t0)\n\tvfadd.vv v8, v10, v8\n\t";

        if (op.rescale()) c += "vfmul.vf v8, v8, ft0\n\t";
        c += "vse32.v v8, (t0)\n\t";
        return c;
      };
      if (fullTile) code += "li t3, 16\n\t";
      code += "1:\n\t";
      for (int row = 0; row < (fullTile ? 4 : 1); ++row) {
        if (op.rescale()) code += "flw ft0, " + std::to_string(fullTile ? 4*row : 0) + "(t2)\n\t";
        code += data();
        code += "addi t0, t0, 256\n\t";
        if (op.fromSource()) code += "addi t1, t1, 256\n\t";
        if (op.rescale() && !fullTile) code += "addi t2, t2, 4\n\t";
      }
      if (op.rescale() && fullTile) code += "addi t2, t2, 16\n\t";
      code += "addi t3, t3, -1\n\tbnez t3, 1b\n\tj 5f\n\t";

      code += "2:\n\t";
      if (op.rescale()) code += "flw ft0, 0(t2)\n\t";
      code += "li t4, 64\n\t3:\n\tvsetvli t5, t4, e32, m2, ta, ma\n\t";
      code += data();
      code += "slli t6, t5, 2\n\tadd t0, t0, t6\n\t";
      if (op.fromSource()) code += "add t1, t1, t6\n\t";
      code += "sub t4, t4, t5\n\tbnez t4, 3b\n\t";
      if (op.rescale()) code += "addi t2, t2, 4\n\t";
      code += "addi t3, t3, -1\n\tbnez t3, 2b\n\t5:\n\tcsrr t5, fcsr\n\t";
      OperationState state(loc, LLVM::InlineAsmOp::getOperationName());
      state.addOperands(SmallVector<Value,4>{ap, dp, xp, rows32});
      state.addAttribute("asm_string", ab.getStringAttr(code));
      state.addAttribute("constraints", ab.getStringAttr(
          "r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},"
          "~{f0},~{v8},~{v9},~{v10},~{v11},~{vl},~{vtype},~{memory}"));
      state.addAttribute("has_side_effects", ab.getUnitAttr());
      ab.create(state);
    };
    Value full = ib.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rows32, ib.create<arith::ConstantIntOp>(loc, 64, 32));
    ib.create<scf::IfOp>(loc, full,
      [&](OpBuilder &fb, Location fl) { emit(fb, true); fb.create<scf::YieldOp>(fl); },
      [&](OpBuilder &tb, Location tl) { emit(tb, false); tb.create<scf::YieldOp>(tl); });
    rewriter.eraseOp(op);
    return success();
  }
};

struct VectorReciprocalApproxLowering : public OpRewritePattern<spatz::VectorReciprocalApproxOp> {
  using OpRewritePattern<spatz::VectorReciprocalApproxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::VectorReciprocalApproxOp op, PatternRewriter &rewriter) const override {
    OpBuilder &ib = rewriter;
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value srcPtr = getSpatzF32Address(ib, loc, module, op.input()), dstPtr = getSpatzF32Address(ib, loc, module, op.output());
    Value queries = ib.create<arith::IndexCastOp>(loc, ib.getI32Type(), op.len());
    std::string code =
        "mv t0, $0\n\tmv t1, $1\n\tmv t2, $2\n\tbeqz t2, 5f\n\t"
        "li t5, 0x7ef311c3\n\tli t6, 0x40000000\n\tfmv.w.x ft0, t6\n\t"
        "1:\n\tvsetvli t3, t2, e32, m2, ta, ma\n\t"
        "vle32.v v8, (t0)\n\tvmv.v.x v12, t5\n\t"
        "vsub.vv v10, v12, v8\n\tvfmv.v.f v14, ft0\n\t";

    for (int iteration = 0; iteration < 3; ++iteration) {
      code += "vfmul.vv v16, v8, v10\n\t"
              "vfsub.vv v16, v14, v16\n\t"
              "vfmul.vv v10, v10, v16\n\t";
    }
    code += "vse32.v v10, (t1)\n\tsub t2, t2, t3\n\t"
            "slli t4, t3, 2\n\tadd t0, t0, t4\n\tadd t1, t1, t4\n\t"
            "bnez t2, 1b\n\t5:\n\tcsrr t5, fcsr\n\t";
    OperationState state(loc, LLVM::InlineAsmOp::getOperationName());
    state.addOperands(SmallVector<Value,3>{srcPtr, dstPtr, queries});
    state.addAttribute("asm_string", ib.getStringAttr(code));
    state.addAttribute("constraints", ib.getStringAttr(
        "r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{f0},"
        "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},"
        "~{v16},~{v17},~{vl},~{vtype},~{memory}"));
    state.addAttribute("has_side_effects", ib.getUnitAttr());
    ib.create(state);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerSpatzMatrixAddPass : public LowerSpatzMatrixAddBase<LowerSpatzMatrixAddPass> {
  LowerSpatzMatrixAddPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    LowerSpatzMatrixAddBase<LowerSpatzMatrixAddPass>::getDependentDialects(registry);
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<AttentionSoftmaxUpdateLowering, MatrixUpdateLowering, VectorReciprocalApproxLowering, LinalgLayerNormFusionPattern, LayerNormLowering, MatrixAddLowering, MatrixVectorAddLowering, MatrixScalarMulLowering, MatrixSubLowering,
    MatrixExpReduceLowering, MatrixColumnMaxLowering, VectorMaxUpdateLowering, VectorExpLowering, VectorAddLowering,
    VectorMulLowering, MatrixRowScaleLowering, MatrixRowDivLowering>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) signalPassFailure();
  }
};

}

namespace mlir {
std::unique_ptr<Pass> createLowerSpatzMatrixAddPass() {
  return std::make_unique<LowerSpatzMatrixAddPass>();
}
}
