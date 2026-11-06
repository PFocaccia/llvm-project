#include "PassDetail.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quadrilatero/QuadrilateroDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Spatz/SpatzDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cmath>
#include <string>
#include <cstdint>
#include <climits>
#include <vector>

using namespace mlir;

namespace {

static int32_t getDataTypeCode(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    bool isUnsigned = intType.isUnsigned();
    if (width == 8) return isUnsigned ? 8 : 0;
    if (width == 16) return isUnsigned ? 9 : 1;
    if (width == 32) return isUnsigned ? 10 : 2;
  } else if (type.isF32()) return 6;
  else if (type.isF16()) return 5;
  else if (type.isBF16()) return 13;
  return -1;
}

static Value createMinI32(OpBuilder &builder, Location loc, Value lhs, Value rhs) {

  Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
  return builder.create<arith::SelectOp>(loc, cmp, lhs, rhs);

}

static Value createSubview2D(OpBuilder &builder, Location loc, Value base, Value off0, Value off1, Value size0, Value size1,
                             Type elementType, Attribute memorySpace) {

  SmallVector<OpFoldResult, 2> offsets = {off0, off1};
  SmallVector<OpFoldResult, 2> sizes = {size0, size1};
  SmallVector<OpFoldResult, 2> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
  return builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides).getResult();

}

static Value createSubview3DTo2D(OpBuilder &builder, Location loc, Value base, Value off0, Value off1, Value off2,
                                 Value size1, Value size2) {

  SmallVector<OpFoldResult, 3> offsets = {off0, off1, off2};
  SmallVector<OpFoldResult, 3> sizes = {builder.getIndexAttr(1), size1, size2};
  SmallVector<OpFoldResult, 3> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};

  auto sourceType = base.getType().cast<MemRefType>();

  auto resultType = memref::SubViewOp::inferRankReducedResultType( 2, sourceType, offsets, sizes, strides).cast<MemRefType>();

  return builder.create<memref::SubViewOp>(loc, resultType, base, offsets, sizes, strides).getResult();
}

static Value createSubview1D(OpBuilder &builder, Location loc, Value base, Value off0, Value size0,
                             Type elementType, Attribute memorySpace) {

  SmallVector<OpFoldResult, 1> offsets = {off0};
  SmallVector<OpFoldResult, 1> sizes = {size0};
  SmallVector<OpFoldResult, 1> strides = {builder.getIndexAttr(1)};

  return builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides).getResult();

}

static void emitConvSequentialDrain(OpBuilder &b, Location loc) {
  OperationState state(loc, LLVM::InlineAsmOp::getOperationName());
  state.addAttribute("asm_string", b.getStringAttr("csrr zero, fcsr"));
  state.addAttribute("constraints", b.getStringAttr("~{memory}"));
  state.addAttribute("has_side_effects", b.getUnitAttr());
  b.create(state);
}

struct ConvDmaGroup {
  int64_t src, dst, commands, repetitions;
  int64_t srcPitch, dstPitch, srcStep, dstStep;
};

static std::vector<ConvDmaGroup> buildConvDmaPlan(
    int64_t panelK, int64_t totalK, int64_t kh, int64_t kw,
    int64_t h, int64_t w, int64_t ow, int64_t spatialRows) {
  constexpr int64_t TK = 64, PITCH = 64;
  const int64_t area = kh * kw;
  const int64_t validK = std::min<int64_t>(TK, totalK-panelK);
  std::vector<ConvDmaGroup> groups;
  for (int64_t ky = 0; ky < kh; ++ky) {
    for (int64_t kx = 0; kx < kw; ++kx) {
      const int64_t t = ky*kw+kx;
      const int64_t first = (std::max<int64_t>(panelK-t,0)+area-1)/area;
      const int64_t last = (std::max<int64_t>(panelK+validK-t,0)+area-1)/area;
      const int64_t channels = last-first;
      if (channels == 0) continue;
      const int64_t src = first*h*w+ky*w+kx;
      const int64_t dst = (first*area+t-panelK)*PITCH;

      if (channels > spatialRows)
        groups.push_back({src,dst,spatialRows,channels,h*w,area*PITCH,w,ow});
      else
        groups.push_back({src,dst,channels,spatialRows,w,ow,h*w,area*PITCH});
    }
  }
  return groups;
}

struct LowerLinalgMatmulToQuadrilateroPass : public LowerLinalgMatmulToQuadrilateroBase<LowerLinalgMatmulToQuadrilateroPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    LowerLinalgMatmulToQuadrilateroBase<
        LowerLinalgMatmulToQuadrilateroPass>::getDependentDialects(registry);
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {

    func::FuncOp funcOp = getOperation();
    if (funcOp->hasAttr("quadrilatero.fused_attention_serial") ||
        funcOp->hasAttr("quadrilatero.fused_attention_pipeline")) {
      if (failed(lowerPipelinedFusedAttention(funcOp))) signalPassFailure();
      return;
    }
    llvm::SmallPtrSet<Operation*, 8> erasedOps;

    SmallVector<linalg::BatchMatmulOp, 4> batchMatmuls;
    funcOp.walk([&](linalg::BatchMatmulOp op) { batchMatmuls.push_back(op); });

    for (linalg::BatchMatmulOp op : batchMatmuls) {
      if (erasedOps.count(op)) continue;
      if (succeeded(lowerFlashAttention(op, erasedOps))) { continue; }
    }

    SmallVector<linalg::BatchMatmulOp, 4> remainingMatmuls;
    funcOp.walk([&](linalg::BatchMatmulOp op) { remainingMatmuls.push_back(op); });

    for (linalg::BatchMatmulOp op : remainingMatmuls) {
      if (failed(lowerBatchMatmul(op))) signalPassFailure();
    }

    SmallVector<linalg::Conv2DNchwFchwOp, 4> convs;
    funcOp.walk([&](linalg::Conv2DNchwFchwOp op) { convs.push_back(op); });
    for (auto op : convs)
      if (failed(lowerConv2D(op))) { signalPassFailure(); return; }

    SmallVector<linalg::MatmulOp, 4> matmuls;
    funcOp.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    for (linalg::MatmulOp op : matmuls) {
      if (failed(lowerMatmul(op))) signalPassFailure();
    }
  }

  LogicalResult lowerPipelinedFusedAttention(func::FuncOp fn) {
    auto reject = [&](StringRef message) -> LogicalResult {
      fn.emitError(message);
      return failure();
    };
    if (fn.isExternal() || fn.getBody().getBlocks().size() != 1 ||
        fn.getNumArguments() != 3 || fn.getNumResults() != 1)
      return reject("pipelined fused attention requires three inputs and one returned memref");
    SmallVector<MemRefType, 4> types;
    for (Type t : fn.getArgumentTypes()) {
      auto mt = t.dyn_cast<MemRefType>();
      if (!mt || mt.getRank() != 3 || !mt.hasStaticShape() ||
          !(mt.getElementType().isF32() || mt.getElementType().isInteger(32)) ||
          !mt.getLayout().isIdentity() ||
          mt.getMemorySpaceAsInt() != 0)
        return reject("expected static contiguous rank-3 FP32 or packed-i32 memrefs; bufferize first");
      types.push_back(mt);
    }
    auto ot = fn.getResultTypes()[0].dyn_cast<MemRefType>();
    if (!ot || ot.getRank() != 3 || !ot.hasStaticShape() ||
        !ot.getElementType().isF32() || !ot.getLayout().isIdentity() ||
        ot.getMemorySpaceAsInt() != 0)
      return reject("expected a static contiguous rank-3 FP32 output memref");

    const bool packedQK = types[0].getElementType().isInteger(32);
    auto formatAttr = fn->getAttrOfType<StringAttr>("quadrilatero.attention_qk_type");
    StringRef format = formatAttr ? formatAttr.getValue() : (packedQK ? "fp16" : "fp32");
    const bool fp8 = format == "fp8e4m3" || format == "fp8e5m2";
    if ((!packedQK && format != "fp32") ||
        (packedQK && format != "fp16" && !fp8))
      return reject("attention_qk_type must match FP32 or packed FP16/FP8 inputs");
    const int64_t pack = fp8 ? 4 : (packedQK ? 2 : 1);
    const int64_t qkRows = types[0].getDimSize(1);
    const int64_t qkDim = qkRows * pack;
    if (types[1].getElementType() != types[0].getElementType() ||
        !types[2].getElementType().isF32() ||
        (!packedQK && qkDim != 64) ||
        (packedQK && qkDim != 64 && qkDim != 128 && !(fp8 && qkDim == 256)))
      return reject("attention supports FP32 Dqk=64, FP16 Dqk=64/128, FP8 Dqk=64/128/256; V/output are FP32 Dv=64");
    int64_t heads = types[0].getDimSize(0);
    int64_t lq = types[0].getDimSize(2), lk = types[1].getDimSize(2);
    if (heads <= 0 || heads >= INT32_MAX || lq <= 0 || lk <= 0 ||
        lq > INT32_MAX - 64 || lk > INT32_MAX - 64 ||
        heads > INT32_MAX / 256 / std::max(lq, lk) ||
        types[1].getDimSize(1) != qkRows ||
        types[1].getDimSize(0) != heads || types[2].getDimSize(0) != heads ||
        types[2].getDimSize(1) != lk || types[2].getDimSize(2) != 64 ||
        ot.getDimSize(0) != heads || ot.getDimSize(1) != lq || ot.getDimSize(2) != 64)
      return reject("attention requires QT[H,Dqk/pack,Lq], KT[H,Dqk/pack,Lk], V[H,Lk,64] -> O[H,Lq,64]");

    Block &entry = fn.getBody().front();
    while (!entry.empty()) entry.back().erase();
    fn->removeAttr("quadrilatero.fused_attention_serial");
    fn->removeAttr("quadrilatero.fused_attention_pipeline");
    OpBuilder b(&entry, entry.end());
    Location loc = fn.getLoc();
    auto module = fn->getParentOfType<ModuleOp>();
    auto runtime = [&](StringRef name, FunctionType ft) {
      if (auto existing = module.lookupSymbol<func::FuncOp>(name)) return existing;
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      auto f = b.create<func::FuncOp>(loc, name, ft);
      f.setPrivate();
      return f;
    };
    auto coreFn = runtime("snrt_cluster_core_idx", b.getFunctionType({}, {b.getI32Type()}));
    auto barrierFn = runtime("snrt_cluster_hw_barrier", b.getFunctionType({}, {}));
    auto resetFn = runtime("snrt_l1alloc_reset", b.getFunctionType({}, {}));
    auto ci = [&](int64_t n) -> Value { return b.create<arith::ConstantIndexOp>(loc, n); };
    Value z = ci(0), one = ci(1), d = ci(64), br = ci(64), eight = ci(8);
    Value packedRows = ci(qkRows), logicalQK = ci(qkDim);
    Value qkShift = ci(fp8 ? 2 : (packedQK ? 1 : 0));

    auto qkDT = b.getI32IntegerAttr(fp8 ? (format == "fp8e4m3" ? 12 : 4) : (packedQK ? 5 : 6));
    Value hEnd = ci(heads), qEnd = ci(lq);
    Value cid = b.create<func::CallOp>(loc, coreFn, ValueRange{}).getResult(0);
    Value zero32 = b.create<arith::ConstantIntOp>(loc, 0, 32);
    Value one32 = b.create<arith::ConstantIntOp>(loc, 1, 32);
    Value core0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, zero32);
    Value core1 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, one32);
    Attribute l1 = b.getI64IntegerAttr(1);
    auto alloc = [&](ArrayRef<int64_t> shape) -> Value {
      return b.create<memref::AllocOp>(loc, MemRefType::get(shape, b.getF32Type(), MemRefLayoutAttrInterface{}, l1));
    };

    Value out = b.create<memref::AllocOp>(loc, ot);
    Type qkElement = packedQK ? Type(b.getI32Type()) : Type(b.getF32Type());
    auto allocQK = [&]() -> Value {
      return b.create<memref::AllocOp>(loc,
          MemRefType::get({qkRows,64}, qkElement, MemRefLayoutAttrInterface{}, l1));
    };
    Value qt = allocQK();
    Value kt[2], finalDelta;
    if (packedQK) {
      Value storage = b.create<memref::AllocOp>(loc,
          MemRefType::get({64*64*4}, b.getI8Type(), MemRefLayoutAttrInterface{}, l1));
      auto words = MemRefType::get({64,64}, b.getI32Type(), MemRefLayoutAttrInterface{}, l1);
      auto floats = MemRefType::get({64,64}, b.getF32Type(), MemRefLayoutAttrInterface{}, l1);
      kt[0] = b.create<memref::ViewOp>(loc, words, storage, z, ValueRange{});
      finalDelta = b.create<memref::ViewOp>(loc, floats, storage, z, ValueRange{});
      kt[1] = allocQK();
    } else {
      kt[0] = allocQK(); kt[1] = allocQK(); finalDelta = kt[0];
    }
    Value vt[2] = {alloc({32,64}), alloc({32,64})};
    Value sp[2] = {alloc({32,64}), alloc({32,64})};
    Value delta = alloc({64,64}), acc = alloc({64,64});
    Value max = alloc({64}), denom = alloc({64}), alpha = alloc({64});
    auto tagTy = MemRefType::get({1}, b.getI32Type());
    Value tag = b.create<memref::AllocaOp>(loc, tagTy);
    Value qtag = b.create<memref::AllocaOp>(loc, tagTy);
    Value ktag[2] = {b.create<memref::AllocaOp>(loc, tagTy), b.create<memref::AllocaOp>(loc, tagTy)};
    Value vtag[2] = {b.create<memref::AllocaOp>(loc, tagTy), b.create<memref::AllocaOp>(loc, tagTy)};

    auto barrier = [&](OpBuilder &ib) {
      ib.create<scf::IfOp>(loc, core0, [&](OpBuilder &cb, Location cl) {
        OperationState state(loc, LLVM::InlineAsmOp::getOperationName());
        state.addAttribute("asm_string", cb.getStringAttr("csrr t0, fcsr"));
        state.addAttribute("constraints", cb.getStringAttr("~{t0},~{memory}"));
        state.addAttribute("has_side_effects", cb.getUnitAttr());
        cb.create(state);
        cb.create<scf::YieldOp>(cl);
      });
      ib.create<func::CallOp>(loc, barrierFn, ValueRange{});
    };
    auto on = [&](OpBuilder &ib, Value who, auto body) {
      ib.create<scf::IfOp>(loc, who, [&](OpBuilder &pb, Location pl) {
        body(pb);
        pb.create<scf::YieldOp>(pl);
      });
    };
    auto phase = [&](OpBuilder &ib, Value who, auto body) {
      on(ib, who, body);
      barrier(ib);
    };

    auto loop = [&](OpBuilder &ib, Value end, Value step, auto body) {
      Value end32 = ib.create<arith::IndexCastOp>(loc, ib.getI32Type(), end);
      Value step32 = ib.create<arith::IndexCastOp>(loc, ib.getI32Type(), step);
      auto w = ib.create<scf::WhileOp>(loc, TypeRange{ib.getI32Type()}, ValueRange{zero32});
      OpBuilder::InsertionGuard guard(ib);
      Block *before = ib.createBlock(&w.getBefore(), w.getBefore().end(),
                                    TypeRange{ib.getI32Type()}, SmallVector<Location>{loc});
      Value iv = before->getArgument(0);
      Value cond = ib.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, iv, end32);
      ib.create<scf::ConditionOp>(loc, cond, ValueRange{iv});
      Block *after = ib.createBlock(&w.getAfter(), w.getAfter().end(),
                                   TypeRange{ib.getI32Type()}, SmallVector<Location>{loc});
      iv = after->getArgument(0);
      Value idx = ib.create<arith::IndexCastOp>(loc, ib.getIndexType(), iv);
      body(ib, idx);
      Value next = ib.create<arith::AddIOp>(loc, iv, step32);
      ib.create<scf::YieldOp>(loc, ValueRange{next});
    };
    auto i32 = [&](OpBuilder &ib, Value v) -> Value {
      return v.getType().isInteger(32) ? v : ib.create<arith::IndexCastOp>(loc, ib.getI32Type(), v).getResult();
    };
    auto index = [&](OpBuilder &ib, Value v) -> Value {
      return v.getType().isIndex() ? v : ib.create<arith::IndexCastOp>(loc, ib.getIndexType(), v).getResult();
    };
    auto extent = [&](OpBuilder &ib, Value end, Value off, Value tile) -> Value {
      Value left = ib.create<arith::SubIOp>(loc, i32(ib, end), i32(ib, off));
      Value tile32 = i32(ib, tile);
      Value smaller = ib.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, left, tile32);
      return index(ib, ib.create<arith::SelectOp>(loc, smaller, left, tile32));
    };

    auto stamp = [&](OpBuilder &ib, bool finish) {
      OperationState state(loc, LLVM::InlineAsmOp::getOperationName());
      std::string code = finish
          ? "csrr t5, mcycle\n\tlla t6, attention_kernel_timing\n\t"
            "lw t4, 0(t6)\n\tsub t5, t5, t4\n\tsw t5, 4(t6)"
          : "lla t6, attention_kernel_timing\n\tcsrr t5, mcycle\n\tsw t5, 0(t6)";
      state.addAttribute("asm_string", ib.getStringAttr(code));
      state.addAttribute("constraints", ib.getStringAttr("~{t4},~{t5},~{t6},~{memory}"));
      state.addAttribute("has_side_effects", ib.getUnitAttr());
      ib.create(state);
    };
    auto sub = [&](OpBuilder &ib, Value base, Value rows, Value cols) {
      return createSubview2D(ib, loc, base, z, z, rows, cols,
                             base.getType().cast<MemRefType>().getElementType(), l1);
    };
    auto vecSub = [&](OpBuilder &ib, Value base, Value off, Value size) -> Value {
      SmallVector<OpFoldResult> offsets{off}, sizes{size}, strides{ib.getIndexAttr(1)};
      return ib.create<memref::SubViewOp>(loc, base, offsets, sizes, strides);
    };
    auto start = [&](OpBuilder &ib, Value src, Value dst, Value rows, Value cols, Value t) {
      {
      Value count32 = ib.create<arith::MulIOp>(loc, i32(ib, rows), i32(ib, cols));
      Value count = index(ib, count32);
      ib.create<memref::DmaStartOp>(loc, src, ValueRange{z,z}, dst,
                                  ValueRange{z,z}, count, t, ValueRange{z});
      }
    };
    auto wait = [&](OpBuilder &ib, Value rows, Value cols, Value t) {
      {
      Value count32 = ib.create<arith::MulIOp>(loc, i32(ib, rows), i32(ib, cols));
      Value count = index(ib, count32);
      ib.create<memref::DmaWaitOp>(loc, t, ValueRange{z}, count);
      }
    };
    auto pitch = b.getI64IntegerAttr(64);
    auto dt = b.getI32IntegerAttr(6);

    int64_t blocks = (lk + 31) / 32;
    SmallVector<Value> offsets, counts;
    for (int64_t j = 0; j < blocks; ++j) {
      offsets.push_back(ci(j * 32));
      counts.push_back(ci(std::min<int64_t>(32, lk - j * 32)));
    }
    loop(b, hEnd, one, [&](OpBuilder &hb, Value h) {
      loop(hb, qEnd, br, [&](OpBuilder &qb, Value q) {
        Value m = extent(qb, qEnd, q, br);

        Value h32 = i32(qb, h), q32 = i32(qb, q);
        Value hEnd32 = i32(qb, hEnd), qEnd32 = i32(qb, qEnd);
        Value firstH = qb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, h32, zero32);
        Value firstQ = qb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, q32, zero32);
        Value firstTile = qb.create<arith::AndIOp>(loc, firstH, firstQ);
        Value qPlus32 = qb.create<arith::AddIOp>(loc, q32, i32(qb, br));
        Value sameHead = qb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, qPlus32, qEnd32);
        Value hPlus32 = qb.create<arith::AddIOp>(loc, h32, one32);
        Value moreHeads = qb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, hPlus32, hEnd32);
        Value hasNext = qb.create<arith::OrIOp>(loc, sameHead, moreHeads);
        Value nextH32 = qb.create<arith::SelectOp>(loc, sameHead, h32, hPlus32);
        Value nextQ32 = qb.create<arith::SelectOp>(loc, sameHead, qPlus32, zero32);
        Value nextH = index(qb, nextH32), nextQ = index(qb, nextQ32);
        Value nextM = extent(qb, qEnd32, nextQ32, i32(qb, br));

        int64_t issued = 0, completed = 0;
        int64_t kseq[2] = {0,0}, vseq[2] = {0,0};
        auto loadK = [&](OpBuilder &pb, int64_t j) {
          if (j >= blocks) return;
          Value src = createSubview3DTo2D(pb, loc, entry.getArgument(1), h, z, offsets[j], packedRows, counts[j]);
          start(pb, src, sub(pb, kt[1-(j%2)], packedRows, counts[j]), packedRows, counts[j], ktag[j%2]);
          kseq[j%2] = ++issued;
        };
        auto loadV = [&](OpBuilder &pb, int64_t j) {
          if (j >= blocks) return;
          Value src = createSubview3DTo2D(pb, loc, entry.getArgument(2), h, offsets[j], z, counts[j], d);
          start(pb, src, sub(pb, vt[j%2], counts[j], d), counts[j], d, vtag[j%2]);
          vseq[j%2] = ++issued;
        };
        auto waitK = [&](OpBuilder &pb, int64_t j) {
          if (j < blocks && kseq[j%2] > completed) {
            wait(pb, packedRows, counts[j], ktag[j%2]);
            completed = kseq[j%2];
          }
        };
        auto waitV = [&](OpBuilder &pb, int64_t j) {
          if (j < blocks && vseq[j%2] > completed) {
            wait(pb, counts[j], d, vtag[j%2]);
            completed = vseq[j%2];
          }
        };
        auto qk = [&](OpBuilder &pb, int64_t j) {
          {
          pb.create<quadrilatero::TcdmMatmulMemRefOp>(loc, sub(pb, kt[1-(j%2)], packedRows, counts[j]),
              sub(pb, qt, packedRows, m), sub(pb, sp[j%2], counts[j], m),
              counts[j], m, logicalQK, qkShift, qkDT, qkDT, qkDT);
          }
        };
        auto pv = [&](OpBuilder &pb, int64_t j) {

          Value dst = (blocks > 1 && j + 1 == blocks) ? finalDelta : delta;
          pb.create<quadrilatero::TcdmMatmulMemRefOp>(loc, sub(pb, sp[j%2], counts[j], m),
              sub(pb, vt[j%2], counts[j], d), sub(pb, dst, m, d), m, d, counts[j], z, dt, dt, dt);
        };
        auto prepare = [&](OpBuilder &ib, int64_t j) {
          ib.create<spatz::AttentionSoftmaxUpdateOp>(loc, sp[j%2], max, alpha, denom, m,
              ib.getI64IntegerAttr(std::min<int64_t>(32, lk - j*32)),
              ib.getI64IntegerAttr(qkDim), ib.getBoolAttr(j == 0));
        };
        auto update = [&](OpBuilder &ib, bool fromSource, bool add, bool rescale,
                          Value source = Value{}) {
          ib.create<spatz::MatrixUpdateOp>(loc, acc, source ? source : delta, alpha, m,
              ib.getBoolAttr(fromSource), ib.getBoolAttr(add), ib.getBoolAttr(rescale));
        };
        auto reciprocal = [&](OpBuilder &ib) {
          ib.create<spatz::VectorReciprocalApproxOp>(loc, denom, alpha, m);
        };
        auto integrate = [&](OpBuilder &pb, bool rescale, bool first) {
          {
            {
              update(pb, true, !first, rescale);
            }
          }
        };

        auto prefetchQ = [&](OpBuilder &pb) {
          on(pb, hasNext, [&](OpBuilder &nb) {
            Value src = createSubview3DTo2D(nb, loc, entry.getArgument(0), nextH, z, nextQ, packedRows, nextM);
            start(nb, src, sub(nb, qt, packedRows, nextM), packedRows, nextM, qtag);
          });
        };
        on(qb, firstTile, [&](OpBuilder &fb) {
          phase(fb, core0, [&](OpBuilder &pb) {
            Value src = createSubview3DTo2D(pb, loc, entry.getArgument(0), h, z, q, packedRows, m);
            Value qdst = sub(pb, qt, packedRows, m);
            stamp(pb, false);
            start(pb, src, qdst, packedRows, m, qtag);
            loadK(pb, 0);
            waitK(pb, 0);
          });
          on(fb, core1, [&](OpBuilder &pb) { qk(pb, 0); });
          on(fb, core0, [&](OpBuilder &pb) {
            loadV(pb, 0); loadK(pb, 1);
            if (blocks > 1) waitK(pb, 1); else waitV(pb, 0);
          });
          barrier(fb);
        });
        if (blocks > 1) {
          on(qb, core1, [&](OpBuilder &pb) { qk(pb, 1); });
          on(qb, core0, [&](OpBuilder &pb) {

            loadV(pb, 1); loadK(pb, 2);
            prepare(pb, 0); waitV(pb, 0);
          });
          barrier(qb);
        } else {
          phase(qb, core0, [&](OpBuilder &pb) { prepare(pb, 0); });
        }
        for (int64_t j = 0; j < blocks; ++j) {
          on(qb, core1, [&](OpBuilder &pb) { pv(pb, j); });
          on(qb, core0, [&](OpBuilder &pb) {

            if (j == std::max<int64_t>(0, blocks - 2)) prefetchQ(pb);
            if (j + 1 < blocks) {
              prepare(pb, j + 1);
            } else {

              on(pb, hasNext, [&](OpBuilder &nb) {
                Value src = createSubview3DTo2D(nb, loc, entry.getArgument(1), nextH, z, z, packedRows, counts[0]);
                start(nb, src, sub(nb, kt[1], packedRows, counts[0]), packedRows, counts[0], ktag[0]);
              });
              if (blocks > 1) integrate(pb, true, blocks == 2);
              reciprocal(pb);

              on(pb, hasNext, [&](OpBuilder &nb) { wait(nb, packedRows, counts[0], ktag[0]); });
            }
            waitK(pb, j + 2);

            if (blocks > 1 && j + 2 == blocks) waitV(pb, j + 1);
          });
          barrier(qb);
          if (j + 2 < blocks) {
            on(qb, core1, [&](OpBuilder &pb) { qk(pb, j + 2); });
            on(qb, core0, [&](OpBuilder &pb) {

              loadV(pb, j + 2); loadK(pb, j + 3);
              integrate(pb, true, j == 0); waitV(pb, j + 1);
            });
            barrier(qb);
          }
        }

        on(qb, core1, [&](OpBuilder &pb) {
          on(pb, hasNext, [&](OpBuilder &nb) {
            nb.create<quadrilatero::TcdmMatmulMemRefOp>(loc,
                sub(nb, kt[1], packedRows, counts[0]), sub(nb, qt, packedRows, nextM),
                sub(nb, sp[0], counts[0], nextM), counts[0], nextM, logicalQK,
                qkShift, qkDT, qkDT, qkDT);
          });
        });
        on(qb, core0, [&](OpBuilder &pb) {
          update(pb, true, blocks > 1,
                    true, blocks > 1 ? finalDelta : delta);
          Value dst = createSubview3DTo2D(pb, loc, out, h, q, z, m, d);
          start(pb, sub(pb, acc, m, d), dst, m, d, tag);

          pb.create<scf::IfOp>(loc, hasNext,
            [&](OpBuilder &nb, Location nl) {
              Value vs = createSubview3DTo2D(nb, loc, entry.getArgument(2), nextH, z, z, counts[0], d);
              start(nb, vs, sub(nb, vt[0], counts[0], d), counts[0], d, vtag[0]);
              if (blocks > 1) {
                Value ks = createSubview3DTo2D(nb, loc, entry.getArgument(1), nextH, z, offsets[1], packedRows, counts[1]);
                start(nb, ks, sub(nb, kt[0], packedRows, counts[1]), packedRows, counts[1], ktag[1]);
                wait(nb, packedRows, counts[1], ktag[1]);
              } else {
                wait(nb, counts[0], d, vtag[0]);
              }
              nb.create<scf::YieldOp>(nl);
            },
            [&](OpBuilder &lb, Location ll) {
              wait(lb, m, d, tag);
              stamp(lb, true);
              lb.create<scf::YieldOp>(ll);
            });
        });

        barrier(qb);
      });
    });

    {
      b.create<func::CallOp>(loc, resetFn, ValueRange{});
    }
    barrier(b);
    b.create<func::ReturnOp>(loc, ValueRange{out});
    return success();
  }

LogicalResult lowerFlashAttention(linalg::BatchMatmulOp matmul2Op, llvm::SmallPtrSet<Operation*, 8>& erasedOps) {

    Value P_input = matmul2Op.inputs()[0];
    Value V_input = matmul2Op.inputs()[1];
    Value O_output = matmul2Op.outputs()[0];

    auto pType = P_input.getType().dyn_cast<MemRefType>();
    if (!pType || pType.getRank() != 3) return failure();
    if (pType.getShape()[1] != pType.getShape()[2]) return failure();

    linalg::BatchMatmulOp matmul1Op = nullptr;
    SmallVector<Operation*, 16> opsToErase;
    Operation *currentOp = matmul2Op->getPrevNode();
    int limit = 150;

    while (currentOp && limit > 0) {
        if (isa<linalg::GenericOp, linalg::FillOp, memref::CopyOp, func::CallOp>(currentOp)) {
            opsToErase.push_back(currentOp);
        } else if (auto bmm = dyn_cast<linalg::BatchMatmulOp>(currentOp)) {
            matmul1Op = bmm;
            opsToErase.push_back(currentOp);
            break;
        }
        currentOp = currentOp->getPrevNode();
        limit--;
    }

    if (!matmul1Op) return failure();

    bool hasSoftmaxDivision = false;
    for (Operation *op : opsToErase) {
        auto genericOp = dyn_cast<linalg::GenericOp>(op);
        if (!genericOp)
            continue;

        genericOp.getRegion().walk([&](arith::DivFOp) {
            hasSoftmaxDivision = true;
        });

        if (hasSoftmaxDivision)
            break;
    }

    if (!hasSoftmaxDivision)
        return failure();

    Value biasVal = nullptr;
    for (auto* op : opsToErase) {
        if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
            if (genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1) {
                if (!genericOp.getRegion().empty() && !genericOp.getRegion().front().empty()) {
                    if (isa<arith::AddFOp>(genericOp.getRegion().front().front())) {
                        biasVal = genericOp.inputs()[1];
                        break;
                    }
                }
            }
        }
    }
    bool hasBias = (biasVal != nullptr);

    Value Q_input = matmul1Op.inputs()[0];
    Value K_T_input = matmul1Op.inputs()[1];

    auto qType = Q_input.getType().dyn_cast<MemRefType>();
    auto kType = K_T_input.getType().dyn_cast<MemRefType>();
    auto vType = V_input.getType().dyn_cast<MemRefType>();
    auto oType = O_output.getType().dyn_cast<MemRefType>();

    if (!qType || !kType || !vType || !oType || qType.getRank() != 3) return failure();

    OpBuilder builder(matmul2Op);
    Location loc = matmul2Op.getLoc();

    auto module = matmul2Op->getParentOfType<ModuleOp>();
    auto i32Ty = builder.getI32Type();
    auto indexTy = builder.getIndexType();
    SymbolTable symbolTable(module);

    auto getOrInsertFn = [&](StringRef name, FunctionType type) {
      if (auto fn = symbolTable.lookup<func::FuncOp>(name)) return fn;
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto fn = builder.create<func::FuncOp>(loc, name, type);
      fn.setPrivate();
      symbolTable.insert(fn);
      return fn;
    };

    auto getCoreIdxFn = getOrInsertFn("snrt_cluster_core_idx", builder.getFunctionType({}, {i32Ty}));
    auto hwBarrierFn = getOrInsertFn("snrt_cluster_hw_barrier", builder.getFunctionType({}, {}));
    auto l1ResetFn = getOrInsertFn("snrt_l1alloc_reset", builder.getFunctionType({}, {}));

    Value cid = builder.create<func::CallOp>(loc, getCoreIdxFn, ValueRange{}).getResult(0);
    Value c0_i32 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value c1_i32 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    Value c2_i32 = builder.create<arith::ConstantIntOp>(loc, 2, 32);

    Value isCore0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, c0_i32);
    Value isCore1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, c1_i32);

    Value c0_idx = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1_idx = builder.create<arith::ConstantIndexOp>(loc, 1);

    Type elemType = qType.getElementType();
    int32_t dt_val = getDataTypeCode(elemType);
    unsigned bitWidth = elemType.getIntOrFloatBitWidth();
    int64_t shiftVal = (bitWidth == 32) ? 0 : (bitWidth == 16) ? 1 : 2;

    int64_t D_val = 64;
    if (qType.getShape()[2] != ShapedType::kDynamicSize) {
        D_val = qType.getShape()[2];
    }

    int64_t B_r_val = 16;
    int64_t B_c_val = 64;
    int64_t HW_Stride_Elements = 64;

    Value tk_i32 = builder.create<arith::ConstantIntOp>(loc, D_val, 32);
    Value tm_i32 = builder.create<arith::ConstantIntOp>(loc, B_r_val, 32);
    Value tn_i32 = builder.create<arith::ConstantIntOp>(loc, B_c_val, 32);
    Value tm_idx = builder.create<arith::ConstantIndexOp>(loc, B_r_val);
    Value tn_idx = builder.create<arith::ConstantIndexOp>(loc, B_c_val);
    Value shift_idx = builder.create<arith::ConstantIndexOp>(loc, shiftVal);
    Value hw_stride_idx = builder.create<arith::ConstantIndexOp>(loc, HW_Stride_Elements);

    auto l1SpaceAttr = IntegerAttr::get(builder.getI64Type(), 1);

    auto qTL1Type  =  MemRefType::get({D_val, HW_Stride_Elements}, elemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto pTL1Type  =  MemRefType::get({B_c_val, HW_Stride_Elements}, elemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto qL1Type   =  MemRefType::get({B_r_val, HW_Stride_Elements}, elemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto kvL1Type  =  MemRefType::get({D_val, HW_Stride_Elements}, elemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto vL1Type   =  MemRefType::get({B_c_val, HW_Stride_Elements}, elemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto sL1Type   =  MemRefType::get({B_c_val, HW_Stride_Elements}, elemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto oL1Type   =  MemRefType::get({B_r_val, HW_Stride_Elements}, elemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto mlL1Type  =  MemRefType::get({B_r_val}, builder.getF32Type(), MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto mBlockType = MemRefType::get({B_r_val}, builder.getF32Type(), MemRefLayoutAttrInterface{}, l1SpaceAttr);

    Value q_dma   = builder.create<memref::AllocOp>(loc, qL1Type);
    Value q_T     = builder.create<memref::AllocOp>(loc, qTL1Type);
    Value k_0     = builder.create<memref::AllocOp>(loc, kvL1Type);
    Value k_1     = builder.create<memref::AllocOp>(loc, kvL1Type);
    Value v_0     = builder.create<memref::AllocOp>(loc, vL1Type);
    Value v_1     = builder.create<memref::AllocOp>(loc, vL1Type);
    Value s_buf   = builder.create<memref::AllocOp>(loc, sL1Type);
    Value p_T     = builder.create<memref::AllocOp>(loc, pTL1Type);
    Value o_tile  = builder.create<memref::AllocOp>(loc, oL1Type);
    Value o_buf   = builder.create<memref::AllocOp>(loc, oL1Type);
    Value m_val   = builder.create<memref::AllocOp>(loc, mlL1Type);
    Value l_val   = builder.create<memref::AllocOp>(loc, mlL1Type);
    Value m_block = builder.create<memref::AllocOp>(loc, mBlockType);
    Value sum_buf = builder.create<memref::AllocOp>(loc, mBlockType);
    Value m_diff_buf = builder.create<memref::AllocOp>(loc, mBlockType);
    Value diff_prev_buf = builder.create<memref::AllocOp>(loc, mBlockType);

    auto tagType = MemRefType::get({1}, builder.getI32Type());
    Value tagQ = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagK_0 = builder.create<memref::AllocaOp>(loc, tagType), tagK_1 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagV_0 = builder.create<memref::AllocaOp>(loc, tagType), tagV_1 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagO = builder.create<memref::AllocaOp>(loc, tagType);

    Value bias_0 = nullptr, bias_1 = nullptr, tagBias_0 = nullptr, tagBias_1 = nullptr;
    if (hasBias) {
        auto biasL1Type = MemRefType::get({HW_Stride_Elements}, elemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
        bias_0 = builder.create<memref::AllocOp>(loc, biasL1Type);
        bias_1 = builder.create<memref::AllocOp>(loc, biasL1Type);
        tagBias_0 = builder.create<memref::AllocaOp>(loc, tagType);
        tagBias_1 = builder.create<memref::AllocaOp>(loc, tagType);
    }

    float scale_factor = 1.0f / std::sqrt(static_cast<float>(D_val));
    Value m_scale = builder.create<arith::ConstantFloatOp>(loc, llvm::APFloat(scale_factor), builder.getF32Type());

    Value f_zero = builder.create<arith::ConstantFloatOp>(loc, llvm::APFloat(0.0f), builder.getF32Type());
    Value neg_inf = builder.create<arith::ConstantFloatOp>(loc, llvm::APFloat(-INFINITY), builder.getF32Type());

    Value dimBatch_idx = builder.create<memref::DimOp>(loc, Q_input, 0);
    Value dimSeq_idx = builder.create<memref::DimOp>(loc, Q_input, 1);
    Value dimSeq = builder.create<arith::IndexCastOp>(loc, i32Ty, dimSeq_idx);

    Value kSourceStrideIdx = builder.create<memref::DimOp>(loc, K_T_input, 2);
    Value vSourceStrideIdx = builder.create<memref::DimOp>(loc, V_input, 2);

    builder.create<scf::ForOp>(loc, c0_idx, dimBatch_idx, c1_idx, ValueRange{},
        [&](OpBuilder &bB, Location lB, Value batch_idx, ValueRange argsB) {

        bB.create<scf::ForOp>(lB, c0_idx, dimSeq_idx, tm_idx, ValueRange{},
            [&](OpBuilder &bQ, Location lQ, Value q_idx, ValueRange argsQ) {

                Value qOff = bQ.create<arith::IndexCastOp>(lQ, i32Ty, q_idx);
                Value mE = createMinI32(bQ, lQ, bQ.create<arith::SubIOp>(lQ, dimSeq, qOff), tm_i32);
                Value mE_idx = bQ.create<arith::IndexCastOp>(lQ, indexTy, mE);
                Value D_idx = bQ.create<arith::ConstantIndexOp>(lQ, D_val);

                bQ.create<scf::IfOp>(lQ, isCore0, [&](OpBuilder &b0, Location l0) {
                    Value qSub = createSubview3DTo2D(b0, l0, Q_input, batch_idx, q_idx, c0_idx, mE_idx, D_idx);
                    Value qL1Sub = createSubview2D(b0, l0, q_dma, c0_idx, c0_idx, mE_idx, D_idx, elemType, l1SpaceAttr);
                    Value szQ_idx = b0.create<arith::IndexCastOp>(l0, indexTy, b0.create<arith::MulIOp>(l0, mE, tk_i32));

                    b0.create<memref::DmaStartOp>(l0, qSub, ValueRange{c0_idx, c0_idx}, qL1Sub, ValueRange{c0_idx, c0_idx}, szQ_idx, tagQ, ValueRange{c0_idx});
                    b0.create<memref::DmaWaitOp>(l0, tagQ, ValueRange{c0_idx}, szQ_idx);

                    b0.create<linalg::FillOp>(l0, ValueRange{f_zero}, ValueRange{o_buf});
                    b0.create<linalg::FillOp>(l0, ValueRange{f_zero}, ValueRange{o_tile});
                    b0.create<linalg::FillOp>(l0, ValueRange{neg_inf}, ValueRange{m_val});
                    b0.create<linalg::FillOp>(l0, ValueRange{f_zero}, ValueRange{l_val});

                    b0.create<scf::ForOp>(l0, c0_idx, mE_idx, c1_idx, ValueRange{}, [&](OpBuilder &bR, Location lR, Value r_idx, ValueRange) {
                        bR.create<scf::ForOp>(lR, c0_idx, D_idx, c1_idx, ValueRange{}, [&](OpBuilder &bC, Location lC, Value c_idx, ValueRange) {
                            Value val = bC.create<memref::LoadOp>(lC, q_dma, ValueRange{r_idx, c_idx});
                            bC.create<memref::StoreOp>(lC, val, q_T, ValueRange{c_idx, r_idx});
                            bC.create<scf::YieldOp>(lC);
                        });
                        bR.create<scf::YieldOp>(lR);
                    });

                    Value nE_0 = createMinI32(b0, l0, dimSeq, tn_i32);
                    Value nE_0_idx = b0.create<arith::IndexCastOp>(l0, indexTy, nE_0);

                    Value kSub_0 = createSubview3DTo2D(b0, l0, K_T_input, batch_idx, c0_idx, c0_idx, D_idx, nE_0_idx);
                    Value vSub_0 = createSubview3DTo2D(b0, l0, V_input, batch_idx, c0_idx, c0_idx, nE_0_idx, D_idx);

                    Value kL1Sub_0 = createSubview2D(b0, l0, k_0, c0_idx, c0_idx, D_idx, nE_0_idx, elemType, l1SpaceAttr);
                    Value vL1Sub_0 = createSubview2D(b0, l0, v_0, c0_idx, c0_idx, nE_0_idx, D_idx, elemType, l1SpaceAttr);

                    Value szK_0_idx = b0.create<arith::IndexCastOp>(l0, indexTy, b0.create<arith::MulIOp>(l0, tk_i32, nE_0));

                    b0.create<memref::DmaStartOp>(l0, kSub_0, ValueRange{c0_idx, c0_idx}, kL1Sub_0, ValueRange{c0_idx, c0_idx}, szK_0_idx, tagK_0, ValueRange{c0_idx}, kSourceStrideIdx, nE_0_idx);
                    b0.create<memref::DmaStartOp>(l0, vSub_0, ValueRange{c0_idx, c0_idx}, vL1Sub_0, ValueRange{c0_idx, c0_idx}, szK_0_idx, tagV_0, ValueRange{c0_idx}, vSourceStrideIdx, nE_0_idx);

                    if (hasBias) {
                        Value biasSub_0 = createSubview1D(b0, l0, biasVal, c0_idx, nE_0_idx, elemType, biasVal.getType().cast<MemRefType>().getMemorySpace());
                        Value biasL1Sub_0 = createSubview1D(b0, l0, bias_0, c0_idx, nE_0_idx, elemType, l1SpaceAttr);
                        b0.create<memref::DmaStartOp>(l0, biasSub_0, ValueRange{c0_idx}, biasL1Sub_0, ValueRange{c0_idx}, nE_0_idx, tagBias_0, ValueRange{c0_idx});
                        b0.create<memref::DmaWaitOp>(l0, tagBias_0, ValueRange{c0_idx}, nE_0_idx);
                    }

                    b0.create<memref::DmaWaitOp>(l0, tagK_0, ValueRange{c0_idx}, szK_0_idx);
                    b0.create<memref::DmaWaitOp>(l0, tagV_0, ValueRange{c0_idx}, szK_0_idx);
                    b0.create<scf::YieldOp>(l0);
                });

                bQ.create<func::CallOp>(lQ, hwBarrierFn, ValueRange{});

                bQ.create<scf::ForOp>(lQ, c0_idx, dimSeq_idx, tn_idx, ValueRange{c0_i32},
                    [&](OpBuilder &bKV, Location lKV, Value kv_idx, ValueRange argsKV) {

                        Value flatIdx = argsKV[0];
                        Value kvOff = bKV.create<arith::IndexCastOp>(lKV, i32Ty, kv_idx);

                        Value parity = bKV.create<arith::RemUIOp>(lKV, flatIdx, c2_i32);
                        Value is_even = bKV.create<arith::CmpIOp>(lKV, arith::CmpIPredicate::eq, parity, c0_i32);

                        Value k_curr = bKV.create<arith::SelectOp>(lKV, is_even, k_0, k_1);
                        Value v_curr = bKV.create<arith::SelectOp>(lKV, is_even, v_0, v_1);
                        Value k_next = bKV.create<arith::SelectOp>(lKV, is_even, k_1, k_0);
                        Value v_next = bKV.create<arith::SelectOp>(lKV, is_even, v_1, v_0);
                        Value tagK_n = bKV.create<arith::SelectOp>(lKV, is_even, tagK_1, tagK_0);
                        Value tagV_n = bKV.create<arith::SelectOp>(lKV, is_even, tagV_1, tagV_0);

                        Value bias_curr = nullptr, bias_next = nullptr, tagBias_n = nullptr;
                        if (hasBias) {
                            bias_curr = bKV.create<arith::SelectOp>(lKV, is_even, bias_0, bias_1);
                            bias_next = bKV.create<arith::SelectOp>(lKV, is_even, bias_1, bias_0);
                            tagBias_n = bKV.create<arith::SelectOp>(lKV, is_even, tagBias_1, tagBias_0);
                        }

                        Value nE = createMinI32(bKV, lKV, bKV.create<arith::SubIOp>(lKV, dimSeq, kvOff), tn_i32);
                        Value nE_idx = bKV.create<arith::IndexCastOp>(lKV, indexTy, nE);

                        bKV.create<scf::IfOp>(lKV, isCore0, [&](OpBuilder &b0, Location l0) {
                            Value next_kvOff = b0.create<arith::AddIOp>(l0, kvOff, tn_i32);
                            Value has_next = b0.create<arith::CmpIOp>(l0, arith::CmpIPredicate::slt, next_kvOff, dimSeq);
                            b0.create<scf::IfOp>(l0, has_next, [&](OpBuilder &bN, Location lN) {
                                Value n_nE = createMinI32(bN, lN, bN.create<arith::SubIOp>(lN, dimSeq, next_kvOff), tn_i32);
                                Value n_nE_idx = bN.create<arith::IndexCastOp>(lN, indexTy, n_nE);
                                Value next_kv_idx = bN.create<arith::IndexCastOp>(lN, indexTy, next_kvOff);

                                Value kSub_n = createSubview3DTo2D(bN, lN, K_T_input, batch_idx, c0_idx, next_kv_idx, D_idx, n_nE_idx);
                                Value vSub_n = createSubview3DTo2D(bN, lN, V_input, batch_idx, next_kv_idx, c0_idx, n_nE_idx, D_idx);

                                Value kL1Sub_n = createSubview2D(bN, lN, k_next, c0_idx, c0_idx, D_idx, n_nE_idx, elemType, l1SpaceAttr);
                                Value vL1Sub_n = createSubview2D(bN, lN, v_next, c0_idx, c0_idx, n_nE_idx, D_idx, elemType, l1SpaceAttr);

                                Value szN_idx = bN.create<arith::IndexCastOp>(lN, indexTy, bN.create<arith::MulIOp>(lN, tk_i32, n_nE));

                                bN.create<memref::DmaStartOp>(lN, kSub_n, ValueRange{c0_idx, c0_idx}, kL1Sub_n, ValueRange{c0_idx, c0_idx}, szN_idx, tagK_n, ValueRange{c0_idx}, kSourceStrideIdx, n_nE_idx);
                                bN.create<memref::DmaStartOp>(lN, vSub_n, ValueRange{c0_idx, c0_idx}, vL1Sub_n, ValueRange{c0_idx, c0_idx}, szN_idx, tagV_n, ValueRange{c0_idx}, vSourceStrideIdx, n_nE_idx);

                                if (hasBias) {
                                    Value biasSub_n = createSubview1D(bN, lN, biasVal, next_kv_idx, n_nE_idx, elemType, biasVal.getType().cast<MemRefType>().getMemorySpace());
                                    Value biasL1Sub_n = createSubview1D(bN, lN, bias_next, c0_idx, n_nE_idx, elemType, l1SpaceAttr);
                                    bN.create<memref::DmaStartOp>(lN, biasSub_n, ValueRange{c0_idx}, biasL1Sub_n, ValueRange{c0_idx}, n_nE_idx, tagBias_n, ValueRange{c0_idx});
                                }
                                bN.create<scf::YieldOp>(lN);
                            });
                            b0.create<scf::YieldOp>(l0);
                        });

                        bKV.create<scf::IfOp>(lKV, isCore1, [&](OpBuilder &b1, Location l1) {
                            Value qTSub = createSubview2D(b1, l1, q_T, c0_idx, c0_idx, D_idx, mE_idx, elemType, l1SpaceAttr);
                            Value kSub  = createSubview2D(b1, l1, k_curr, c0_idx, c0_idx, D_idx, nE_idx, elemType, l1SpaceAttr);
                            Value sTSub = createSubview2D(b1, l1, s_buf, c0_idx, c0_idx, nE_idx, mE_idx, elemType, l1SpaceAttr);

                            b1.create<quadrilatero::TcdmMatmulMemRefOp>(l1, kSub, qTSub, sTSub, nE_idx, mE_idx, D_idx,
                                        shift_idx, builder.getI32IntegerAttr(dt_val), builder.getI32IntegerAttr(dt_val), builder.getI32IntegerAttr(dt_val));

                            b1.create<scf::YieldOp>(l1);
                        });

                        bKV.create<func::CallOp>(lKV, hwBarrierFn, ValueRange{});

                        bKV.create<scf::IfOp>(lKV, isCore0, [&](OpBuilder &b0, Location l0) {

                            b0.create<spatz::MatrixScalarMulOp>(l0, s_buf, m_scale, nE_idx, mE_idx, b0.getI64IntegerAttr(HW_Stride_Elements), b0.getI32IntegerAttr(dt_val));
                            b0.create<spatz::MatrixColumnMaxOp>(l0, s_buf, m_block, nE_idx, mE_idx, b0.getI64IntegerAttr(HW_Stride_Elements), b0.getI32IntegerAttr(dt_val));

                            auto strideAttr = b0.getI64IntegerAttr(HW_Stride_Elements);
                            auto dtAttr = b0.getI32IntegerAttr(dt_val);

                            b0.create<spatz::VectorMaxUpdateOp>(l0, m_val, m_block, m_diff_buf, mE_idx, strideAttr, dtAttr);

                            b0.create<spatz::VectorExpOp>(l0, m_diff_buf, diff_prev_buf, mE_idx, strideAttr, dtAttr);

                            b0.create<spatz::VectorMulOp>(l0, l_val, diff_prev_buf, mE_idx, strideAttr, dtAttr);

                            b0.create<spatz::MatrixRowScaleOp>(l0, o_buf, diff_prev_buf, mE_idx, D_idx, strideAttr, dtAttr);

                            if (hasBias) {
                                b0.create<spatz::MatrixExpReduceOp>(l0, s_buf, p_T, m_val, sum_buf, bias_curr, nE_idx, mE_idx, strideAttr, dtAttr);
                            } else {
                                b0.create<spatz::MatrixExpReduceOp>(l0, s_buf, p_T, m_val, sum_buf, Value(), nE_idx, mE_idx, strideAttr, dtAttr);
                            }

                            b0.create<spatz::VectorAddOp>(l0, l_val, sum_buf, mE_idx, strideAttr, dtAttr);

                            b0.create<scf::YieldOp>(l0);
                        });

                        bKV.create<func::CallOp>(lKV, hwBarrierFn, ValueRange{});

                        bKV.create<scf::IfOp>(lKV, isCore1, [&](OpBuilder &b1, Location l1) {
                            Value pTSub = createSubview2D(b1, l1, p_T, c0_idx, c0_idx, nE_idx, mE_idx, elemType, l1SpaceAttr);
                            Value vSub  = createSubview2D(b1, l1, v_curr, c0_idx, c0_idx, nE_idx, D_idx, elemType, l1SpaceAttr);
                            Value oTileSub = createSubview2D(b1, l1, o_tile, c0_idx, c0_idx, mE_idx, D_idx, elemType, l1SpaceAttr);
                            b1.create<quadrilatero::TcdmMatmulMemRefOp>(l1, pTSub, vSub, oTileSub, mE_idx, D_idx, nE_idx, shift_idx, builder.getI32IntegerAttr(dt_val), builder.getI32IntegerAttr(dt_val), builder.getI32IntegerAttr(dt_val));
                            b1.create<scf::YieldOp>(l1);
                        });

                        bKV.create<func::CallOp>(lKV, hwBarrierFn, ValueRange{});

                        bKV.create<scf::IfOp>(lKV, isCore0, [&](OpBuilder &b0, Location l0) {
                            b0.create<scf::ForOp>(l0, c0_idx, mE_idx, c1_idx, ValueRange{}, [&](OpBuilder &bR, Location lR, Value r_idx, ValueRange) {
                                bR.create<scf::ForOp>(lR, c0_idx, D_idx, c1_idx, ValueRange{}, [&](OpBuilder &bC, Location lC, Value d_idx, ValueRange) {
                                    Value o_acc = bC.create<memref::LoadOp>(lC, o_buf, ValueRange{r_idx, d_idx});
                                    Value o_new = bC.create<memref::LoadOp>(lC, o_tile, ValueRange{r_idx, d_idx});
                                    Value o_sum = bC.create<arith::AddFOp>(lC, o_acc, o_new);
                                    bC.create<memref::StoreOp>(lC, o_sum, o_buf, ValueRange{r_idx, d_idx});
                                    bC.create<scf::YieldOp>(lC);
                                });
                                bR.create<scf::YieldOp>(lR);
                            });

                            Value next_kvOff = b0.create<arith::AddIOp>(l0, kvOff, tn_i32);
                            Value has_next = b0.create<arith::CmpIOp>(l0, arith::CmpIPredicate::slt, next_kvOff, dimSeq);
                            b0.create<scf::IfOp>(l0, has_next, [&](OpBuilder &bN, Location lN) {
                                Value n_nE = createMinI32(bN, lN, bN.create<arith::SubIOp>(lN, dimSeq, next_kvOff), tn_i32);
                                Value n_nE_idx = bN.create<arith::IndexCastOp>(lN, indexTy, n_nE);
                                Value szN_idx = bN.create<arith::IndexCastOp>(lN, indexTy, bN.create<arith::MulIOp>(lN, tk_i32, n_nE));
                                bN.create<memref::DmaWaitOp>(lN, tagK_n, ValueRange{c0_idx}, szN_idx);
                                bN.create<memref::DmaWaitOp>(lN, tagV_n, ValueRange{c0_idx}, szN_idx);
                                if (hasBias) bN.create<memref::DmaWaitOp>(lN, tagBias_n, ValueRange{c0_idx}, n_nE_idx);
                                bN.create<scf::YieldOp>(lN);
                            });
                            b0.create<scf::YieldOp>(l0);
                        });

                        bKV.create<func::CallOp>(lKV, hwBarrierFn, ValueRange{});

                        Value nextFlatIdx = bKV.create<arith::AddIOp>(lKV, flatIdx, c1_i32);
                        bKV.create<scf::YieldOp>(lKV, ValueRange{nextFlatIdx});
                    });

                    bQ.create<scf::IfOp>(lQ, isCore0, [&](OpBuilder &b0, Location l0) {

                    b0.create<spatz::MatrixRowDivOp>(l0, o_buf, l_val, mE_idx, D_idx, b0.getI64IntegerAttr(HW_Stride_Elements), b0.getI32IntegerAttr(dt_val));

                    Value oSub_out = createSubview3DTo2D(b0, l0, O_output, batch_idx, q_idx, c0_idx, mE_idx, D_idx);
                    Value oL1Sub = createSubview2D(b0, l0, o_buf, c0_idx, c0_idx, mE_idx, D_idx, elemType, l1SpaceAttr);
                    Value szO_idx = b0.create<arith::IndexCastOp>(l0, indexTy, b0.create<arith::MulIOp>(l0, mE, tk_i32));

                    b0.create<memref::DmaStartOp>(l0, oL1Sub, ValueRange{c0_idx, c0_idx}, oSub_out, ValueRange{c0_idx, c0_idx}, szO_idx, tagO, ValueRange{c0_idx});
                    b0.create<memref::DmaWaitOp>(l0, tagO, ValueRange{c0_idx}, szO_idx);
                    b0.create<scf::YieldOp>(l0);
                });

                bQ.create<func::CallOp>(lQ, hwBarrierFn, ValueRange{});
                bQ.create<scf::YieldOp>(lQ);
            });
            bB.create<scf::YieldOp>(lB);
        });

    builder.create<func::CallOp>(loc, l1ResetFn, ValueRange{});

    erasedOps.insert(matmul2Op);
    matmul2Op.erase();

    for (auto* op : opsToErase) {
        op->dropAllReferences();
        erasedOps.insert(op);
        op->erase();
    }

    return success();
}

  LogicalResult lowerBatchMatmul(linalg::BatchMatmulOp op) {
    if (op.getNumInputs() != 2 || op.getNumOutputs() != 1) return failure();

    Value a = op.inputs()[0];
    Value b = op.inputs()[1];
    Value c = op.outputs()[0];

    auto aType = a.getType().dyn_cast<MemRefType>();
    auto bType = b.getType().dyn_cast<MemRefType>();
    auto cType = c.getType().dyn_cast<MemRefType>();

    if (!aType || !bType || !cType || aType.getRank() != 3) return failure();

    OpBuilder builder(op);
    Location loc = op.getLoc();

    Value batchSize;
    if (aType.isDynamicDim(0)) {
      batchSize = builder.create<memref::DimOp>(loc, a, 0);
    } else {
      batchSize = builder.create<arith::ConstantIndexOp>(loc, aType.getDimSize(0));
    }

    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

    auto loop = builder.create<scf::ForOp>(loc, c0, batchSize, c1);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());

    Value b_idx = loop.getInductionVar();

    auto create2DSubview = [&](Value val) -> Value {
      auto type = val.getType().cast<MemRefType>();

      auto getDim = [&](int dim) -> OpFoldResult {
          if (type.isDynamicDim(dim))
              return builder.create<memref::DimOp>(loc, val, dim).getResult();
          return builder.getIndexAttr(type.getDimSize(dim));
      };

      SmallVector<OpFoldResult> offsets = {b_idx, builder.getIndexAttr(0), builder.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1), getDim(1), getDim(2)};
      SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};

      auto targetType = memref::SubViewOp::inferRankReducedResultType(
          2, type, offsets, sizes, strides).cast<MemRefType>();

      return builder.create<memref::SubViewOp>(loc, targetType, val, offsets, sizes, strides);
    };

    Value a2D = create2DSubview(a);
    Value b2D = create2DSubview(b);
    Value c2D = create2DSubview(c);

    builder.create<linalg::MatmulOp>(loc, ValueRange{a2D, b2D}, ValueRange{c2D});

    op.erase();
    return success();
  }

  LogicalResult lowerConv2D(linalg::Conv2DNchwFchwOp op) {
    if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
      return op.emitError("expected two inputs and one output"), failure();
    auto xt = op.inputs()[0].getType().dyn_cast<MemRefType>();
    auto ft = op.inputs()[1].getType().dyn_cast<MemRefType>();
    auto yt = op.outputs()[0].getType().dyn_cast<MemRefType>();
    if (!xt || !ft || !yt)
      return op.emitError("requires bufferized memrefs"), failure();
    if (xt.getElementType().isF32() && ft.getElementType().isF32() &&
        yt.getElementType().isF32()) {

      if (op->hasAttr("quadrilatero.conv_phase_packed"))
        return lowerConv2DPacked(op,false,6);
      return lowerConv2DF32(op);
    }
    if (xt.getElementType().isF16() && ft.getElementType().isF16() &&
        yt.getElementType().isF32()) return lowerConv2DPacked(op, false, 5);
    if (xt.getElementType().isInteger(8) && ft.getElementType().isInteger(8) &&
        yt.getElementType().isF32()) {

      auto format=op->getAttrOfType<StringAttr>("quadrilatero.conv_input_type");
      if (!format || (format.getValue()!="fp8e5m2" && format.getValue()!="fp8e4m3"))
        return op.emitError("FP8 byte carriers require quadrilatero.conv_input_type = fp8e5m2 or fp8e4m3"), failure();
      return lowerConv2DPacked(op,true,format.getValue()=="fp8e4m3" ? 4 : 12);
    }
    return op.emitError("expected f32/f16 or explicitly marked FP8 convolution, with FP32 output"), failure();
  }

  LogicalResult lowerConv2DPacked(linalg::Conv2DNchwFchwOp op, bool fp8, int32_t dataType) {
    if (op.getNumInputs()!=2 || op.getNumOutputs()!=1)
      return op.emitError("requires 2 inputs and 1 output"), failure();

    const bool sequential=true;
    const bool fp32=dataType==6;
    const int64_t pack=fp32 ? 1 : (fp8 ? 4 : 2);
    const int64_t packShift=fp32 ? 0 : (fp8 ? 2 : 1);
    const int64_t elementBytes=4/pack, kShift=6+packShift;
    Value x=op.inputs()[0], filter=op.inputs()[1], y=op.outputs()[0];
    auto xt=x.getType().dyn_cast<MemRefType>();
    auto ft=filter.getType().dyn_cast<MemRefType>();
    auto yt=y.getType().dyn_cast<MemRefType>();
    if (!xt || !ft || !yt || xt.getRank()!=4 || ft.getRank()!=4 ||
        yt.getRank()!=4 || !xt.hasStaticShape() || !ft.hasStaticShape() ||
        !yt.hasStaticShape() ||
        !(fp8 ? xt.getElementType().isInteger(8) :
          (fp32 ? xt.getElementType().isF32() : xt.getElementType().isF16())) ||
        ft.getElementType()!=xt.getElementType() || !yt.getElementType().isF32())
      return op.emitError("packed convolution requires static memref NCHW/FCHW -> FP32"), failure();
    const int64_t batch=xt.getShape()[0], cin=xt.getShape()[1];
    const int64_t h=xt.getShape()[2], w=xt.getShape()[3];
    const int64_t cout=ft.getShape()[0], kh=ft.getShape()[2], kw=ft.getShape()[3];
    const int64_t oh=yt.getShape()[2], ow=yt.getShape()[3];
    int64_t strideH=1,strideW=1;
    if (auto strides=op->getAttrOfType<DenseIntElementsAttr>("strides")) {
      if (strides.getNumElements()!=2) return op.emitError("packed convolution requires two strides"), failure();
      auto values=strides.getValues<APInt>();
      strideH=(*values.begin()).getSExtValue();
      strideW=(*(values.begin()+1)).getSExtValue();
    }
    if (auto dilations=op->getAttrOfType<DenseIntElementsAttr>("dilations"))
      for (APInt value:dilations.getValues<APInt>())
        if (value.getSExtValue()!=1) return op.emitError("packed convolution requires dilation=1"), failure();
    if (batch!=1 || cin<=0 || cout<=0 || h<=0 || w<=0 || kh<=0 || kw<=0 ||
        strideH<=0 || strideW<=0 || h<kh || w<kw || cin%pack ||
        ft.getShape()[1]!=cin || yt.getShape()[0]!=batch || yt.getShape()[1]!=cout ||
        oh!=(h-kh)/strideH+1 || ow!=(w-kw)/strideW+1)
      return op.emitError("packed convolution requires batch=1, Cin divisible by the packing factor, positive strides, valid convolution geometry"), failure();

    constexpr int64_t TM=64, TN=64, PITCH_WORDS=64, PACKED_ROWS=64;
    const int64_t TK=PACKED_ROWS*pack, PACKED_ROW_ELEMENTS=pack*PITCH_WORDS;
    const int64_t mTileStep=ow<=TM ? (TM/ow)*ow : TM;
    const int64_t totalM=oh*ow, totalK=cin*kh*kw;
    const int64_t mt=(totalM+mTileStep-1)/mTileStep;
    const int64_t nt=(cout+TN-1)/TN;
    const int64_t kt=(totalK+TK-1)/TK;

    static_assert(4*PACKED_ROWS*256 + 4096 + 3*TM*256 <= 120*1024,
                  "Packed panels/cache exceed 120 KiB TCDM budget");
    if (cin*h*w*elementBytes > INT32_MAX || cout*cin*kh*kw > INT32_MAX ||
        cout*oh*ow > INT32_MAX ||
        totalM>INT32_MAX || totalK>INT32_MAX ||
        mt*kt*PACKED_ROWS*PACKED_ROW_ELEMENTS>INT32_MAX ||
        nt*kt*PACKED_ROWS*PACKED_ROW_ELEMENTS>INT32_MAX)
      return op.emitError("packed convolution panel count exceeds i32 indexing"), failure();
    auto module=op->getParentOfType<ModuleOp>();
    Value source=filter;
    if (auto cast=source.getDefiningOp<memref::CastOp>()) source=cast.source();
    if (auto alloc=source.getDefiningOp<memref::AllocOp>()) {
      memref::CopyOp definingCopy=nullptr;
      for (Operation *user:source.getUsers())
        if (auto copy=dyn_cast<memref::CopyOp>(user))
          if (copy.getTarget()==source) {
            if (definingCopy) return op.emitError("ambiguous weights copy"), failure();
            definingCopy=copy;
          }
      if (definingCopy) source=definingCopy.getSource();
    }
    if (auto cast=source.getDefiningOp<memref::CastOp>()) source=cast.source();
    auto getGlobal=source.getDefiningOp<memref::GetGlobalOp>();
    if (!getGlobal)
      return op.emitError("packed convolution needs constant weights from memref.get_global"), failure();
    auto symbol=getGlobal->getAttrOfType<FlatSymbolRefAttr>("name");
    if (!symbol) return op.emitError("missing global symbol"), failure();
    auto global=module.lookupSymbol<memref::GlobalOp>(symbol.getValue());
    if (!global) return op.emitError("weights global unavailable"), failure();
    auto weights=global->getAttrOfType<DenseElementsAttr>("initial_value");
    if (!weights || weights.getElementType()!=xt.getElementType() ||
        weights.getNumElements()!=cout*cin*kh*kw)
      return op.emitError("requires dense weights matching the packed input type"), failure();
    SmallVector<APFloat,256> original;
    SmallVector<APInt,256> originalBytes, packedBytes;
    if (fp8) {
      for (APInt v:weights.getValues<APInt>()) originalBytes.push_back(v);
    } else {
      for (APFloat v:weights.getValues<APFloat>()) original.push_back(v);
    }
    const APFloat zero=fp32 ? APFloat(0.0f) : APFloat(APFloat::IEEEhalf(),APInt(16,0));
    SmallVector<APFloat,256> packed;
    if (fp8) packedBytes.reserve(nt*kt*PACKED_ROWS*PACKED_ROW_ELEMENTS);
    else packed.reserve(nt*kt*PACKED_ROWS*PACKED_ROW_ELEMENTS);

    for (int64_t qn=0;qn<nt;++qn)
      for (int64_t qk=0;qk<kt;++qk)
        for (int64_t pk=0;pk<PACKED_ROWS;++pk)
          for (int64_t col=0;col<PITCH_WORDS;++col)
            for (int lane=0;lane<pack;++lane) {
              const int64_t k=qk*TK+pk*pack+lane, n=qn*TN+col;
              if (k<totalK && col<TN && n<cout) {
                const int64_t spatial=k/cin, c=k%cin;
                const int64_t offset=((n*cin+c)*kh+spatial/kw)*kw+spatial%kw;
                if (fp8) packedBytes.push_back(originalBytes[offset]);
                else packed.push_back(original[offset]);
              } else {
                if (fp8) packedBytes.push_back(APInt(8,0));
                else packed.push_back(zero);
              }
            }
    OpBuilder b(op); Location loc=op.getLoc();

    const bool useI32Indices = totalM <= INT32_MAX-256 &&
        cout <= INT32_MAX-256 && totalK <= INT32_MAX/256 &&
        cin <= INT32_MAX/h/w/elementBytes && totalM <= INT32_MAX/cout &&
        nt <= INT32_MAX/kt/PACKED_ROWS/PACKED_ROW_ELEMENTS;
    auto toI32=[&](OpBuilder &ib,Value v)->Value {
      if (v.getType().isIndex())
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getI32Type(),v);
      return v;
    };
    auto addInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::AddIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::AddIOp>(loc,lhs,rhs);
    };
    auto subInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::SubIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::SubIOp>(loc,lhs,rhs);
    };
    auto mulInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::MulIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::MulIOp>(loc,lhs,rhs);
    };
    auto divUInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::DivUIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::DivUIOp>(loc,lhs,rhs);
    };
    auto remUInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::RemUIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::RemUIOp>(loc,lhs,rhs);
    };
    auto cmpInt=[&](OpBuilder &ib,arith::CmpIPredicate pred,
                       Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        lhs=toI32(ib,lhs); rhs=toI32(ib,rhs);
      }
      return ib.createOrFold<arith::CmpIOp>(loc,pred,lhs,rhs);
    };

    auto tileLoop=[&](OpBuilder &ib,Value lower,Value upper,Value step,
                      auto &&body) {
      if (!useI32Indices) {
        ib.create<scf::ForOp>(loc,lower,upper,step,ValueRange{},
            [&](OpBuilder &lb,Location,Value iv,ValueRange){
          body(lb,iv); lb.create<scf::YieldOp>(loc);
        });
        return;
      }
      Value lo32=toI32(ib,lower), hi32=toI32(ib,upper), step32=toI32(ib,step);
      auto wh=ib.create<scf::WhileOp>(loc,TypeRange{ib.getI32Type()},ValueRange{lo32});
      OpBuilder::InsertionGuard guard(ib);
      Block *before=ib.createBlock(&wh.getBefore(),wh.getBefore().end(),
          TypeRange{ib.getI32Type()},SmallVector<Location>{loc});
      ib.setInsertionPointToEnd(before);
      Value iv=before->getArgument(0);
      Value cond=ib.createOrFold<arith::CmpIOp>(loc,arith::CmpIPredicate::slt,iv,hi32);
      ib.create<scf::ConditionOp>(loc,cond,ValueRange{iv});
      Block *after=ib.createBlock(&wh.getAfter(),wh.getAfter().end(),
          TypeRange{ib.getI32Type()},SmallVector<Location>{loc});
      ib.setInsertionPointToEnd(after);
      iv=after->getArgument(0);
      Value indexIV=ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),iv);
      body(ib,indexIV);
      Value next=ib.createOrFold<arith::AddIOp>(loc,iv,step32);
      ib.create<scf::YieldOp>(loc,ValueRange{next});
    };
    auto shrUInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::ShRUIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::ShRUIOp>(loc,lhs,rhs);
    };
    auto andInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.create<arith::AndIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.create<arith::AndIOp>(loc,lhs,rhs);
    };
    auto packedName=std::string(fp8 ? "__quad_conv_fp8_weights_" : "__quad_conv_fp16_weights_")+
       std::to_string(reinterpret_cast<uintptr_t>(op.getOperation()));
    Type packedElement=xt.getElementType();
    const int64_t packedSize=fp8 ? packedBytes.size() : packed.size();
    auto packedTensor=RankedTensorType::get({packedSize},packedElement);
    auto packedAttr=fp8 ? DenseElementsAttr::get(packedTensor,ArrayRef<APInt>(packedBytes))
                       : DenseElementsAttr::get(packedTensor,ArrayRef<APFloat>(packed));
    auto packedType=MemRefType::get({packedSize},packedElement);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      b.create<memref::GlobalOp>(loc,packedName,b.getStringAttr("private"),
          packedType,packedAttr,false,IntegerAttr());
    }
    Value weightsPacked=b.create<memref::GetGlobalOp>(loc,packedType,packedName);
    SymbolTable st(module);
    auto getFn=[&](StringRef name,FunctionType type) {
      if (auto fn=st.lookup<func::FuncOp>(name)) return fn;
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      auto fn=b.create<func::FuncOp>(loc,name,type);
      fn.setPrivate(); st.insert(fn); return fn;
    };
    auto coreFn=getFn("snrt_cluster_core_idx",b.getFunctionType({},{b.getI32Type()}));
    auto barrierFn=getFn("snrt_cluster_hw_barrier",b.getFunctionType({},{}));
    auto beginFn=getFn("conv_lowering_begin",b.getFunctionType({},{}));
    auto endFn=getFn("conv_lowering_end",b.getFunctionType({},{}));

    Value core=b.create<func::CallOp>(loc,coreFn,ValueRange{}).getResult(0);
    Value c0i=b.create<arith::ConstantIntOp>(loc,0,32);
    Value c1i=b.create<arith::ConstantIntOp>(loc,1,32);
    Value isCore0=cmpInt(b,arith::CmpIPredicate::eq,core,c0i);
    Value isCore1=cmpInt(b,arith::CmpIPredicate::eq,core,c1i);
    auto ci=[&](int64_t v)->Value {return b.create<arith::ConstantIndexOp>(loc,v);};
    Value z=ci(0),one=ci(1),tmStep=ci(mTileStep),tnStep=ci(TN),tkStep=ci(TK);
    Value mEnd=ci(totalM),nEnd=ci(cout),kEnd=ci(totalK);
    auto l1=IntegerAttr::get(b.getI64Type(),1);
    auto packedTileTy=MemRefType::get({PACKED_ROWS,PACKED_ROW_ELEMENTS},packedElement,
                                    MemRefLayoutAttrInterface{},l1);
    auto floatTileTy=MemRefType::get({TM,PITCH_WORDS},b.getF32Type(),
                                     MemRefLayoutAttrInterface{},l1);
    Value aT0=b.create<memref::AllocOp>(loc,packedTileTy);
    Value aT1=b.create<memref::AllocOp>(loc,packedTileTy);

    const int64_t B_CACHE_ELEMENTS = 2*PACKED_ROWS*PACKED_ROW_ELEMENTS + 4096/elementBytes;
    const bool residentWeights = nt <= B_CACHE_ELEMENTS/PACKED_ROW_ELEMENTS/(totalK/pack);
    Value weightsL1,bT0,bT1;
    if (residentWeights) {
      auto cacheTy=MemRefType::get({nt*(totalK/pack),PACKED_ROW_ELEMENTS},packedElement,
                                  MemRefLayoutAttrInterface{},l1);
      weightsL1=b.create<memref::AllocOp>(loc,cacheTy);
    } else {
      bT0=b.create<memref::AllocOp>(loc,packedTileTy);
      bT1=b.create<memref::AllocOp>(loc,packedTileTy);
    }
    Value cAcc=b.create<memref::AllocOp>(loc,floatTileTy);
    Value cP0=b.create<memref::AllocOp>(loc,floatTileTy);
    Value cP1=b.create<memref::AllocOp>(loc,floatTileTy);
    Value dmaTagA=b.create<memref::AllocaOp>(loc,MemRefType::get({1},b.getI32Type()));
    Value dmaTagB=b.create<memref::AllocaOp>(loc,MemRefType::get({1},b.getI32Type()));

    Value dmaLastCount=b.create<memref::AllocaOp>(loc,MemRefType::get({1},b.getI32Type()));
    Value dmaTagOut=b.create<memref::AllocaOp>(loc,MemRefType::get({1},b.getI32Type()));

    b.create<func::CallOp>(loc,beginFn,ValueRange{});

    auto inputFlatTy=MemRefType::get({cin*h*w},packedElement,
                                    MemRefLayoutAttrInterface{},xt.getMemorySpace());
    Value inputFlat=b.create<memref::ReinterpretCastOp>(loc,inputFlatTy,x,z,
         SmallVector<OpFoldResult,1>{ci(cin*h*w)},
         SmallVector<OpFoldResult,1>{one}).getResult();
    auto l1flatTy=MemRefType::get({PACKED_ROWS*PACKED_ROW_ELEMENTS},packedElement,
                                  MemRefLayoutAttrInterface{},l1);
    auto pick=[&](OpBuilder &ob,Value parity,Value first,Value second,
                  MemRefType ty)->Value {
      Value even=cmpInt(ob,arith::CmpIPredicate::eq,
                                         parity,z);
      auto sel=ob.create<scf::IfOp>(loc,TypeRange{ty},even,
         [&](OpBuilder &yes,Location){
           yes.create<scf::YieldOp>(loc,ValueRange{first});
         },[&](OpBuilder &no,Location){
           no.create<scf::YieldOp>(loc,ValueRange{second});
         });
      return sel.getResult(0);
    };
    auto parityOf=[&](OpBuilder &ob,Value k0)->Value {
      Value panel=shrUInt(ob,k0,ci(kShift));
      return andInt(ob,panel,one);
    };
    auto minIndex=[&](OpBuilder &ob,Value a,Value v)->Value {
      Value cmp=cmpInt(ob,arith::CmpIPredicate::slt,a,v);
      return ob.createOrFold<arith::SelectOp>(loc,cmp,a,v);
    };

    auto tileExtent=[&](OpBuilder &qb,Value start,int64_t extent,int64_t step)->Value {
      auto constant=[&](int64_t v)->Value {
        return qb.create<arith::ConstantIndexOp>(loc,v);
      };
      if (extent<=step) return constant(extent);
      const int64_t tail=extent%step;
      if (tail==0) return constant(step);
      Value last=cmpInt(qb,arith::CmpIPredicate::eq,start,constant(extent-tail));
      return qb.createOrFold<arith::SelectOp>(loc,last,constant(tail),constant(step));
    };
    auto loop=[&](OpBuilder &ob,Value end,Value step,auto &&body) {
      tileLoop(ob,z,end,step,[&](OpBuilder &lb,Value iv){body(lb,iv);});
    };

    auto dmaC32=[&](OpBuilder &ob,int64_t v)->Value {
      return ob.create<arith::ConstantIntOp>(loc,v,32);
    };
    auto dmaI32=[&](OpBuilder &ob,Value v)->Value {
      return ob.createOrFold<arith::IndexCastOp>(loc,ob.getI32Type(),v);
    };
    auto dmaIndex=[&](OpBuilder &ob,Value v)->Value {
      return ob.createOrFold<arith::IndexCastOp>(loc,ob.getIndexType(),v);
    };
    auto dmaLoop32=[&](OpBuilder &ob,Value upper,auto &&body) {
      auto wh=ob.create<scf::WhileOp>(loc,TypeRange{ob.getI32Type()},
                                    ValueRange{dmaC32(ob,0)});
      OpBuilder::InsertionGuard guard(ob);
      Block *before=ob.createBlock(&wh.getBefore(),wh.getBefore().end(),
          TypeRange{ob.getI32Type()},SmallVector<Location>{loc});
      ob.setInsertionPointToEnd(before);
      Value iv=before->getArgument(0);
      Value pred=cmpInt(ob,arith::CmpIPredicate::ult,iv,upper);
      ob.create<scf::ConditionOp>(loc,pred,ValueRange{iv});
      Block *after=ob.createBlock(&wh.getAfter(),wh.getAfter().end(),
          TypeRange{ob.getI32Type()},SmallVector<Location>{loc});
      ob.setInsertionPointToEnd(after);
      iv=after->getArgument(0); body(ob,iv);
      Value next=addInt(ob,iv,dmaC32(ob,1));
      ob.create<scf::YieldOp>(loc,ValueRange{next});
    };
    auto barrier=[&](OpBuilder &ob) {
      ob.create<func::CallOp>(loc,barrierFn,ValueRange{});
    };

    Value c0Buf=cAcc, c1Buf=cP0, c2Buf=cP1;
    auto tileTy=packedTileTy;
    auto outTileTy=floatTileTy;
    Value at0=aT0, at1=aT1, gt0=bT0, gt1=bT1;
    Value mStep=tmStep, tn=tnStep, tk=tkStep;
    constexpr int64_t PITCH=PITCH_WORDS;
    Value cOHOW=ci(oh*ow), cCout=ci(cout);
    auto sync=[&](OpBuilder &ob) {
      ob.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &db,Location){
        emitConvSequentialDrain(db,loc);
        db.create<scf::YieldOp>(loc);
      });
      barrier(ob);
    };
    auto chooseC=[&](OpBuilder &ob,Value slot)->Value {
      Value isZero=cmpInt(ob,arith::CmpIPredicate::eq,slot,z);
      auto result=ob.create<scf::IfOp>(loc,TypeRange{outTileTy},isZero,
          [&](OpBuilder &yes,Location) {
            yes.create<scf::YieldOp>(loc,ValueRange{c0Buf});
          },[&](OpBuilder &no,Location) {
            Value second=pick(no,subInt(no,slot,one),
                              c1Buf,c2Buf,outTileTy);
            no.create<scf::YieldOp>(loc,ValueRange{second});
          });
      return result.getResult(0);
    };
    auto chooseAB=[&](OpBuilder &ob,Value k0,Value first,Value second,
                      MemRefType ty)->Value {
      return pick(ob,parityOf(ob,k0),first,second,ty);
    };
    auto selectB=[&](OpBuilder &ob,Value k0,Value effectiveK)->Value {
      if (residentWeights) return weightsL1;
      return chooseAB(ob,effectiveK,gt0,gt1,tileTy);
    };

    Value yFlat=b.create<memref::CollapseShapeOp>(loc,
        MemRefType::get({batch*cout*oh*ow},b.getF32Type()),y,
        SmallVector<ReassociationIndices>{{0,1,2,3}});

    auto staticTiles=[&](OpBuilder &q,int64_t extent,int64_t step,auto &&body) {
      auto constant=[&](OpBuilder &qb,int64_t v)->Value {
        return qb.create<arith::ConstantIndexOp>(loc,v);
      };
      const int64_t fullEnd=extent/step*step;
      if (fullEnd == step) {
        body(q,constant(q,0),constant(q,step));
      } else if (fullEnd != 0) {
        tileLoop(q,constant(q,0),constant(q,fullEnd),constant(q,step),
            [&](OpBuilder &qb,Value iv) { body(qb,iv,constant(qb,step)); });
      }
      if (fullEnd != extent)
        body(q,constant(q,fullEnd),constant(q,extent-fullEnd));
    };
    staticTiles(b,totalM,mTileStep,[&](OpBuilder &bm,Value m0,Value mE){
      staticTiles(bm,cout,TN,[&](OpBuilder &bn,Value n0,Value nE){

        Value tileM=divUInt(bn,m0,mStep);
        Value tileN=divUInt(bn,n0,tn);
        Value tileSerial=addInt(bn,
            mulInt(bn,tileM,
                ci((cout+TN-1)/TN)),tileN);

        Value slot=remUInt(bn,
            mulInt(bn,tileSerial,ci(kt%2 ? 1 : 2)),ci(3));

        Value abShift = z;
        Value nextAbShift = z;
        if (kt % 2 != 0) {
          abShift=mulInt(bn,
              remUInt(bn,tileSerial,ci(2)),tk);
          nextAbShift=subInt(bn,tk,abShift);
        }
        Value candidateN=addInt(bn,n0,tn);
        Value sameM=cmpInt(bn,arith::CmpIPredicate::ult,candidateN,nEnd);
        Value nextTileN=bn.createOrFold<arith::SelectOp>(loc,sameM,candidateN,z);
        Value nextTileM=bn.createOrFold<arith::SelectOp>(loc,sameM,m0,addInt(bn,m0,mStep));
        const bool crossTilePrefetch = !sequential && (cout<=TN || cout%TN==0);
        const bool directIm2colDMA = true;
        const bool overlapNextMatmul = crossTilePrefetch && kt>=2;
        const bool staticDirectPanels = useI32Indices &&
            ((kh==3 && kw==3 && kt<=9) ||
             (strideH==1 && strideW==1 && kt<=32 && kh*kw<=49));
        const bool deferOutputWait = overlapNextMatmul;
        Value acc=chooseC(bn,slot);
        Value partial0=chooseC(bn,remUInt(bn,
            addInt(bn,slot,ci(1)),ci(3)));
        Value partial1=chooseC(bn,remUInt(bn,
            addInt(bn,slot,ci(2)),ci(3)));
        Value nextAccumulator=(kt%2 ? partial0 : partial1);

        auto waitDirectOperands = [&](OpBuilder &d,Value k0,Value targetM) {
          Value count32=d.create<memref::LoadOp>(loc,dmaLastCount,ValueRange{z});
          Value dmaCount=dmaIndex(d,count32);

          d.create<memref::DmaWaitOp>(loc,dmaTagA,ValueRange{z},dmaCount);
        };
        auto prepareOperandsImpl = [&](OpBuilder &d,Value k0,Value targetM,
                                       Value targetME,Value targetN,Value targetAbShift,
                                       bool deferWait) {

          if (auto constant=k0.getDefiningOp<arith::ConstantIndexOp>())
            if (constant.value() >= totalK) return;
          if (auto constant=targetME.getDefiningOp<arith::ConstantIndexOp>())
            if (constant.value() <= 0) return;
          Value effectiveK=addInt(d,k0,targetAbShift);
          Value activeA=chooseAB(d,effectiveK,at0,at1,tileTy);
          Value activeB=selectB(d,k0,effectiveK);
          Value kE=totalK%TK==0 ? tkStep : minIndex(d,subInt(d,kEnd,k0),tkStep);
          Value qk=shrUInt(d,k0,ci(kShift));
          Value qn=divUInt(d,targetN,tn);
          Value targetNE=tileExtent(d,targetN,cout,TN);
          Value aFlat=d.create<memref::CollapseShapeOp>(loc,l1flatTy,activeA,
                  SmallVector<ReassociationIndices>{{0,1}});
          auto issueWeights=[&](OpBuilder &q,Value rows,Value dstBase,
                                  Value dest,Value srcBase) {

            auto map=AffineMapAttr::get(AffineMap::get(2,1,
                getAffineDimExpr(0,q.getContext())*PITCH_WORDS+
                getAffineDimExpr(1,q.getContext())+getAffineSymbolExpr(0,q.getContext())));
            auto st=MemRefType::get({ShapedType::kDynamicSize,ShapedType::kDynamicSize},
                                    packedElement,map,packedType.getMemorySpace());
            auto dt=MemRefType::get({ShapedType::kDynamicSize,ShapedType::kDynamicSize},
                                    packedElement,map,l1);
            Value src=q.create<memref::ReinterpretCastOp>(loc,st,weightsPacked,srcBase,
                SmallVector<OpFoldResult,2>{rows,targetNE},
                SmallVector<OpFoldResult,2>{ci(PITCH_WORDS),one}).getResult();
            Value dst=q.create<memref::ReinterpretCastOp>(loc,dt,dest,dstBase,
                SmallVector<OpFoldResult,2>{rows,targetNE},
                SmallVector<OpFoldResult,2>{ci(PITCH_WORDS),one}).getResult();
            Value count=mulInt(q,rows,targetNE);
            q.create<memref::DmaStartOp>(loc,src,ValueRange{z,z},dst,ValueRange{z,z},
                                        count,dmaTagB,ValueRange{z});
          };
          if (residentWeights) {
            Value firstM=cmpInt(d,arith::CmpIPredicate::eq,targetM,z);
            Value firstK=cmpInt(d,arith::CmpIPredicate::eq,k0,z);
            Value first=andInt(d,firstM,firstK);
            d.create<scf::IfOp>(loc,first,[&](OpBuilder &q,Location){
              Value sourceBase=mulInt(q,qn,ci(kt*PACKED_ROWS*PACKED_ROW_ELEMENTS));
              Value destBase=mulInt(q,qn,ci((totalK/pack)*PACKED_ROW_ELEMENTS));
              issueWeights(q,kEnd,destBase,weightsL1,sourceBase);
              q.create<scf::YieldOp>(loc);
            });
          } else {
            Value panel=addInt(d,
                mulInt(d,qn,ci(kt)),qk);
            Value sourceBase=mulInt(d,panel,ci(PACKED_ROWS*PACKED_ROW_ELEMENTS));
            issueWeights(d,kE,z,activeB,sourceBase);
          }

          DataLayoutAnalysis layoutAnalysis(module);
          LowerToLLVMOptions addressOptions(d.getContext(),layoutAnalysis.getAtOrAbove(op.getOperation()));
          LLVMTypeConverter addressConverter(d.getContext(),addressOptions,&layoutAnalysis);
          auto byteAddress=[&](Value memref)->Value {
            Type descTy=addressConverter.convertType(memref.getType());
            auto fields=descTy.cast<LLVM::LLVMStructType>().getBody();
            Value desc=d.create<UnrealizedConversionCastOp>(loc,TypeRange{descTy},ValueRange{memref}).getResult(0);
            Value aligned=d.create<LLVM::ExtractValueOp>(loc,fields[1],desc,d.getI64ArrayAttr({1}));
            Value offset=d.create<LLVM::ExtractValueOp>(loc,fields[2],desc,d.getI64ArrayAttr({2}));
            unsigned bits=offset.getType().cast<IntegerType>().getWidth();
            Value off32=offset;
            if (bits>32) off32=d.create<LLVM::TruncOp>(loc,d.getI32Type(),offset);
            else if (bits<32) off32=d.create<LLVM::ZExtOp>(loc,d.getI32Type(),offset);
            Value base=d.create<LLVM::PtrToIntOp>(loc,d.getI32Type(),aligned);
            return addInt(d,base,mulInt(d,off32,dmaC32(d,elementBytes)));
          };
          Value sourceBytes=byteAddress(inputFlat), targetBytes=byteAddress(aFlat);
          auto emitGroup=[&](OpBuilder &pb,Value srcWord,Value dstWord,Value commands,
                             Value reps,Value srcPitch,Value dstPitch,
                             Value srcStep,Value dstStep,Value widthWords,bool single) {
            Value src=addInt(pb,sourceBytes,mulInt(pb,srcWord,dmaC32(pb,4)));
            Value dst=addInt(pb,targetBytes,mulInt(pb,dstWord,dmaC32(pb,4)));
            Value width=mulInt(pb,widthWords,dmaC32(pb,4));
            OperationState state(loc,LLVM::InlineAsmOp::getOperationName());
            state.addTypes(pb.getI32Type());
            if (single) {
              state.addOperands(SmallVector<Value,7>{src,dst,reps,mulInt(pb,srcPitch,dmaC32(pb,4)),
                  mulInt(pb,dstPitch,dmaC32(pb,4)),width,dmaC32(pb,2)});
              state.addAttribute("asm_string",pb.getStringAttr(
                  "dmstr $4, $5\n\tdmrep $3\n\tdmsrc $1, zero\n\tdmdst $2, zero\n\tdmcpy $0, $6, $7"));
              state.addAttribute("constraints",pb.getStringAttr("=&r,r,r,r,r,r,r,r,~{memory}"));
            } else {
              state.addOperands(SmallVector<Value,10>{src,dst,commands,reps,mulInt(pb,srcPitch,dmaC32(pb,4)),
                  mulInt(pb,dstPitch,dmaC32(pb,4)),mulInt(pb,srcStep,dmaC32(pb,4)),mulInt(pb,dstStep,dmaC32(pb,4)),width,dmaC32(pb,2)});
              state.addAttribute("asm_string",pb.getStringAttr(
                  "mv t0, $1\n\tmv t1, $2\n\tmv t2, $3\n\t"
                  "dmstr $5, $6\n\tdmrep $4\n\t"
                  "1:\n\tdmsrc t0, zero\n\tdmdst t1, zero\n\tdmcpy $0, $9, $10\n\t"
                  "add t0, t0, $7\n\tadd t1, t1, $8\n\taddi t2, t2, -1\n\tbnez t2, 1b"));
              state.addAttribute("constraints",pb.getStringAttr(
                  "=&r,r,r,r,r,r,r,r,r,r,r,~{t0},~{t1},~{t2},~{memory}"));
            }
            state.addAttribute("has_side_effects",pb.getUnitAttr());
            Value last=pb.create(state)->getResult(0);
            pb.create<memref::StoreOp>(loc,last,dmaTagA,ValueRange{z});
            Value elements=mulInt(pb,mulInt(pb,widthWords,reps),dmaC32(pb,pack));
            pb.create<memref::StoreOp>(loc,elements,dmaLastCount,ValueRange{z});
          };
          Value m32=dmaI32(d,targetM), me32=dmaI32(d,targetME);
          Value outRow32=divUInt(d,m32,dmaC32(d,ow));
          const bool wholeRows = ow<=TM;

          auto emitSpatialGroup=[&](OpBuilder &g,Value src,Value dst,Value run,
                                    Value plane,Value phaseW,int64_t knownRun) {
            if (wholeRows) {
              Value rows=divUInt(g,me32,dmaC32(g,ow));
              auto byChannels=[&](OpBuilder &q) {
                emitGroup(q,src,dst,rows,run,plane,dmaC32(q,PITCH_WORDS),phaseW,dmaC32(q,ow),dmaC32(q,ow),mTileStep==ow);
              };
              auto byRows=[&](OpBuilder &q) {
                emitGroup(q,src,dst,run,rows,phaseW,dmaC32(q,ow),plane,dmaC32(q,PITCH_WORDS),dmaC32(q,ow),knownRun==1);
              };
              if (auto constant=targetME.getDefiningOp<arith::ConstantIndexOp>()) {
                const int64_t knownRows=constant.value()/ow;
                if (knownRun>0 && knownRun>=knownRows) byChannels(g);
                else if (knownRun>0) byRows(g);
                else {
                  Value byChannel=cmpInt(g,arith::CmpIPredicate::uge,run,dmaC32(g,knownRows));
                  g.create<scf::IfOp>(loc,byChannel,[&](OpBuilder &q,Location) {
                    byChannels(q); q.create<scf::YieldOp>(loc);
                  },[&](OpBuilder &q,Location) {
                    byRows(q); q.create<scf::YieldOp>(loc);
                  });
                }
              } else if (knownRun>=mTileStep/ow) byChannels(g);
              else if (knownRun==1) byRows(g);
              else {
                Value groupChannels=cmpInt(g,arith::CmpIPredicate::ugt,run,rows);
                g.create<scf::IfOp>(loc,groupChannels,[&](OpBuilder &q,Location) {
                  byChannels(q); q.create<scf::YieldOp>(loc);
                },[&](OpBuilder &q,Location) {
                  byRows(q); q.create<scf::YieldOp>(loc);
                });
              }
            } else {
              Value firstCol=remUInt(g,m32,dmaC32(g,ow));
              Value segments=divUInt(g,addInt(g,addInt(g,firstCol,me32),dmaC32(g,ow-1)),dmaC32(g,ow));
              dmaLoop32(g,segments,[&](OpBuilder &v,Value r) {
                Value rowStart=mulInt(v,r,dmaC32(v,ow));
                Value begin=v.createOrFold<arith::SelectOp>(loc,cmpInt(v,arith::CmpIPredicate::eq,r,dmaC32(v,0)),firstCol,rowStart);
                Value end=minIndex(v,addInt(v,rowStart,dmaC32(v,ow)),addInt(v,firstCol,me32));
                Value len=subInt(v,end,begin), local=subInt(v,begin,firstCol);
                Value col=remUInt(v,begin,dmaC32(v,ow));
                Value source=addInt(v,src,addInt(v,mulInt(v,r,phaseW),col));
                Value dest=addInt(v,dst,local);
                emitGroup(v,source,dest,dmaC32(v,1),run,plane,dmaC32(v,PITCH_WORDS),
                          dmaC32(v,0),dmaC32(v,0),len,true);
              });
            }
          };

          if (kh==1 && kw==1 && strideH==1 && strideW==1) {

            Value firstPair=shrUInt(d,dmaI32(d,k0),dmaC32(d,packShift));
            Value pairs=shrUInt(d,dmaI32(d,kE),dmaC32(d,packShift));
            Value src=addInt(d,mulInt(d,firstPair,dmaC32(d,h*w)),m32);
            emitGroup(d,src,dmaC32(d,0),dmaC32(d,1),pairs,
                      dmaC32(d,h*w),dmaC32(d,PITCH_WORDS),
                      dmaC32(d,0),dmaC32(d,0),me32,true);
          } else if (cin%TK==0 && useI32Indices) {

            Value k32=dmaI32(d,k0);
            Value spatial=divUInt(d,k32,dmaC32(d,cin));
            Value cp=shrUInt(d,remUInt(d,k32,dmaC32(d,cin)),dmaC32(d,packShift));
            Value ky=divUInt(d,spatial,dmaC32(d,kw));
            Value kx=remUInt(d,spatial,dmaC32(d,kw));
            Value py=remUInt(d,ky,dmaC32(d,strideH));
            Value px=remUInt(d,kx,dmaC32(d,strideW));
            Value phaseH=h%strideH==0 ? dmaC32(d,h/strideH) :
                addInt(d,divUInt(d,subInt(d,dmaC32(d,h-1),py),dmaC32(d,strideH)),dmaC32(d,1));
            Value phaseW=w%strideW==0 ? dmaC32(d,w/strideW) :
                addInt(d,divUInt(d,subInt(d,dmaC32(d,w-1),px),dmaC32(d,strideW)),dmaC32(d,1));
            Value prefixH=addInt(d,mulInt(d,py,dmaC32(d,h/strideH)),minIndex(d,py,dmaC32(d,h%strideH)));
            Value prefixW=addInt(d,mulInt(d,px,dmaC32(d,w/strideW)),minIndex(d,px,dmaC32(d,w%strideW)));
            Value phaseBase=mulInt(d,addInt(d,mulInt(d,prefixH,dmaC32(d,w)),mulInt(d,phaseH,prefixW)),dmaC32(d,cin/pack));
            Value plane=mulInt(d,phaseH,phaseW);
            Value row=addInt(d,outRow32,divUInt(d,ky,dmaC32(d,strideH)));
            Value col=divUInt(d,kx,dmaC32(d,strideW));
            Value src=addInt(d,phaseBase,addInt(d,mulInt(d,cp,plane),addInt(d,mulInt(d,row,phaseW),col)));
            emitSpatialGroup(d,src,dmaC32(d,0),dmaC32(d,PACKED_ROWS),plane,phaseW,PACKED_ROWS);
          } else if (staticDirectPanels) {
            auto emitStaticPanel=[&](OpBuilder &pb,int64_t panelK) {
              const int64_t first=panelK/pack, end=first+std::min<int64_t>(TK,totalK-panelK)/pack;
              const int64_t pairs=cin/pack;
              for (int64_t spatial=first/pairs;spatial<=(end-1)/pairs;++spatial) {
                const int64_t start=std::max<int64_t>(first,spatial*pairs);
                const int64_t stop=std::min<int64_t>(end,(spatial+1)*pairs);
                const int64_t run=stop-start, cp=start-spatial*pairs, localK=start-first;
                const int64_t ky=spatial/kw,kx=spatial%kw;

                const int64_t py=ky%strideH, px=kx%strideW;
                const int64_t phaseH=(h-1-py)/strideH+1, phaseW=(w-1-px)/strideW+1;
                const int64_t prefixH=py*(h/strideH)+std::min<int64_t>(py,h%strideH);
                const int64_t prefixW=px*(w/strideW)+std::min<int64_t>(px,w%strideW);
                const int64_t phaseBase=pairs*(prefixH*w+phaseH*prefixW);
                const int64_t plane=phaseH*phaseW;
                Value row=addInt(pb,outRow32,dmaC32(pb,ky/strideH));
                Value src=addInt(pb,dmaC32(pb,phaseBase+cp*plane+kx/strideW),mulInt(pb,row,dmaC32(pb,phaseW)));
                Value dst=dmaC32(pb,localK*PITCH_WORDS);
                emitSpatialGroup(pb,src,dst,dmaC32(pb,run),dmaC32(pb,plane),dmaC32(pb,phaseW),run);
              }
            };
            auto dispatch=[&](auto &&self,OpBuilder &pb,int64_t panel)->void {
              if (auto constant=k0.getDefiningOp<arith::ConstantIndexOp>()) {
                emitStaticPanel(pb,constant.value());return;
              }
              if (panel==kt-1) {emitStaticPanel(pb,panel*TK);return;}
              Value match=cmpInt(pb,arith::CmpIPredicate::eq,k0,ci(panel*TK));
              pb.create<scf::IfOp>(loc,match,[&](OpBuilder &yes,Location) {
                emitStaticPanel(yes,panel*TK);yes.create<scf::YieldOp>(loc);
              },[&](OpBuilder &no,Location) {
                self(self,no,panel+1);no.create<scf::YieldOp>(loc);
              });
            };
            dispatch(dispatch,d,0);
          } else {
            Value first32=shrUInt(d,dmaI32(d,k0),dmaC32(d,packShift));
            Value count32=shrUInt(d,dmaI32(d,kE),dmaC32(d,packShift));
            Value end32=addInt(d,first32,count32);
            Value spatialFirst=divUInt(d,first32,dmaC32(d,cin/pack));
            Value spatialLast=divUInt(d,subInt(d,end32,dmaC32(d,1)),dmaC32(d,cin/pack));
            Value spatialCount=addInt(d,subInt(d,spatialLast,spatialFirst),dmaC32(d,1));
            dmaLoop32(d,spatialCount,[&](OpBuilder &g,Value si) {
              Value spatial=addInt(g,spatialFirst,si);
              Value base=mulInt(g,spatial,dmaC32(g,cin/pack));
              Value start=g.createOrFold<arith::SelectOp>(loc,cmpInt(g,arith::CmpIPredicate::ult,base,first32),first32,base);
              Value stop=minIndex(g,addInt(g,base,dmaC32(g,cin/pack)),end32);
              Value run=subInt(g,stop,start), cp=subInt(g,start,base), localK=subInt(g,start,first32);
              Value ky=divUInt(g,spatial,dmaC32(g,kw)), kx=remUInt(g,spatial,dmaC32(g,kw));
              Value py=remUInt(g,ky,dmaC32(g,strideH)), px=remUInt(g,kx,dmaC32(g,strideW));
              Value phaseH=addInt(g,divUInt(g,subInt(g,dmaC32(g,h-1),py),dmaC32(g,strideH)),dmaC32(g,1));
              Value phaseW=addInt(g,divUInt(g,subInt(g,dmaC32(g,w-1),px),dmaC32(g,strideW)),dmaC32(g,1));
              Value prefixH=addInt(g,mulInt(g,py,dmaC32(g,h/strideH)),minIndex(g,py,dmaC32(g,h%strideH)));
              Value prefixW=addInt(g,mulInt(g,px,dmaC32(g,w/strideW)),minIndex(g,px,dmaC32(g,w%strideW)));
              Value phaseBase=mulInt(g,addInt(g,mulInt(g,prefixH,dmaC32(g,w)),mulInt(g,phaseH,prefixW)),dmaC32(g,cin/pack));
              Value plane=mulInt(g,phaseH,phaseW);
              Value row=addInt(g,outRow32,divUInt(g,ky,dmaC32(g,strideH)));
              Value col=divUInt(g,kx,dmaC32(g,strideW));
              Value src=addInt(g,phaseBase,addInt(g,mulInt(g,cp,plane),addInt(g,mulInt(g,row,phaseW),col)));
              Value dst=mulInt(g,localK,dmaC32(g,PITCH_WORDS));
              emitSpatialGroup(g,src,dst,run,plane,phaseW,0);
            });
          }
          if (!deferWait) waitDirectOperands(d,k0,targetM);
        };
        auto prepareOperands = [&](OpBuilder &d,Value k0,Value targetM,
                                   Value targetME,Value targetN,Value targetAbShift) {
          prepareOperandsImpl(d,k0,targetM,targetME,targetN,targetAbShift,false);
        };
        auto emitPackedMatmul = [&](OpBuilder &q,Value targetM,Value k0,
                                    Value me,Value ke,Value targetN,Value a,Value bb,Value dest) {
          Value targetNE=tileExtent(q,targetN,cout,TN);

          auto logicalLayout=AffineMapAttr::get(AffineMap::get(2,1,
              getAffineDimExpr(0,q.getContext())*PACKED_ROW_ELEMENTS+
              getAffineDimExpr(1,q.getContext())+
              getAffineSymbolExpr(0,q.getContext())));
          auto logicalTy=MemRefType::get({TK,PACKED_ROW_ELEMENTS},packedElement,
              logicalLayout,l1);
          auto logicalView=[&](Value physical,Value base)->Value {
            return q.create<memref::ReinterpretCastOp>(loc,logicalTy,
                physical,base,SmallVector<OpFoldResult,2>{ci(TK),ci(PACKED_ROW_ELEMENTS)},
                SmallVector<OpFoldResult,2>{ci(PACKED_ROW_ELEMENTS),one}).getResult();
          };
          Value bBase=z;
          if (residentWeights) {
            Value qn=divUInt(q,targetN,tn);
            bBase=addInt(q,
                mulInt(q,qn,ci((totalK/pack)*PACKED_ROW_ELEMENTS)),
                mulInt(q,k0,ci(PITCH_WORDS)));
          }
          Value as=createSubview2D(q,loc,logicalView(a,z),z,z,ke,me,packedElement,l1);
          Value bs=createSubview2D(q,loc,logicalView(bb,bBase),z,z,ke,targetNE,packedElement,l1);
          Value cs=createSubview2D(q,loc,dest,z,z,me,targetNE,q.getF32Type(),l1);
          q.create<quadrilatero::TcdmMatmulMemRefOp>(loc,as,bs,cs,
              me,targetNE,ke,ci(packShift),q.getI32IntegerAttr(dataType),q.getI32IntegerAttr(dataType),
              q.getI32IntegerAttr(dataType));
        };

        auto accumulatePanel=[&](OpBuilder &ib,Value panel){
          Value parity=remUInt(ib,panel,ci(2));
          Value odd=cmpInt(ib,arith::CmpIPredicate::ne,parity,z);
          Value accSub=createSubview2D(ib,loc,acc,z,z,mE,nE,ib.getF32Type(),l1);

          ib.create<scf::IfOp>(loc,odd,[&](OpBuilder &ob,Location){
            Value pSub=createSubview2D(ob,loc,partial0,z,z,mE,nE,ob.getF32Type(),l1);
            ob.create<spatz::MatrixAddOp>(loc, accSub, pSub, mE, nE,
                ob.getI64IntegerAttr(64), ob.getI32IntegerAttr(6));
            ob.create<scf::YieldOp>(loc);
          },[&](OpBuilder &eb,Location){
            Value pSub=createSubview2D(eb,loc,partial1,z,z,mE,nE,eb.getF32Type(),l1);
            eb.create<spatz::MatrixAddOp>(loc, accSub, pSub, mE, nE,
                eb.getI64IntegerAttr(64), eb.getI32IntegerAttr(6));
            eb.create<scf::YieldOp>(loc);
          });
        };

        bn.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){
          if (crossTilePrefetch) {
            Value firstTile=cmpInt(ib,
                arith::CmpIPredicate::eq,tileSerial,z);
            ib.create<scf::IfOp>(loc,firstTile,[&](OpBuilder &pr,Location){
              prepareOperands(pr,z,m0,mE,n0,abShift);
              pr.create<scf::YieldOp>(loc);
            });
          } else {
            prepareOperands(ib,z,m0,mE,n0,abShift);
          }
          ib.create<scf::YieldOp>(loc);
        });
        Value kBegin=z;
        if (overlapNextMatmul) {

          Value firstSpatialTile=cmpInt(bn,
              arith::CmpIPredicate::eq,tileSerial,z);
          bn.create<scf::IfOp>(loc,firstSpatialTile,[&](OpBuilder &ib,Location){
            sync(ib);
            ib.create<scf::YieldOp>(loc);
          });
          kBegin=bn.createOrFold<arith::SelectOp>(loc,firstSpatialTile,z,tk);
        } else {
          sync(bn);
        }

        auto kStages=[&](auto &&body) {
          const int64_t fullEnd=totalK/TK*TK;
          if (fullEnd == TK && totalK == TK) {
            body(bn,z,tk);
          } else if (fullEnd != 0) {
            Value end=bn.create<arith::ConstantIndexOp>(loc,fullEnd);
            tileLoop(bn,kBegin,end,tk,[&](OpBuilder &kb,Value k0) {
              body(kb,k0,tk);
            });
          }
          if (fullEnd != totalK) {
            Value last=bn.create<arith::ConstantIndexOp>(loc,fullEnd);
            Value extent=bn.create<arith::ConstantIndexOp>(loc,totalK-fullEnd);
            body(bn,last,extent);
          }
        };
        kStages([&](OpBuilder &bk,Value k0,Value kE){
          Value qk=divUInt(bk,k0,tk);
          Value isFirst=cmpInt(bk,arith::CmpIPredicate::eq,qk,z);
          Value parity=remUInt(bk,qk,ci(2));
          Value isOdd=cmpInt(bk,arith::CmpIPredicate::ne,parity,z);
          bk.create<scf::IfOp>(loc,isCore1,[&](OpBuilder &ib,Location){
            Value effectiveK=addInt(ib,k0,abShift);
            Value activeA=chooseAB(ib,effectiveK,at0,at1,tileTy);
            Value activeB=selectB(ib,k0,effectiveK);
            auto emitMatmul=[&](OpBuilder &mb,Value dest) {
              emitPackedMatmul(mb,m0,k0,mE,kE,n0,activeA,activeB,dest);
            };
            ib.create<scf::IfOp>(loc,isFirst,[&](OpBuilder &fb,Location){

              emitMatmul(fb,acc);
              fb.create<scf::YieldOp>(loc);
            },[&](OpBuilder &nf,Location){
              nf.create<scf::IfOp>(loc,isOdd,[&](OpBuilder &ob,Location){
                emitMatmul(ob,partial0);
                ob.create<scf::YieldOp>(loc);
              },[&](OpBuilder &eb,Location){
                emitMatmul(eb,partial1);
                eb.create<scf::YieldOp>(loc);
              });
              nf.create<scf::YieldOp>(loc);
            });
            ib.create<scf::YieldOp>(loc);
          });

          if (sequential) sync(bk);
          bk.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){
            Value nextK=addInt(ib,k0,tk);
            Value hasNext=cmpInt(ib,arith::CmpIPredicate::ult,nextK,kEnd);
            Value hasPrevious=cmpInt(ib,arith::CmpIPredicate::ugt,qk,ci(1));
            if (directIm2colDMA) {

              Value nextM=nextTileM;
              Value lastPanel=cmpInt(ib,
                  arith::CmpIPredicate::eq,qk,ci(kt-1));
              Value nextExists=cmpInt(ib,
                  arith::CmpIPredicate::ult,nextM,mEnd);
              Value nextSpatial=andInt(ib,lastPanel,nextExists);
              auto pendingPreparation = [&](OpBuilder &pb, bool issue) {
                pb.create<scf::IfOp>(loc,hasNext,[&](OpBuilder &pr,Location){
                  if (issue)
                    prepareOperandsImpl(pr,nextK,m0,mE,n0,abShift,true);
                  else
                    waitDirectOperands(pr,nextK,m0);
                  pr.create<scf::YieldOp>(loc);
                });
                if (crossTilePrefetch) {
                  pb.create<scf::IfOp>(loc,nextSpatial,[&](OpBuilder &pr,Location){
                    if (issue) {
                      Value nextME=tileExtent(pr,nextM,totalM,mTileStep);
                      prepareOperandsImpl(pr,z,nextM,nextME,nextTileN,nextAbShift,true);
                    } else {
                      waitDirectOperands(pr,z,nextM);
                    }
                    pr.create<scf::YieldOp>(loc);
                  });
                }
              };
              pendingPreparation(ib,true);

              if (sequential) pendingPreparation(ib,false);
              ib.create<scf::IfOp>(loc,hasPrevious,[&](OpBuilder &ac,Location){
                Value prev=subInt(ac,qk,one);
                accumulatePanel(ac,prev);
                ac.create<scf::YieldOp>(loc);
              });
              if (!sequential) pendingPreparation(ib,false);
            } else {
              ib.create<scf::IfOp>(loc,hasNext,[&](OpBuilder &pr,Location){
                prepareOperands(pr,nextK,m0,mE,n0,abShift);
                pr.create<scf::YieldOp>(loc);
              });
              ib.create<scf::IfOp>(loc,hasPrevious,[&](OpBuilder &ac,Location){
                Value prev=subInt(ac,qk,one);
                accumulatePanel(ac,prev);
                ac.create<scf::YieldOp>(loc);
              });

              if (crossTilePrefetch) {
                Value lastPanel=cmpInt(ib,
                    arith::CmpIPredicate::eq,qk,ci(kt-1));
                Value nextM=nextTileM;
                Value nextExists=cmpInt(ib,
                    arith::CmpIPredicate::ult,nextM,mEnd);
                Value shouldPrefetch=andInt(ib,
                    lastPanel,nextExists);
                ib.create<scf::IfOp>(loc,shouldPrefetch,[&](OpBuilder &pr,Location){
                  Value nextME=tileExtent(pr,nextM,totalM,mTileStep);
                  prepareOperands(pr,z,nextM,nextME,nextTileN,nextAbShift);
                  pr.create<scf::YieldOp>(loc);
                });
              }
            }
            ib.create<scf::YieldOp>(loc);
          });
          bk.create<scf::IfOp>(loc,isCore1,[&](OpBuilder &p,Location){

            p.create<scf::YieldOp>(loc);
          });
          sync(bk);
          bk.create<scf::IfOp>(loc,isCore1,[&](OpBuilder &p,Location){

            p.create<scf::YieldOp>(loc);
          });
        });

        if (overlapNextMatmul) {
          bn.create<scf::IfOp>(loc,isCore1,[&](OpBuilder &ib,Location){
            Value nextM=nextTileM;
            Value nextExists=cmpInt(ib,
                arith::CmpIPredicate::ult,nextM,mEnd);
            ib.create<scf::IfOp>(loc,nextExists,[&](OpBuilder &nb,Location){
              Value nextME=tileExtent(nb,nextM,totalM,mTileStep);
              Value firstKE=minIndex(nb,kEnd,tk);
              Value nextA=chooseAB(nb,nextAbShift,at0,at1,tileTy);
              Value nextB=selectB(nb,z,nextAbShift);
              emitPackedMatmul(nb,nextM,z,nextME,firstKE,nextTileN,nextA,nextB,nextAccumulator);
              nb.create<scf::YieldOp>(loc);
            });
            ib.create<scf::YieldOp>(loc);
          });
        }

        auto prepareNextSecondPanel = [&](bool issue) {
          if (!overlapNextMatmul)
            return;
          bn.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){
            Value nextM=nextTileM;
            Value nextExists=cmpInt(ib,
                arith::CmpIPredicate::ult,nextM,mEnd);
            ib.create<scf::IfOp>(loc,nextExists,[&](OpBuilder &nb,Location){
              if (issue) {
                Value nextME=tileExtent(nb,nextM,totalM,mTileStep);
                prepareOperandsImpl(nb,tk,nextM,nextME,nextTileN,nextAbShift,true);
              } else {
                waitDirectOperands(nb,tk,nextM);
              }
              nb.create<scf::YieldOp>(loc);
            });
            ib.create<scf::YieldOp>(loc);
          });
        };

        if (!(deferOutputWait && kt%2 == 0)) prepareNextSecondPanel(true);

        bn.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){
          Value lastPanel=ib.create<arith::ConstantIndexOp>(loc,kt-1);
          Value hasAccum=cmpInt(ib,arith::CmpIPredicate::ugt,lastPanel,z);
          ib.create<scf::IfOp>(loc,hasAccum,[&](OpBuilder &ac,Location){
            accumulatePanel(ac,lastPanel);
            ac.create<scf::YieldOp>(loc);
          });
          ib.create<scf::YieldOp>(loc);
        });
        if (!(deferOutputWait && kt%2 == 0)) prepareNextSecondPanel(false);
        bn.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){

          Value src=createSubview2D(ib,loc,acc,z,z,mE,nE,ib.getF32Type(),l1);
          Value base=addInt(ib,
              mulInt(ib,m0,cCout),n0);
          auto map=AffineMapAttr::get(AffineMap::get(2,1,
              getAffineDimExpr(0,ib.getContext())*cout+
              getAffineDimExpr(1,ib.getContext())+getAffineSymbolExpr(0,ib.getContext())));
          auto ty=MemRefType::get({ShapedType::kDynamicSize,ShapedType::kDynamicSize},
                                  ib.getF32Type(),map,yt.getMemorySpace());
          Value dst=ib.create<memref::ReinterpretCastOp>(loc,ty,yFlat,base,
              SmallVector<OpFoldResult,2>{mE,nE},
              SmallVector<OpFoldResult,2>{cCout,one}).getResult();
          Value count=mulInt(ib,mE,nE);
          ib.create<memref::DmaStartOp>(loc,src,ValueRange{z,z},dst,ValueRange{z,z},
                                      count,dmaTagOut,ValueRange{z});
          if (deferOutputWait) {
            Value nextM=nextTileM;
            Value last=cmpInt(ib,arith::CmpIPredicate::uge,nextM,mEnd);
            ib.create<scf::IfOp>(loc,last,[&](OpBuilder &wb,Location){
              wb.create<memref::DmaWaitOp>(loc,dmaTagOut,ValueRange{z},count);
              wb.create<scf::YieldOp>(loc);
            });
          } else {
            ib.create<memref::DmaWaitOp>(loc,dmaTagOut,ValueRange{z},count);
          }

          ib.create<scf::YieldOp>(loc);
        });
        if (deferOutputWait && kt%2 == 0) {
          prepareNextSecondPanel(true);
          prepareNextSecondPanel(false);
        }
        sync(bn);
      });
    });
    barrier(b);
    b.create<func::CallOp>(loc,endFn,ValueRange{});
    op.erase();
    return success();
  }

  LogicalResult lowerConv2DF32(linalg::Conv2DNchwFchwOp op) {
    if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
      return op.emitError("expected two inputs and one output"), failure();
    const bool sequential=true;
    Value x = op.inputs()[0], filter = op.inputs()[1], y = op.outputs()[0];
    auto xt = x.getType().dyn_cast<MemRefType>();
    auto ft = filter.getType().dyn_cast<MemRefType>();
    auto yt = y.getType().dyn_cast<MemRefType>();
    if (!xt || !ft || !yt || xt.getRank() != 4 || ft.getRank() != 4 ||
        yt.getRank() != 4 || !xt.getElementType().isF32() ||
        !ft.getElementType().isF32() || !yt.getElementType().isF32() ||
        !xt.hasStaticShape() || !ft.hasStaticShape() || !yt.hasStaticShape())
      return op.emitError("requires bufferized, static-shape FP32 NCHW/FCHW memrefs"), failure();
    const int64_t batch=xt.getShape()[0], cin=xt.getShape()[1];
    const int64_t h=xt.getShape()[2], w=xt.getShape()[3];
    const int64_t cout=ft.getShape()[0], kh=ft.getShape()[2], kw=ft.getShape()[3];
    const int64_t oh=yt.getShape()[2], ow=yt.getShape()[3];
    if (batch <= 0 || cin <= 0 || cout <= 0 || kh <= 0 || kw <= 0 ||
        ft.getShape()[1] != cin || yt.getShape()[0] != batch ||
        yt.getShape()[1] != cout)
      return op.emitError("invalid convolution dimensions"), failure();
    if (oh <= 0 || ow <= 0 || oh != h-kh+1 || ow != w-kw+1)
      return op.emitError("supports only unit stride/dilation with no implicit padding"), failure();
    auto strides = op->getAttrOfType<DenseIntElementsAttr>("strides");
    auto dilations = op->getAttrOfType<DenseIntElementsAttr>("dilations");
    auto allOnes = [](DenseIntElementsAttr a) {
      if (!a) return true;
      for (APInt v : a.getValues<APInt>()) if (v.getSExtValue()!=1) return false;
      return true;
    };
    if (!allOnes(strides) || !allOnes(dilations))
      return op.emitError("currently requires stride=dilation=1"), failure();
    auto module = op->getParentOfType<ModuleOp>();
    Value source = filter;
    if (auto cast = source.getDefiningOp<memref::CastOp>()) source = cast.source();
    if (auto alloc = source.getDefiningOp<memref::AllocOp>()) {
      memref::CopyOp definingCopy = nullptr;
      for (Operation *user : source.getUsers()) {
        if (auto copy = dyn_cast<memref::CopyOp>(user)) {
          if (copy.getTarget() == source) {
            if (definingCopy) return op.emitError("ambiguous constant filter copy"), failure();
            definingCopy = copy;
          }
        }
      }
      if (definingCopy) source = definingCopy.getSource();
    }
    if (auto cast = source.getDefiningOp<memref::CastOp>()) source = cast.source();
    auto getGlobal = source.getDefiningOp<memref::GetGlobalOp>();
    if (!getGlobal)
      return op.emitError("constant filters must be visible as memref.get_global after bufferization"), failure();
    auto symbol = getGlobal->getAttrOfType<FlatSymbolRefAttr>("name");
    if (!symbol)
      return op.emitError("memref.get_global is missing its symbol reference"), failure();
    auto global = module.lookupSymbol<memref::GlobalOp>(symbol.getValue());
    if (!global)
      return op.emitError("filter global symbol not found"), failure();
    auto weights = global->getAttrOfType<DenseElementsAttr>("initial_value");
    if (!weights || !weights.getElementType().isF32() ||
        weights.getNumElements() != cout*cin*kh*kw)
      return op.emitError("filter global needs an FP32 dense initializer"), failure();
    constexpr int64_t TM=64, TK=64, TN=64, PITCH=64;

    const int64_t mTileStep = std::max<int64_t>(1, TM / ow) * std::min<int64_t>(ow, TM);
    const int64_t totalM=batch*oh*ow, totalK=cin*kh*kw;
    if (totalM > INT32_MAX || totalK > INT32_MAX || oh*ow > INT32_MAX ||
        batch*oh*ow > INT32_MAX || h > INT32_MAX || w > INT32_MAX)
      return op.emitError("convolution dimensions exceed 32-bit index range"), failure();

    constexpr int64_t RAW_ELEMS = 2048;
    const int64_t maxCols = std::min<int64_t>(ow, TM) + kw - 1;
    const int64_t maxRows = (TM / std::min<int64_t>(ow, TM)) + kh + 1;
    const int64_t nt=(cout+TN-1)/TN, kt=(totalK+TK-1)/TK;

    const bool directIm2colDMA = batch == 1 &&
        (ow <= TM || (kh == 1 && kw == 1));

    if (!directIm2colDMA &&
        (maxCols <= 0 || maxRows <= 0 || maxRows > RAW_ELEMS / maxCols))
      return op.emitError("RAW tile exceeds the 8 KiB staging capacity; smaller M tiling required"), failure();

    const bool fullNTiles = cout % TN == 0;
    const bool crossTilePrefetch = !sequential && batch == 1 &&
        (cout <= TN || (directIm2colDMA && fullNTiles));
    SmallVector<APFloat, 256> packed;
    packed.reserve(nt*kt*TK*PITCH);
    SmallVector<APFloat, 256> original;
    original.reserve(weights.getNumElements());
    for (APFloat v : weights.getValues<APFloat>()) original.push_back(v);
    APFloat zero(0.0f);
    for (int64_t qn=0; qn<nt; ++qn)
      for (int64_t qk=0; qk<kt; ++qk)
        for (int64_t j=0; j<TK; ++j)
          for (int64_t l=0; l<PITCH; ++l) {
            int64_t k=qk*TK+j, n=qn*TN+l;
            if (k<totalK && l<TN && n<cout) {
              int64_t c=k/(kh*kw), r=k%(kh*kw), ky=r/kw, kx=r%kw;
              packed.push_back(original[((n*cin+c)*kh+ky)*kw+kx]);
            } else packed.push_back(zero);
          }
    OpBuilder b(op);
    Location loc=op.getLoc();

    const bool useI32Indices =
        totalM <= INT32_MAX-128 && cout <= INT32_MAX-128 &&
        totalK <= INT32_MAX/128 &&
        batch <= INT32_MAX/cin/h/w &&
        totalM <= INT32_MAX/cout &&
        nt <= INT32_MAX/kt/TK/PITCH;
    auto toI32=[&](OpBuilder &ib,Value v)->Value {
      if (v.getType().isIndex())
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getI32Type(),v);
      return v;
    };
    auto addInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::AddIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::AddIOp>(loc,lhs,rhs);
    };
    auto subInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::SubIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::SubIOp>(loc,lhs,rhs);
    };
    auto mulInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::MulIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::MulIOp>(loc,lhs,rhs);
    };
    auto divUInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::DivUIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::DivUIOp>(loc,lhs,rhs);
    };
    auto remUInt=[&](OpBuilder &ib,Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        Value result=ib.createOrFold<arith::RemUIOp>(loc,toI32(ib,lhs),toI32(ib,rhs));
        return ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),result);
      }
      return ib.createOrFold<arith::RemUIOp>(loc,lhs,rhs);
    };
    auto cmpInt=[&](OpBuilder &ib,arith::CmpIPredicate pred,
                       Value lhs,Value rhs)->Value {
      if (useI32Indices && lhs.getType().isIndex()) {
        lhs=toI32(ib,lhs); rhs=toI32(ib,rhs);
      }
      return ib.createOrFold<arith::CmpIOp>(loc,pred,lhs,rhs);
    };

    auto tileLoop=[&](OpBuilder &ib,Value lower,Value upper,Value step,
                      auto &&body) {
      if (!useI32Indices) {
        ib.create<scf::ForOp>(loc,lower,upper,step,ValueRange{},
            [&](OpBuilder &lb,Location,Value iv,ValueRange){
          body(lb,iv); lb.create<scf::YieldOp>(loc);
        });
        return;
      }
      Value lo32=toI32(ib,lower), hi32=toI32(ib,upper), step32=toI32(ib,step);
      auto wh=ib.create<scf::WhileOp>(loc,TypeRange{ib.getI32Type()},ValueRange{lo32});
      OpBuilder::InsertionGuard guard(ib);
      Block *before=ib.createBlock(&wh.getBefore(),wh.getBefore().end(),
          TypeRange{ib.getI32Type()},SmallVector<Location>{loc});
      ib.setInsertionPointToEnd(before);
      Value iv=before->getArgument(0);
      Value cond=ib.createOrFold<arith::CmpIOp>(loc,arith::CmpIPredicate::slt,iv,hi32);
      ib.create<scf::ConditionOp>(loc,cond,ValueRange{iv});
      Block *after=ib.createBlock(&wh.getAfter(),wh.getAfter().end(),
          TypeRange{ib.getI32Type()},SmallVector<Location>{loc});
      ib.setInsertionPointToEnd(after);
      iv=after->getArgument(0);
      Value indexIV=ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),iv);
      body(ib,indexIV);
      Value next=ib.createOrFold<arith::AddIOp>(loc,iv,step32);
      ib.create<scf::YieldOp>(loc,ValueRange{next});
    };
    const std::string packedName = ("__quad_conv_packed_" +
                                    std::to_string(reinterpret_cast<uintptr_t>(op.getOperation())));
    auto packedTensorType=RankedTensorType::get({(int64_t)packed.size()}, b.getF32Type());
    auto packedAttr=DenseElementsAttr::get(packedTensorType, ArrayRef<APFloat>(packed));
    auto packedMemrefType=MemRefType::get({(int64_t)packed.size()}, b.getF32Type());
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      b.create<memref::GlobalOp>(loc, packedName, b.getStringAttr("private"),
                                  packedMemrefType, packedAttr, false, IntegerAttr());
    }
    Value weightsPacked=b.create<memref::GetGlobalOp>(loc, packedMemrefType, packedName);
    SymbolTable st(module);
    auto getFn=[&](StringRef name, FunctionType type) {
      if (auto fn=st.lookup<func::FuncOp>(name)) return fn;
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      auto fn=b.create<func::FuncOp>(loc, name, type);
      fn.setPrivate(); st.insert(fn); return fn;
    };
    auto coreFn=getFn("snrt_cluster_core_idx", b.getFunctionType({}, {b.getI32Type()}));
    auto barrierFn=getFn("snrt_cluster_hw_barrier", b.getFunctionType({}, {}));

    auto lowerBeginFn=getFn("conv_lowering_begin", b.getFunctionType({}, {}));
    auto lowerEndFn=getFn("conv_lowering_end", b.getFunctionType({}, {}));

    Value core=b.create<func::CallOp>(loc, coreFn, ValueRange{}).getResult(0);
    Value isCore0=cmpInt(b, arith::CmpIPredicate::eq,
                   core, b.create<arith::ConstantIntOp>(loc, 0, 32));
    Value isCore1=cmpInt(b, arith::CmpIPredicate::eq,
                   core, b.create<arith::ConstantIntOp>(loc, 1, 32));
    auto l1=IntegerAttr::get(b.getI64Type(), 1);
    auto tileTy=MemRefType::get({TK,PITCH}, b.getF32Type(),
                                MemRefLayoutAttrInterface{}, l1);
    auto outTileTy=MemRefType::get({TM,PITCH}, b.getF32Type(),
                                   MemRefLayoutAttrInterface{}, l1);

    Value at0=b.create<memref::AllocOp>(loc,tileTy);
    Value at1=b.create<memref::AllocOp>(loc,tileTy);

    constexpr int64_t B_CACHE_ELEMS = 2*TK*PITCH + 4096/sizeof(float);
    const bool residentWeights = nt <= B_CACHE_ELEMS/PITCH/totalK;
    Value weightsL1, gt0, gt1;
    if (residentWeights) {
      auto cacheTy=MemRefType::get({nt*totalK,PITCH},b.getF32Type(),
                                  MemRefLayoutAttrInterface{},l1);
      weightsL1=b.create<memref::AllocOp>(loc,cacheTy);
    } else {
      gt0=b.create<memref::AllocOp>(loc,tileTy);
      gt1=b.create<memref::AllocOp>(loc,tileTy);
    }

    Value c0Buf=b.create<memref::AllocOp>(loc,outTileTy);
    Value c1Buf=b.create<memref::AllocOp>(loc,outTileTy);
    Value c2Buf=b.create<memref::AllocOp>(loc,outTileTy);

    auto chooseC=[&](OpBuilder &ib,Value slot)->Value {
      Value isZero=cmpInt(ib,arith::CmpIPredicate::eq,
          slot,ib.create<arith::ConstantIndexOp>(loc,0));
      auto outer=ib.create<scf::IfOp>(loc,TypeRange{outTileTy},isZero,
          [&](OpBuilder &yes,Location){
            yes.create<scf::YieldOp>(loc,ValueRange{c0Buf});
          },[&](OpBuilder &no,Location){
            Value isOne=cmpInt(no,arith::CmpIPredicate::eq,
                slot,no.create<arith::ConstantIndexOp>(loc,1));
            auto inner=no.create<scf::IfOp>(loc,TypeRange{outTileTy},isOne,
                [&](OpBuilder &yes1,Location){
                  yes1.create<scf::YieldOp>(loc,ValueRange{c1Buf});
                },[&](OpBuilder &no1,Location){
                  no1.create<scf::YieldOp>(loc,ValueRange{c2Buf});
                });
            no.create<scf::YieldOp>(loc,ValueRange{inner.getResult(0)});
          });
      return outer.getResult(0);
    };
    auto chooseAB=[&](OpBuilder &ib,Value k0,Value first,Value second,
                      MemRefType ty)->Value {
      Value panel=divUInt(ib,k0,
          ib.create<arith::ConstantIndexOp>(loc,TK));
      Value parity=remUInt(ib,panel,
          ib.create<arith::ConstantIndexOp>(loc,2));
      Value even=cmpInt(ib,arith::CmpIPredicate::eq,
          parity,ib.create<arith::ConstantIndexOp>(loc,0));
      auto branch=ib.create<scf::IfOp>(loc,TypeRange{ty},even,
          [&](OpBuilder &yes,Location){
            yes.create<scf::YieldOp>(loc,ValueRange{first});
          },[&](OpBuilder &no,Location){
            no.create<scf::YieldOp>(loc,ValueRange{second});
          });
      return branch.getResult(0);
    };
    Value yFlat=b.create<memref::CollapseShapeOp>(
        loc,MemRefType::get({batch*cout*oh*ow},b.getF32Type()),y,
        SmallVector<ReassociationIndices>{{0,1,2,3}});

    Value raw;
    if (!directIm2colDMA)
      raw=b.create<memref::AllocOp>(loc,
          MemRefType::get({RAW_ELEMS},b.getF32Type(),MemRefLayoutAttrInterface{},l1));
    Value dmaTag=b.create<memref::AllocaOp>(loc, MemRefType::get({1},b.getI32Type()));
    Value dmaTagWeights=b.create<memref::AllocaOp>(loc, MemRefType::get({1},b.getI32Type()));

    Value dmaTagRawGroup=b.create<memref::AllocaOp>(loc,
        MemRefType::get({9},b.getI32Type()));

    Value dmaTagOutput=b.create<memref::AllocaOp>(loc, MemRefType::get({1},b.getI32Type()));

    Value dmaTagDirectA=b.create<memref::AllocaOp>(loc,
        MemRefType::get({1},b.getI32Type()));

    Value dmaDirectCount=b.create<memref::AllocaOp>(loc,
        MemRefType::get({1},b.getI32Type()));

    b.create<func::CallOp>(loc,lowerBeginFn,ValueRange{});
    auto ci=[&](int64_t v)->Value {return b.create<arith::ConstantIndexOp>(loc,v);};
    Value z=ci(0), one=ci(1), mEnd=ci(totalM), nEnd=ci(cout), kEnd=ci(totalK);
    Value tm=ci(TM), mStep=ci(mTileStep), tn=ci(TN), tk=ci(TK), pitch=ci(PITCH);
    Value cOH=ci(oh), cOW=ci(ow), cOHOW=ci(oh*ow);
    Value cKH=ci(kh), cKW=ci(kw), cKHKW=ci(kh*kw), cCout=ci(cout);
    Value cKT=ci(kt), cTileLen=ci(TK*PITCH);
    auto minIndex=[&](OpBuilder &ib, Value a, Value v)->Value {
      Value cmp=cmpInt(ib,arith::CmpIPredicate::slt,a,v);
      return ib.createOrFold<arith::SelectOp>(loc,cmp,a,v);
    };

    auto tileExtent=[&](OpBuilder &qb,Value start,int64_t extent,int64_t step)->Value {
      auto constant=[&](int64_t v)->Value {
        return qb.create<arith::ConstantIndexOp>(loc,v);
      };
      if (extent<=step) return constant(extent);
      const int64_t tail=extent%step;
      if (tail==0) return constant(step);
      Value last=cmpInt(qb,arith::CmpIPredicate::eq,start,constant(extent-tail));
      return qb.createOrFold<arith::SelectOp>(loc,last,constant(tail),constant(step));
    };
    auto weightRows=[&](OpBuilder &ib,Value n0,Value k0,Value rows)->Value {
      Value qn=divUInt(ib,n0,tn);
      Value row=addInt(ib,
          mulInt(ib,qn,kEnd),k0);
      return createSubview2D(ib,loc,weightsL1,row,z,rows,pitch,
                             ib.getF32Type(),l1);
    };
    auto selectB=[&](OpBuilder &ib,Value n0,Value k0,Value effectiveK)->Value {
      if (residentWeights) {
        Value rows=totalK%TK==0 ? tk : minIndex(ib,subInt(ib,kEnd,k0),tk);
        return weightRows(ib,n0,k0,rows);
      }
      return chooseAB(ib,effectiveK,gt0,gt1,tileTy);
    };
    auto loop=[&](OpBuilder &ib, Value end, Value step, auto &&body) {
      tileLoop(ib,z,end,step,[&](OpBuilder &lb,Value iv){body(lb,iv);});
    };
    auto tileOf=[&](OpBuilder &ib,Value spatial)->Value {
      return divUInt(ib,spatial,mStep);
    };
    auto sync=[&](OpBuilder &ib){
      ib.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &db,Location){
        emitConvSequentialDrain(db,loc);
        db.create<scf::YieldOp>(loc);
      });
      ib.create<func::CallOp>(loc,barrierFn,ValueRange{});
    };

    auto staticTiles=[&](OpBuilder &q,int64_t extent,int64_t step,auto &&body) {
      auto constant=[&](OpBuilder &qb,int64_t v)->Value {
        return qb.create<arith::ConstantIndexOp>(loc,v);
      };
      const int64_t fullEnd=extent/step*step;
      if (fullEnd == step) {
        body(q,constant(q,0),constant(q,step));
      } else if (fullEnd != 0) {
        tileLoop(q,constant(q,0),constant(q,fullEnd),constant(q,step),
            [&](OpBuilder &qb,Value iv) { body(qb,iv,constant(qb,step)); });
      }
      if (fullEnd != extent)
        body(q,constant(q,fullEnd),constant(q,extent-fullEnd));
    };
    staticTiles(b,totalM,mTileStep,[&](OpBuilder &bm,Value m0,Value mE){
      staticTiles(bm,cout,TN,[&](OpBuilder &bn,Value n0,Value nE){

        Value tileM=divUInt(bn,m0,mStep);
        Value tileN=divUInt(bn,n0,tn);
        Value tileSerial=addInt(bn,
            mulInt(bn,tileM,
                ci((cout+TN-1)/TN)),tileN);
        Value slot=remUInt(bn,tileSerial,ci(3));

        Value abShift = z;
        Value nextAbShift = z;
        if (kt % 2 != 0) {
          abShift=mulInt(bn,
              remUInt(bn,tileSerial,ci(2)),tk);
          nextAbShift=subInt(bn,tk,abShift);
        }

        Value candidateN=addInt(bn,n0,tn);
        Value sameM=cmpInt(bn,arith::CmpIPredicate::ult,candidateN,nEnd);
        Value nextTileN=bn.createOrFold<arith::SelectOp>(loc,sameM,candidateN,z);
        Value followingM=addInt(bn,m0,mStep);
        Value nextTileM=bn.createOrFold<arith::SelectOp>(loc,sameM,m0,followingM);
        Value nextTileExists=cmpInt(bn,arith::CmpIPredicate::ult,nextTileM,mEnd);
        const bool staticDirectPanels = directIm2colDMA && kt <= 32 &&
            kh*kw <= 1024/kt && useI32Indices;
        const bool overlapNextMatmul = directIm2colDMA && crossTilePrefetch &&
            kt >= 3 && (kt % 2 != 0);

        const bool deferOutputWait = overlapNextMatmul;
        Value acc=chooseC(bn,slot);
        Value partial0=chooseC(bn,remUInt(bn,
            addInt(bn,slot,ci(1)),ci(3)));
        Value partial1=chooseC(bn,remUInt(bn,
            addInt(bn,slot,ci(2)),ci(3)));

        auto firstWeightUse=[&](OpBuilder &ib,Value k0,Value targetM)->Value {
          Value firstM=cmpInt(ib,arith::CmpIPredicate::eq,targetM,z);
          Value firstK=cmpInt(ib,arith::CmpIPredicate::eq,k0,z);
          return ib.create<arith::AndIOp>(loc,firstM,firstK);
        };
        auto waitWeights=[&](OpBuilder &ib,Value k0,Value targetM,Value count) {
          if (residentWeights) {
            ib.create<scf::IfOp>(loc,firstWeightUse(ib,k0,targetM),
                [&](OpBuilder &wb,Location){
              Value allCount=mulInt(wb,kEnd,nE);
              wb.create<memref::DmaWaitOp>(loc,dmaTagWeights,ValueRange{z},allCount);
              wb.create<scf::YieldOp>(loc);
            });
          } else {
            ib.create<memref::DmaWaitOp>(loc,dmaTagWeights,ValueRange{z},count);
          }
        };
        auto waitDirectOperands = [&](OpBuilder &ib, Value k0, Value targetM) {

          Value lastCount32=ib.create<memref::LoadOp>(loc,dmaDirectCount,ValueRange{z});
          Value lastCount=ib.createOrFold<arith::IndexCastOp>(loc,ib.getIndexType(),lastCount32);
          ib.create<memref::DmaWaitOp>(loc,dmaTagDirectA,ValueRange{z},lastCount);

        };

        auto prepareOperandsImpl = [&](OpBuilder &ib, Value k0, Value targetM, Value targetN,
                                       Value targetME, Value targetAbShift,
                                       bool deferWait) {

          if (auto constant=k0.getDefiningOp<arith::ConstantIndexOp>())
            if (constant.value() >= totalK) return;
          if (auto constant=targetME.getDefiningOp<arith::ConstantIndexOp>())
            if (constant.value() <= 0) return;
          Value targetTile=tileOf(ib,targetM);
          Value targetPanel=divUInt(ib,k0,tk);
          Value kE=totalK%TK==0 ? tk : minIndex(ib,subInt(ib,kEnd,k0),tk);
          Value effectiveK=addInt(ib,k0,targetAbShift);
          Value activeA=chooseAB(ib,effectiveK,at0,at1,tileTy);
          Value activeB=selectB(ib,targetN,k0,effectiveK);
          Value c0=ib.create<arith::ConstantIntOp>(loc,0,32);
          Value c1=ib.create<arith::ConstantIntOp>(loc,1,32);
          Value c16=ib.create<arith::ConstantIntOp>(loc,16,32);
          Value ow32=ib.create<arith::ConstantIntOp>(loc,ow,32);
          Value hw32=ib.create<arith::ConstantIntOp>(loc,oh*ow,32);
          Value kh32=ib.create<arith::ConstantIntOp>(loc,kh,32);
          Value kw32=ib.create<arith::ConstantIntOp>(loc,kw,32);
          Value m032=ib.createOrFold<arith::IndexCastOp>(loc,ib.getI32Type(),targetM);
          Value mE32=ib.createOrFold<arith::IndexCastOp>(loc,ib.getI32Type(),targetME);
          Value k032=ib.createOrFold<arith::IndexCastOp>(loc,ib.getI32Type(),k0);
          Value kE32=ib.createOrFold<arith::IndexCastOp>(loc,ib.getI32Type(),kE);
          auto add=[&](OpBuilder &ob,Value a,Value v)->Value {return addInt(ob,a,v);};
          auto sub=[&](OpBuilder &ob,Value a,Value v)->Value {return subInt(ob,a,v);};
          auto mul=[&](OpBuilder &ob,Value a,Value v)->Value {return mulInt(ob,a,v);};
          auto umin=[&](OpBuilder &ob,Value a,Value v)->Value {
            Value cmp=cmpInt(ob,arith::CmpIPredicate::ult,a,v);
            return ob.createOrFold<arith::SelectOp>(loc,cmp,a,v);
          };
          auto umax=[&](OpBuilder &ob,Value a,Value v)->Value {
            Value cmp=cmpInt(ob,arith::CmpIPredicate::ult,a,v);
            return ob.createOrFold<arith::SelectOp>(loc,cmp,v,a);
          };
          auto ix=[&](OpBuilder &ob,Value v)->Value {
            return ob.createOrFold<arith::IndexCastOp>(loc,ob.getIndexType(),v);
          };

          auto vectorTy=VectorType::get({16},ib.getF32Type(),{true});

          Value weightCount=mulInt(ib,kE,nE);
          auto issueWeights=[&](OpBuilder &wb,Value rows,Value dest,Value count) {
            Value qn=divUInt(wb,targetN,tn);
            Value qk=divUInt(wb,k0,tk);
            Value tileId=addInt(wb,mulInt(wb,qn,cKT),qk);
            Value base=mulInt(wb,tileId,cTileLen);

            auto weightLayout=AffineMapAttr::get(AffineMap::get(2,1,
                getAffineDimExpr(0,wb.getContext())*PITCH+
                getAffineDimExpr(1,wb.getContext())+
                getAffineSymbolExpr(0,wb.getContext())));
            auto weightViewTy=MemRefType::get(
                {ShapedType::kDynamicSize,ShapedType::kDynamicSize},
                wb.getF32Type(),weightLayout,packedMemrefType.getMemorySpace());
            Value srcW=wb.create<memref::ReinterpretCastOp>(loc,weightViewTy,
                weightsPacked,base,SmallVector<OpFoldResult,2>{rows,nE},
                SmallVector<OpFoldResult,2>{pitch,one}).getResult();
            Value dstW=createSubview2D(wb,loc,dest,z,z,rows,nE,
                                      wb.getF32Type(),l1);
            emitConvSequentialDrain(wb,loc);
            wb.create<memref::DmaStartOp>(loc,srcW,ValueRange{z,z},dstW,
                ValueRange{z,z},count,dmaTagWeights,ValueRange{z});
          };
          if (residentWeights) {

            ib.create<scf::IfOp>(loc,firstWeightUse(ib,k0,targetM),
                [&](OpBuilder &wb,Location){
              Value dest=weightRows(wb,targetN,z,kEnd);
              Value count=mulInt(wb,kEnd,nE);
              issueWeights(wb,kEnd,dest,count);
              wb.create<scf::YieldOp>(loc);
            });
          } else {
            issueWeights(ib,kE,activeB,weightCount);
          }
          if (batch==1 && kh==1 && kw==1) {

            auto flatType=MemRefType::get({cin*h*w},ib.getF32Type(),
                                         MemRefLayoutAttrInterface{},xt.getMemorySpace());
            Value flat=ib.create<memref::CollapseShapeOp>(loc,flatType,x,
                SmallVector<ReassociationIndices>{{0,1,2,3}});
            auto layout=AffineMapAttr::get(AffineMap::get(2,1,
                getAffineDimExpr(0,ib.getContext())*(h*w)+
                getAffineDimExpr(1,ib.getContext())+getAffineSymbolExpr(0,ib.getContext())));
            auto sourceType=MemRefType::get(
                {ShapedType::kDynamicSize,ShapedType::kDynamicSize},
                ib.getF32Type(),layout,xt.getMemorySpace());
            Value offset=addInt(ib,mulInt(ib,k0,ci(h*w)),targetM);
            Value src=ib.create<memref::ReinterpretCastOp>(loc,sourceType,flat,offset,
                SmallVector<OpFoldResult,2>{kE,targetME},
                SmallVector<OpFoldResult,2>{ci(h*w),one}).getResult();
            Value dst=createSubview2D(ib,loc,activeA,z,z,kE,targetME,ib.getF32Type(),l1);
            Value count=mulInt(ib,kE,targetME);
            emitConvSequentialDrain(ib,loc);
            ib.create<memref::DmaStartOp>(loc,src,ValueRange{z,z},dst,
                ValueRange{z,z},count,dmaTagDirectA,ValueRange{z});
            Value count32=ib.createOrFold<arith::IndexCastOp>(loc,ib.getI32Type(),count);
            ib.create<memref::StoreOp>(loc,count32,dmaDirectCount,ValueRange{z});
            if (!deferWait) waitDirectOperands(ib,k0,targetM);
            return;
          }

          if (directIm2colDMA) {

            auto c32=[&](OpBuilder &q,int64_t v)->Value {
              return q.create<arith::ConstantIntOp>(loc,v,32);
            };
            auto idx=[&](OpBuilder &q,Value v)->Value {
              return q.createOrFold<arith::IndexCastOp>(loc,q.getIndexType(),v);
            };
            auto as32=[&](Value v)->Value {
              return ib.createOrFold<arith::IndexCastOp>(loc,ib.getI32Type(),v);
            };
            auto loop32=[&](OpBuilder &q,Value upper,auto &&body) {
              auto wh=q.create<scf::WhileOp>(loc,TypeRange{q.getI32Type()},
                                             ValueRange{c32(q,0)});
              OpBuilder::InsertionGuard guard(q);
              Block *before=q.createBlock(&wh.getBefore(),wh.getBefore().end(),
                  TypeRange{q.getI32Type()},SmallVector<Location>{loc});
              q.setInsertionPointToEnd(before);
              Value iv=before->getArgument(0);
              Value pred=cmpInt(q,arith::CmpIPredicate::ult,iv,upper);
              q.create<scf::ConditionOp>(loc,pred,ValueRange{iv});
              Block *after=q.createBlock(&wh.getAfter(),wh.getAfter().end(),
                  TypeRange{q.getI32Type()},SmallVector<Location>{loc});
              q.setInsertionPointToEnd(after);
              iv=after->getArgument(0);body(q,iv);
              Value next=addInt(q,iv,c32(q,1));
              q.create<scf::YieldOp>(loc,ValueRange{next});
            };
            auto flatATy=MemRefType::get({TK*PITCH},ib.getF32Type(),
                                        MemRefLayoutAttrInterface{},l1);
            Value flatA=ib.create<memref::CollapseShapeOp>(loc,flatATy,
                activeA,SmallVector<ReassociationIndices>{{0,1}});
            auto xFlatTy=MemRefType::get({batch*cin*h*w},ib.getF32Type(),
                MemRefLayoutAttrInterface{},xt.getMemorySpace());
            Value xFlat=ib.create<memref::CollapseShapeOp>(loc,xFlatTy,x,
                SmallVector<ReassociationIndices>{{0,1,2,3}});
            Value outRow32=divUInt(ib,as32(targetM),c32(ib,ow));
            auto emit=[&](OpBuilder &q,Value srcOff,Value dstOff,Value reps,
                          int64_t srcPitch,int64_t dstPitch) {
              auto layout=[&](int64_t pitchValue) {
                return AffineMapAttr::get(AffineMap::get(2,1,
                    getAffineDimExpr(0,q.getContext())*pitchValue+
                    getAffineDimExpr(1,q.getContext())+getAffineSymbolExpr(0,q.getContext())));
              };
              auto st=MemRefType::get({ShapedType::kDynamicSize,ow},
                                      q.getF32Type(),layout(srcPitch),xt.getMemorySpace());
              auto dt=MemRefType::get({ShapedType::kDynamicSize,ow},
                                      q.getF32Type(),layout(dstPitch),l1);
              Value n=idx(q,reps);
              Value src=q.create<memref::ReinterpretCastOp>(loc,st,xFlat,idx(q,srcOff),
                  SmallVector<OpFoldResult,2>{n,ci(ow)},
                  SmallVector<OpFoldResult,2>{ci(srcPitch),one}).getResult();
              Value dst=q.create<memref::ReinterpretCastOp>(loc,dt,flatA,idx(q,dstOff),
                  SmallVector<OpFoldResult,2>{n,ci(ow)},
                  SmallVector<OpFoldResult,2>{ci(dstPitch),one}).getResult();
              Value count=mulInt(q,n,ci(ow));
              emitConvSequentialDrain(q,loc);
              q.create<memref::DmaStartOp>(loc,src,ValueRange{z,z},dst,
                  ValueRange{z,z},count,dmaTagDirectA,ValueRange{z});
            };
            if (staticDirectPanels) {

              auto *ctx=ib.getContext();
              const auto &layoutAnalysis=getAnalysis<DataLayoutAnalysis>();
              LowerToLLVMOptions addressOptions(ctx,
                  layoutAnalysis.getAtOrAbove(op.getOperation()));
              LLVMTypeConverter addressConverter(ctx,addressOptions,&layoutAnalysis);
              auto byteAddress=[&](Value memref)->Value {

                Type descTy=addressConverter.convertType(memref.getType());
                auto fields=descTy.cast<LLVM::LLVMStructType>().getBody();
                Value desc=ib.create<UnrealizedConversionCastOp>(loc,
                    TypeRange{descTy},ValueRange{memref}).getResult(0);
                Value aligned=ib.create<LLVM::ExtractValueOp>(loc,fields[1],desc,
                    ib.getI64ArrayAttr({1}));
                Value offset=ib.create<LLVM::ExtractValueOp>(loc,fields[2],desc,
                    ib.getI64ArrayAttr({2}));
                Value off32=offset;
                unsigned bits=offset.getType().cast<IntegerType>().getWidth();
                if (bits > 32)
                  off32=ib.create<LLVM::TruncOp>(loc,ib.getI32Type(),offset);
                else if (bits < 32)
                  off32=ib.create<LLVM::ZExtOp>(loc,ib.getI32Type(),offset);
                Value base=ib.create<LLVM::PtrToIntOp>(loc,ib.getI32Type(),aligned);
                return addInt(ib,base,mulInt(ib,off32,c32(ib,4)));
              };
              Value sourceBytes=byteAddress(xFlat);
              Value targetBytes=byteAddress(flatA);
              auto emitByteGroup=[&](OpBuilder &pb,Value srcElements,
                                     int64_t dstElements,int64_t commands,
                                     int64_t repetitions,int64_t srcPitch,
                                     int64_t dstPitch,int64_t srcStep,
                                     int64_t dstStep) {
                Value src=addInt(pb,sourceBytes,mulInt(pb,srcElements,c32(pb,4)));
                Value dst=addInt(pb,targetBytes,c32(pb,dstElements*4));
                if (commands == 1) {

                  SmallVector<Value,7> args={src,dst,c32(pb,repetitions),
                      c32(pb,srcPitch*4),c32(pb,dstPitch*4),
                      c32(pb,ow*4),c32(pb,2)};
                  OperationState single(loc,LLVM::InlineAsmOp::getOperationName());
                  single.addTypes(pb.getI32Type());
                  single.addOperands(args);
                  single.addAttribute("asm_string",pb.getStringAttr(
                      "dmstr $4, $5\n\t"
                      "dmrep $3\n\t"
                      "dmsrc $1, zero\n\t"
                      "dmdst $2, zero\n\t"
                      "dmcpy $0, $6, $7"));
                  single.addAttribute("constraints",pb.getStringAttr(
                      "=&r,r,r,r,r,r,r,r,~{memory}"));
                  single.addAttribute("has_side_effects",pb.getUnitAttr());
                  Value lastId=pb.create(single)->getResult(0);
                  pb.create<memref::StoreOp>(loc,lastId,dmaTagDirectA,ValueRange{z});
                  return;
                }
                SmallVector<Value,10> operands={src,dst,c32(pb,commands),
                    c32(pb,repetitions),c32(pb,srcPitch*4),c32(pb,dstPitch*4),
                    c32(pb,srcStep*4),c32(pb,dstStep*4),c32(pb,ow*4),c32(pb,2)};

                OperationState state(loc,LLVM::InlineAsmOp::getOperationName());
                state.addTypes(pb.getI32Type());
                state.addOperands(operands);
                state.addAttribute("asm_string",pb.getStringAttr(
                    "mv t0, $1\n\t"
                    "mv t1, $2\n\t"
                    "mv t2, $3\n\t"
                    "dmstr $5, $6\n\t"
                    "dmrep $4\n\t"
                    "1:\n\t"
                    "dmsrc t0, zero\n\t"
                    "dmdst t1, zero\n\t"
                    "dmcpy $0, $9, $10\n\t"
                    "add t0, t0, $7\n\t"
                    "add t1, t1, $8\n\t"
                    "addi t2, t2, -1\n\t"
                    "bnez t2, 1b"));
                state.addAttribute("constraints",pb.getStringAttr(
                    "=&r,r,r,r,r,r,r,r,r,r,r,~{t0},~{t1},~{t2},~{memory}"));
                state.addAttribute("has_side_effects",pb.getUnitAttr());
                Value lastId=pb.create(state)->getResult(0);

                pb.create<memref::StoreOp>(loc,lastId,dmaTagDirectA,ValueRange{z});
              };

              auto emitStaticPanel=[&](OpBuilder &pb,int64_t panelK,
                                         int64_t spatialRows) {
                Value rowBase=mulInt(pb,outRow32,c32(pb,w));
                auto groups=buildConvDmaPlan(panelK,totalK,kh,kw,h,w,ow,spatialRows);
                for (const auto &g:groups) {
                  Value src=addInt(pb,rowBase,c32(pb,g.src));
                  emitByteGroup(pb,src,g.dst,g.commands,g.repetitions,
                                g.srcPitch,g.dstPitch,g.srcStep,g.dstStep);
                }
                pb.create<memref::StoreOp>(loc,
                    c32(pb,groups.back().repetitions*ow),dmaDirectCount,ValueRange{z});
              };

              auto dispatchPanel=[&](auto &&self,OpBuilder &pb,int64_t panel,
                                     int64_t spatialRows)->void {
                if (auto constant=k0.getDefiningOp<arith::ConstantIndexOp>()) {
                  emitStaticPanel(pb,constant.value(),spatialRows);
                  return;
                }
                if (panel == kt-1) {
                  emitStaticPanel(pb,panel*TK,spatialRows);
                  return;
                }
                Value match=cmpInt(pb,arith::CmpIPredicate::eq,k0,ci(panel*TK));
                pb.create<scf::IfOp>(loc,match,[&](OpBuilder &yes,Location){
                  emitStaticPanel(yes,panel*TK,spatialRows);
                  yes.create<scf::YieldOp>(loc);
                },[&](OpBuilder &no,Location){
                  self(self,no,panel+1,spatialRows);
                  no.create<scf::YieldOp>(loc);
                });
              };
              const int64_t tail=totalM%mTileStep;
              if (auto constant=targetME.getDefiningOp<arith::ConstantIndexOp>()) {
                dispatchPanel(dispatchPanel,ib,0,constant.value()/ow);
              } else if (tail == 0) {
                dispatchPanel(dispatchPanel,ib,0,mTileStep/ow);
              } else {

                Value full=cmpInt(ib,arith::CmpIPredicate::eq,targetME,mStep);
                ib.create<scf::IfOp>(loc,full,[&](OpBuilder &yes,Location){
                  dispatchPanel(dispatchPanel,yes,0,mTileStep/ow);
                  yes.create<scf::YieldOp>(loc);
                },[&](OpBuilder &no,Location){
                  dispatchPanel(dispatchPanel,no,0,tail/ow);
                  no.create<scf::YieldOp>(loc);
                });
              }
              if (!deferWait)
                waitDirectOperands(ib,k0,targetM);
              return;
            }
            Value k032=as32(k0),ke32=as32(kE);
            Value end32=addInt(ib,k032,ke32);
            Value rows32=divUInt(ib,as32(targetME),c32(ib,ow));

            for (int64_t ky=0;ky<kh;++ky) {
              for (int64_t kx=0;kx<kw;++kx) {
                int64_t t=ky*kw+kx;
                auto sub=[&](Value a,Value v)->Value {
                  return subInt(ib,a,v);
                };

                Value before=cmpInt(ib,arith::CmpIPredicate::ugt,k032,c32(ib,t));
                Value delta=ib.createOrFold<arith::SelectOp>(loc,before,sub(k032,c32(ib,t)),c32(ib,0));
                Value rounded=addInt(ib,delta,c32(ib,kh*kw-1));
                Value first=divUInt(ib,rounded,c32(ib,kh*kw));
                Value past=cmpInt(ib,arith::CmpIPredicate::ugt,end32,c32(ib,t));
                Value endDelta=ib.createOrFold<arith::SelectOp>(loc,past,sub(end32,c32(ib,t)),c32(ib,0));
                Value endRounded=addInt(ib,endDelta,c32(ib,kh*kw-1));
                Value last=divUInt(ib,endRounded,c32(ib,kh*kw));
                Value num=subInt(ib,last,first);
                Value any=cmpInt(ib,arith::CmpIPredicate::ugt,num,c32(ib,0));
                ib.create<scf::IfOp>(loc,any,[&](OpBuilder &q,Location) {
                  Value channelBase=mulInt(q,first,c32(q,h*w));
                  Value rowBase=mulInt(q,outRow32,c32(q,w));
                  Value srcBase=addInt(q,channelBase,rowBase);
                  srcBase=addInt(q,srcBase,c32(q,ky*w+kx));
                  Value globalK=addInt(q,
                      mulInt(q,first,c32(q,kh*kw)),c32(q,t));
                  Value dstBase=mulInt(q,
                      subInt(q,globalK,k032),c32(q,PITCH));
                  Value byChannels=cmpInt(q,arith::CmpIPredicate::ugt,num,rows32);
                  Value repetitions=q.createOrFold<arith::SelectOp>(loc,byChannels,num,rows32);
                  Value lastCount32=mulInt(q,repetitions,c32(q,ow));
                  q.create<memref::StoreOp>(loc,lastCount32,dmaDirectCount,ValueRange{z});
                  q.create<scf::IfOp>(loc,byChannels,[&](OpBuilder &v,Location) {
                    loop32(v,rows32,[&](OpBuilder &u,Value r) {
                      Value so=addInt(u,srcBase,
                          mulInt(u,r,c32(u,w)));
                      Value dest=addInt(u,dstBase,
                          mulInt(u,r,c32(u,ow)));
                      emit(u,so,dest,num,h*w,kh*kw*PITCH);
                    });
                    v.create<scf::YieldOp>(loc);
                  },[&](OpBuilder &v,Location) {
                    loop32(v,num,[&](OpBuilder &u,Value c) {
                      Value so=addInt(u,srcBase,
                          mulInt(u,c,c32(u,h*w)));
                      Value dest=addInt(u,dstBase,
                          mulInt(u,c,c32(u,kh*kw*PITCH)));
                      emit(u,so,dest,rows32,w,ow);
                    });
                    v.create<scf::YieldOp>(loc);
                  });
                  q.create<scf::YieldOp>(loc);
                });
              }
            }

            if (!deferWait)
              waitDirectOperands(ib,k0,targetM);
            return;
          }

          if (batch == 1 && ow == 16 && kh == 3 && kw == 3 &&
              oh*ow % TM == 0 && mTileStep == TM && RAW_ELEMS >= 9*108) {
            constexpr int64_t RAW_ROWS = 6, RAW_COLS = 18;
            constexpr int64_t RAW_PER_CHANNEL = RAW_ROWS * RAW_COLS;
            Value firstC=divUInt(ib,k0,ci(9));
            Value kLast=subInt(ib,
                addInt(ib,k0,kE),one);
            Value endC=addInt(ib,
                divUInt(ib,kLast,ci(9)),one);
            Value srcRow=divUInt(ib,targetM,ci(ow));
            Value rawRows=ci(RAW_ROWS), rawCols=ci(RAW_COLS);
            Value rawCount=ci(RAW_PER_CHANNEL);
            MLIRContext *rawCtx=ib.getContext();
            auto rawLayout=AffineMapAttr::get(AffineMap::get(2,1,
                getAffineDimExpr(0,rawCtx)*RAW_COLS+
                getAffineDimExpr(1,rawCtx)+getAffineSymbolExpr(0,rawCtx)));
            auto rawTy=MemRefType::get(
                {ShapedType::kDynamicSize,ShapedType::kDynamicSize},
                ib.getF32Type(),rawLayout,l1);
            auto getRawChannel=[&](OpBuilder &ob,Value slot)->Value {
              Value offset=mulInt(ob,slot,rawCount);
              return ob.create<memref::ReinterpretCastOp>(loc,rawTy,
                  raw,offset,SmallVector<OpFoldResult,2>{rawRows,rawCols},
                  SmallVector<OpFoldResult,2>{rawCols,one}).getResult();
            };

            ib.create<scf::ForOp>(loc,firstC,endC,one,ValueRange{},
                [&](OpBuilder &bd,Location,Value c,ValueRange){
              Value slot=subInt(bd,c,firstC);
              SmallVector<OpFoldResult,4> offsets={z,c,srcRow,z};
              SmallVector<OpFoldResult,4> sizes={bd.getIndexAttr(1),
                  bd.getIndexAttr(1),rawRows,rawCols};
              SmallVector<OpFoldResult,4> strides={bd.getIndexAttr(1),
                  bd.getIndexAttr(1),bd.getIndexAttr(1),bd.getIndexAttr(1)};
              auto srcType=memref::SubViewOp::inferRankReducedResultType(
                  2,xt,offsets,sizes,strides).cast<MemRefType>();
              Value inputWindow=bd.create<memref::SubViewOp>(loc,srcType,
                  x,offsets,sizes,strides).getResult();
              Value rawChannel=getRawChannel(bd,slot);
              emitConvSequentialDrain(bd,loc);
              bd.create<memref::DmaStartOp>(loc,inputWindow,ValueRange{z,z},
                  rawChannel,ValueRange{z,z},rawCount,dmaTagRawGroup,
                  ValueRange{slot});
              bd.create<scf::YieldOp>(loc);
            });

            ib.create<scf::ForOp>(loc,firstC,endC,one,ValueRange{},
                [&](OpBuilder &bw,Location,Value c,ValueRange){
              Value slot=subInt(bw,c,firstC);
              bw.create<memref::DmaWaitOp>(loc,dmaTagRawGroup,
                  ValueRange{slot},rawCount);
              bw.create<scf::YieldOp>(loc);
            });

            Value vl16=ib.create<arith::ConstantIntOp>(loc,16,32);
            ib.create<scf::ForOp>(loc,firstC,endC,one,ValueRange{},
                [&](OpBuilder &bc,Location,Value c,ValueRange){
              Value slot=subInt(bc,c,firstC);
              Value rawChannel=getRawChannel(bc,slot);
              for (int64_t ky=0; ky<3; ++ky) {
                for (int64_t kx=0; kx<3; ++kx) {
                  Value flat=addInt(bc,
                      mulInt(bc,c,ci(9)),ci(ky*3+kx));
                  Value lo=cmpInt(bc,
                      arith::CmpIPredicate::uge,flat,k0);
                  Value hi=cmpInt(bc,
                      arith::CmpIPredicate::ult,flat,
                      addInt(bc,k0,kE));
                  Value valid=bc.create<arith::AndIOp>(loc,lo,hi);
                  bc.create<scf::IfOp>(loc,valid,[&](OpBuilder &bv,Location){
                    Value dstK=subInt(bv,flat,k0);
                    for (int64_t ry=0; ry<4; ++ry) {
                      Value sourceY=ci(ry+ky);
                      Value sourceX=ci(kx);
                      Value targetX=ci(ry*16);
                      Value vec=bv.create<spatz::VLEOp>(loc,vectorTy,
                          rawChannel,ValueRange{sourceY,sourceX},vl16);
                      bv.create<spatz::VSEOp>(loc,vec,activeA,
                          ValueRange{dstK,targetX},vl16);
                    }
                    bv.create<scf::YieldOp>(loc);
                  });
                }
              }
              bc.create<scf::YieldOp>(loc);
            });

            waitWeights(ib,k0,targetM,weightCount);

            return;
          }
          Value batchLim=ib.create<arith::ConstantIndexOp>(loc,batch);
          loop(ib,batchLim,one,[&](OpBuilder &bb,Value bi){
            Value batch32=bb.createOrFold<arith::IndexCastOp>(loc,bb.getI32Type(),bi);
            Value base=mul(bb,batch32,hw32);
            Value begin=umax(bb,m032,base);
            Value finish=umin(bb,add(bb,m032,mE32),add(bb,base,hw32));
            Value nonempty=cmpInt(bb,arith::CmpIPredicate::ult,begin,finish);
            bb.create<scf::IfOp>(loc,nonempty,[&](OpBuilder &bs,Location){
              Value spatial=sub(bs,begin,base);
              Value endSpatial=sub(bs,finish,base);
              Value firstRow=divUInt(bs,spatial,ow32);
              Value firstCol=remUInt(bs,spatial,ow32);
              Value len=sub(bs,endSpatial,spatial);
              Value headLimit=umin(bs,len,sub(bs,ow32,firstCol));
              Value headLen=bs.createOrFold<arith::SelectOp>(loc,
                  cmpInt(bs,arith::CmpIPredicate::ne,firstCol,c0),headLimit,c0);
              Value wholeStart=add(bs,spatial,headLen);
              Value wholeRows=divUInt(bs,sub(bs,endSpatial,wholeStart),ow32);
              Value wholeRow=divUInt(bs,wholeStart,ow32);
              Value tailStart=add(bs,wholeStart,mul(bs,wholeRows,ow32));
              Value tailLen=sub(bs,endSpatial,tailStart);
              Value tailRow=add(bs,wholeRow,wholeRows);

              Value cK0=bs.create<arith::ConstantIntOp>(loc,kh*kw,32);
              Value firstChannel32=divUInt(bs,k032,cK0);
              Value lastK32=subInt(bs,
                  addInt(bs,k032,kE32),c1);
              Value lastChannel32=addInt(bs,
                  divUInt(bs,lastK32,cK0),c1);
              Value firstChannel=ix(bs,firstChannel32);
              Value endChannel=ix(bs,lastChannel32);
              bs.create<scf::ForOp>(loc,firstChannel,endChannel,one,ValueRange{},
                  [&](OpBuilder &bc,Location,Value c,ValueRange){
                Value c32=bc.createOrFold<arith::IndexCastOp>(loc,bc.getI32Type(),c);
                auto emitRect=[&](OpBuilder &br,Value spatialOffset,Value row,Value col,Value rows,Value cols, bool wholeAlignedTile){

                  Value rawRows=add(br,rows,br.create<arith::ConstantIntOp>(loc,kh-1,32));
                  Value rawCols=add(br,cols,br.create<arith::ConstantIntOp>(loc,kw-1,32));
                  Value nRaw=mul(br,rawRows,rawCols);
                  Value rr=ix(br,rawRows), rc=ix(br,rawCols);
                  SmallVector<OpFoldResult,4> srcOff={bi,c,ix(br,row),ix(br,col)};
                  SmallVector<OpFoldResult,4> srcSizes={br.getIndexAttr(1),br.getIndexAttr(1),rr,rc};
                  SmallVector<OpFoldResult,4> srcStrides={br.getIndexAttr(1),br.getIndexAttr(1),br.getIndexAttr(1),br.getIndexAttr(1)};
                  auto srcTy=memref::SubViewOp::inferRankReducedResultType(2,xt,srcOff,srcSizes,srcStrides).cast<MemRefType>();
                  Value src=br.create<memref::SubViewOp>(loc,srcTy,x,srcOff,srcSizes,srcStrides).getResult();
                  MLIRContext *ctx=br.getContext();
                  auto rawMap=AffineMapAttr::get(AffineMap::get(2,1,
                      getAffineDimExpr(0,ctx)*getAffineSymbolExpr(0,ctx)
                        + getAffineDimExpr(1,ctx)));
                  auto rawTy=MemRefType::get({ShapedType::kDynamicSize,ShapedType::kDynamicSize},
                      br.getF32Type(),rawMap,l1);
                  Value raw2=br.create<memref::ReinterpretCastOp>(loc,rawTy,raw,z,
                      SmallVector<OpFoldResult,2>{rr,rc},
                      SmallVector<OpFoldResult,2>{rc,br.getIndexAttr(1)}).getResult();

                  Value nRawIdx=ix(br,nRaw);
                  emitConvSequentialDrain(br,loc);
                  br.create<memref::DmaStartOp>(loc,src,ValueRange{z,z},raw2,
                      ValueRange{z,z},nRawIdx,dmaTag,ValueRange{z});

                  Value khLim=br.create<arith::ConstantIndexOp>(loc,kh);
                  Value kwLim=br.create<arith::ConstantIndexOp>(loc,kw);
                  br.create<memref::DmaWaitOp>(loc,dmaTag,ValueRange{z},nRawIdx);

                  if (wholeAlignedTile && (ow % 16 == 0) && (TM % ow == 0)
                      && (oh * ow % TM == 0) && (batch == 1)) {
                    Value dstBase=ix(br,sub(br,add(br,base,spatialOffset),m032));
                    Value vl16=br.create<arith::ConstantIntOp>(loc,16,32);
                    for (int64_t sy=0; sy<kh; ++sy) {
                      for (int64_t sx=0; sx<kw; ++sx) {
                        Value flat=add(br,mul(br,c32,br.create<arith::ConstantIntOp>(loc,kh*kw,32)),
                                       br.create<arith::ConstantIntOp>(loc,sy*kw+sx,32));
                        Value validLo=cmpInt(br,arith::CmpIPredicate::uge,flat,k032);
                        Value validHi=cmpInt(br,arith::CmpIPredicate::ult,flat,add(br,k032,kE32));
                        Value valid=br.create<arith::AndIOp>(loc,validLo,validHi);
                        br.create<scf::IfOp>(loc,valid,[&](OpBuilder &bv,Location){
                          Value dstK=ix(bv,sub(bv,flat,k032));
                          for (int64_t ry=0; ry<TM/ow; ++ry) {
                            for (int64_t xx=0; xx<ow; xx+=16) {
                              Value srcY=bv.create<arith::ConstantIndexOp>(loc,ry+sy);
                              Value srcX=bv.create<arith::ConstantIndexOp>(loc,xx+sx);
                              Value dstX=addInt(bv,dstBase,
                                  bv.create<arith::ConstantIndexOp>(loc,ry*ow+xx));
                              Value v=bv.create<spatz::VLEOp>(loc,vectorTy,raw2,
                                               ValueRange{srcY,srcX},vl16);
                              bv.create<spatz::VSEOp>(loc,v,activeA,ValueRange{dstK,dstX},vl16);
                            }
                          }
                          bv.create<scf::YieldOp>(loc);
                        });
                      }
                    }
                  } else {
                  loop(br,khLim,one,[&](OpBuilder &bkY,Value ky){
                    Value ky32=bkY.createOrFold<arith::IndexCastOp>(loc,bkY.getI32Type(),ky);
                    loop(bkY,kwLim,one,[&](OpBuilder &bkX,Value kx){
                      Value kx32=bkX.createOrFold<arith::IndexCastOp>(loc,bkX.getI32Type(),kx);
                      Value flat=add(bkX,mul(bkX,add(bkX,mul(bkX,c32,kh32),ky32),kw32),kx32);
                      Value validLo=cmpInt(bkX,arith::CmpIPredicate::uge,flat,k032);
                      Value validHi=cmpInt(bkX,arith::CmpIPredicate::ult,flat,add(bkX,k032,kE32));
                      Value valid=bkX.create<arith::AndIOp>(loc,validLo,validHi);
                      bkX.create<scf::IfOp>(loc,valid,[&](OpBuilder &bv,Location){
                        Value dstK=ix(bv,sub(bv,flat,k032));
                        Value off=ix(bv,sub(bv,add(bv,base,spatialOffset),m032));
                        Value kyI=ix(bv,ky32), kxI=ix(bv,kx32);
                        Value rLim=ix(bv,rows);
                        loop(bv,rLim,one,[&](OpBuilder &by,Value r){
                          Value rawY=addInt(by,r,kyI);
                          Value dstRowOff=mulInt(by,r,ix(by,cols));
                          Value dM=addInt(by,off,dstRowOff);
                          Value colLim=ix(by,cols);
                          Value step=by.create<arith::ConstantIndexOp>(loc,16);
                          loop(by,colLim,step,[&](OpBuilder &bx,Value colOff){
                            Value left=subInt(bx,colLim,colOff);
                            Value vlIdx=minIndex(bx,left,step);
                            Value vl=bx.createOrFold<arith::IndexCastOp>(loc,bx.getI32Type(),vlIdx);
                            Value srcX=addInt(bx,kxI,colOff);
                            Value destX=addInt(bx,dM,colOff);
                            Value vec=bx.create<spatz::VLEOp>(loc,vectorTy,raw2,ValueRange{rawY,srcX},vl);
                            bx.create<spatz::VSEOp>(loc,vec,activeA,ValueRange{dstK,destX},vl);
                          });
                        });
                        bv.create<scf::YieldOp>(loc);
                      });
                    });
                  });
                  }
                };
                Value head=cmpInt(bc,arith::CmpIPredicate::ne,headLen,c0);
                bc.create<scf::IfOp>(loc,head,[&](OpBuilder &bp,Location){
                  emitRect(bp,spatial,firstRow,firstCol,c1,headLen,false);
                  bp.create<scf::YieldOp>(loc);
                });
                Value hasWhole=cmpInt(bc,arith::CmpIPredicate::ne,wholeRows,c0);
                bc.create<scf::IfOp>(loc,hasWhole,[&](OpBuilder &bp,Location){
                  emitRect(bp,wholeStart,wholeRow,c0,wholeRows,ow32,true);
                  bp.create<scf::YieldOp>(loc);
                });
                Value hasTail=cmpInt(bc,arith::CmpIPredicate::ne,tailLen,c0);
                bc.create<scf::IfOp>(loc,hasTail,[&](OpBuilder &bp,Location){
                  emitRect(bp,tailStart,tailRow,c0,c1,tailLen,false);
                  bp.create<scf::YieldOp>(loc);
                });
                bc.create<scf::YieldOp>(loc);
              });
              bs.create<scf::YieldOp>(loc);
            });
          });

          waitWeights(ib,k0,targetM,weightCount);
        };
        auto prepareOperands = [&](OpBuilder &ib, Value k0, Value targetM, Value targetN,
                                   Value targetME, Value targetAbShift) {
          prepareOperandsImpl(ib,k0,targetM,targetN,targetME,targetAbShift,false);
        };

        auto accumulatePanel=[&](OpBuilder &ib,Value panel){
          Value parity=remUInt(ib,panel,ci(2));
          Value odd=cmpInt(ib,arith::CmpIPredicate::ne,parity,z);
          Value accSub=createSubview2D(ib,loc,acc,z,z,mE,nE,ib.getF32Type(),l1);

          ib.create<scf::IfOp>(loc,odd,[&](OpBuilder &ob,Location){
            Value pSub=createSubview2D(ob,loc,partial0,z,z,mE,nE,ob.getF32Type(),l1);
            ob.create<spatz::MatrixAddOp>(loc, accSub, pSub, mE, nE,
                ob.getI64IntegerAttr(64), ob.getI32IntegerAttr(6));
            ob.create<scf::YieldOp>(loc);
          },[&](OpBuilder &eb,Location){
            Value pSub=createSubview2D(eb,loc,partial1,z,z,mE,nE,eb.getF32Type(),l1);
            eb.create<spatz::MatrixAddOp>(loc, accSub, pSub, mE, nE,
                eb.getI64IntegerAttr(64), eb.getI32IntegerAttr(6));
            eb.create<scf::YieldOp>(loc);
          });
        };

        bn.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){
          if (crossTilePrefetch) {
            Value firstTile=cmpInt(ib,
                arith::CmpIPredicate::eq,tileSerial,z);
            ib.create<scf::IfOp>(loc,firstTile,[&](OpBuilder &pr,Location){
              prepareOperands(pr,z,m0,n0,mE,abShift);
              pr.create<scf::YieldOp>(loc);
            });
          } else {
            prepareOperands(ib,z,m0,n0,mE,abShift);
          }
          ib.create<scf::YieldOp>(loc);
        });
        Value kBegin=z;
        if (overlapNextMatmul) {

          Value firstSpatialTile=cmpInt(bn,
              arith::CmpIPredicate::eq,tileSerial,z);
          bn.create<scf::IfOp>(loc,firstSpatialTile,[&](OpBuilder &ib,Location){
            sync(ib);
            ib.create<scf::YieldOp>(loc);
          });
          kBegin=bn.createOrFold<arith::SelectOp>(loc,firstSpatialTile,z,tk);
        } else {
          sync(bn);
        }

        auto kStages=[&](auto &&body) {
          const int64_t fullEnd=totalK/TK*TK;
          if (fullEnd == TK && totalK == TK) {
            body(bn,z,tk);
          } else if (fullEnd != 0) {
            Value end=bn.create<arith::ConstantIndexOp>(loc,fullEnd);
            tileLoop(bn,kBegin,end,tk,[&](OpBuilder &kb,Value k0) {
              body(kb,k0,tk);
            });
          }
          if (fullEnd != totalK) {
            Value last=bn.create<arith::ConstantIndexOp>(loc,fullEnd);
            Value extent=bn.create<arith::ConstantIndexOp>(loc,totalK-fullEnd);
            body(bn,last,extent);
          }
        };
        kStages([&](OpBuilder &bk,Value k0,Value kE){
          Value qk=divUInt(bk,k0,tk);
          Value isFirst=cmpInt(bk,arith::CmpIPredicate::eq,qk,z);
          Value parity=remUInt(bk,qk,ci(2));
          Value isOdd=cmpInt(bk,arith::CmpIPredicate::ne,parity,z);
          bk.create<scf::IfOp>(loc,isCore1,[&](OpBuilder &ib,Location){
            Value effectiveK=addInt(ib,k0,abShift);
            Value activeA=chooseAB(ib,effectiveK,at0,at1,tileTy);
            Value activeB=selectB(ib,n0,k0,effectiveK);
            Value aSub=createSubview2D(ib,loc,activeA,z,z,kE,mE,b.getF32Type(),l1);
            Value gSub=createSubview2D(ib,loc,activeB,z,z,kE,nE,b.getF32Type(),l1);
            auto emitMatmul=[&](OpBuilder &mb,Value dest){
              Value cSub=createSubview2D(mb,loc,dest,z,z,mE,nE,
                                          mb.getF32Type(),l1);
              mb.create<quadrilatero::TcdmMatmulMemRefOp>(loc,
                  aSub,gSub,cSub,mE,nE,kE,z,
                  mb.getI32IntegerAttr(6),mb.getI32IntegerAttr(6),
                  mb.getI32IntegerAttr(6));
            };
            ib.create<scf::IfOp>(loc,isFirst,[&](OpBuilder &fb,Location){

              emitMatmul(fb,acc);
              fb.create<scf::YieldOp>(loc);
            },[&](OpBuilder &nf,Location){
              nf.create<scf::IfOp>(loc,isOdd,[&](OpBuilder &ob,Location){
                emitMatmul(ob,partial0);
                ob.create<scf::YieldOp>(loc);
              },[&](OpBuilder &eb,Location){
                emitMatmul(eb,partial1);
                eb.create<scf::YieldOp>(loc);
              });
              nf.create<scf::YieldOp>(loc);
            });
            ib.create<scf::YieldOp>(loc);
          });

          if (sequential) sync(bk);
          bk.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){
            Value nextK=addInt(ib,k0,tk);
            Value hasNext=cmpInt(ib,arith::CmpIPredicate::ult,nextK,kEnd);
            Value hasPrevious=cmpInt(ib,arith::CmpIPredicate::ugt,qk,ci(1));
            if (directIm2colDMA) {

              Value nextM=nextTileM;
              Value lastPanel=cmpInt(ib,
                  arith::CmpIPredicate::eq,qk,ci(kt-1));
              Value nextExists=nextTileExists;
              Value nextSpatial=ib.create<arith::AndIOp>(loc,lastPanel,nextExists);
              auto pendingPreparation = [&](OpBuilder &pb, bool issue) {
                pb.create<scf::IfOp>(loc,hasNext,[&](OpBuilder &pr,Location){
                  if (issue)
                    prepareOperandsImpl(pr,nextK,m0,n0,mE,abShift,true);
                  else
                    waitDirectOperands(pr,nextK,m0);
                  pr.create<scf::YieldOp>(loc);
                });
                if (crossTilePrefetch) {
                  pb.create<scf::IfOp>(loc,nextSpatial,[&](OpBuilder &pr,Location){
                    if (issue) {
                      Value nextME=tileExtent(pr,nextM,totalM,mTileStep);
                      prepareOperandsImpl(pr,z,nextM,nextTileN,nextME,nextAbShift,true);
                    } else {
                      waitDirectOperands(pr,z,nextM);
                    }
                    pr.create<scf::YieldOp>(loc);
                  });
                }
              };
              pendingPreparation(ib,true);
              if (sequential) pendingPreparation(ib,false);
              ib.create<scf::IfOp>(loc,hasPrevious,[&](OpBuilder &ac,Location){
                Value prev=subInt(ac,qk,one);
                accumulatePanel(ac,prev);
                ac.create<scf::YieldOp>(loc);
              });
              if (!sequential) pendingPreparation(ib,false);
            } else {
              ib.create<scf::IfOp>(loc,hasNext,[&](OpBuilder &pr,Location){
                prepareOperands(pr,nextK,m0,n0,mE,abShift);
                pr.create<scf::YieldOp>(loc);
              });
              ib.create<scf::IfOp>(loc,hasPrevious,[&](OpBuilder &ac,Location){
                Value prev=subInt(ac,qk,one);
                accumulatePanel(ac,prev);
                ac.create<scf::YieldOp>(loc);
              });

              if (crossTilePrefetch) {
                Value lastPanel=cmpInt(ib,
                    arith::CmpIPredicate::eq,qk,ci(kt-1));
                Value nextM=nextTileM;
                Value nextExists=nextTileExists;
                Value shouldPrefetch=ib.create<arith::AndIOp>(loc,
                    lastPanel,nextExists);
                ib.create<scf::IfOp>(loc,shouldPrefetch,[&](OpBuilder &pr,Location){
                  Value nextME=tileExtent(pr,nextM,totalM,mTileStep);
                  prepareOperands(pr,z,nextM,nextTileN,nextME,nextAbShift);
                  pr.create<scf::YieldOp>(loc);
                });
              }
            }
            ib.create<scf::YieldOp>(loc);
          });
          bk.create<scf::IfOp>(loc,isCore1,[&](OpBuilder &p,Location){

            p.create<scf::YieldOp>(loc);
          });
          sync(bk);
          bk.create<scf::IfOp>(loc,isCore1,[&](OpBuilder &p,Location){

            p.create<scf::YieldOp>(loc);
          });
        });

        if (overlapNextMatmul) {
          bn.create<scf::IfOp>(loc,isCore1,[&](OpBuilder &ib,Location){
            Value nextM=nextTileM;
            Value nextExists=nextTileExists;
            ib.create<scf::IfOp>(loc,nextExists,[&](OpBuilder &nb,Location){
              Value nextME=tileExtent(nb,nextM,totalM,mTileStep);
              Value firstKE=minIndex(nb,kEnd,tk);
              Value nextA=chooseAB(nb,nextAbShift,at0,at1,tileTy);
              Value nextB=selectB(nb,nextTileN,z,nextAbShift);
              Value aSub=createSubview2D(nb,loc,nextA,z,z,firstKE,nextME,
                                        nb.getF32Type(),l1);
              Value bSub=createSubview2D(nb,loc,nextB,z,z,firstKE,nE,
                                        nb.getF32Type(),l1);
              Value cSub=createSubview2D(nb,loc,partial0,z,z,nextME,nE,
                                        nb.getF32Type(),l1);

              nb.create<quadrilatero::TcdmMatmulMemRefOp>(loc,
                  aSub,bSub,cSub,nextME,nE,firstKE,z,
                  nb.getI32IntegerAttr(6),nb.getI32IntegerAttr(6),
                  nb.getI32IntegerAttr(6));
              nb.create<scf::YieldOp>(loc);
            });
            ib.create<scf::YieldOp>(loc);
          });
        }

        auto prepareNextSecondPanel = [&](bool issue) {
          if (!overlapNextMatmul)
            return;
          bn.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){
            Value nextM=nextTileM;
            Value nextExists=nextTileExists;
            ib.create<scf::IfOp>(loc,nextExists,[&](OpBuilder &nb,Location){
              if (issue) {
                Value nextME=tileExtent(nb,nextM,totalM,mTileStep);
                prepareOperandsImpl(nb,tk,nextM,nextTileN,nextME,nextAbShift,true);
              } else {
                waitDirectOperands(nb,tk,nextM);
              }
              nb.create<scf::YieldOp>(loc);
            });
            ib.create<scf::YieldOp>(loc);
          });
        };
        prepareNextSecondPanel(true);

        bn.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){
          Value lastPanel=ib.create<arith::ConstantIndexOp>(loc,kt-1);
          Value hasAccum=cmpInt(ib,arith::CmpIPredicate::ugt,lastPanel,z);
          ib.create<scf::IfOp>(loc,hasAccum,[&](OpBuilder &ac,Location){
            accumulatePanel(ac,lastPanel);
            ac.create<scf::YieldOp>(loc);
          });
          ib.create<scf::YieldOp>(loc);
        });
        prepareNextSecondPanel(false);
        bn.create<scf::IfOp>(loc,isCore0,[&](OpBuilder &ib,Location){

          Value src2d=createSubview2D(ib,loc,acc,z,z,mE,nE,
                                      ib.getF32Type(),l1);
          Value dstOffset=addInt(ib,
              mulInt(ib,m0,cCout),n0);
          MLIRContext *dstCtx=ib.getContext();
          auto dstLayout=AffineMapAttr::get(AffineMap::get(2,1,
              getAffineDimExpr(0,dstCtx)*cout+
              getAffineDimExpr(1,dstCtx)+getAffineSymbolExpr(0,dstCtx)));
          auto dstTy=MemRefType::get(
              {ShapedType::kDynamicSize,ShapedType::kDynamicSize},
              ib.getF32Type(),dstLayout,yt.getMemorySpace());
          Value dst2d=ib.create<memref::ReinterpretCastOp>(loc,dstTy,
              yFlat,dstOffset,SmallVector<OpFoldResult,2>{mE,nE},
              SmallVector<OpFoldResult,2>{cCout,one}).getResult();
          Value count=mulInt(ib,mE,nE);
          emitConvSequentialDrain(ib,loc);
          ib.create<memref::DmaStartOp>(loc,src2d,ValueRange{z,z},
              dst2d,ValueRange{z,z},count,dmaTagOutput,ValueRange{z});
          if (deferOutputWait) {
            Value nextM=nextTileM;
            Value isLastTile=cmpInt(ib,arith::CmpIPredicate::uge,nextM,mEnd);
            ib.create<scf::IfOp>(loc,isLastTile,[&](OpBuilder &wb,Location){

              wb.create<memref::DmaWaitOp>(loc,dmaTagOutput,ValueRange{z},count);
              wb.create<scf::YieldOp>(loc);
            });
          } else {
            ib.create<memref::DmaWaitOp>(loc,dmaTagOutput,ValueRange{z},count);
          }

          ib.create<scf::YieldOp>(loc);
        });
        sync(bn);
      });
    });
    sync(b);

    b.create<func::CallOp>(loc, lowerEndFn, ValueRange{});
    op.erase();
    return success();
  }

  LogicalResult lowerMatmul(linalg::MatmulOp op) {
    if (op.getNumInputs() != 2 || op.getNumOutputs() != 1) return failure();

    Value a_orig = op.inputs()[0];
    Value b = op.inputs()[1];
    Value c = op.outputs()[0];

    auto aOrigType = a_orig.getType().dyn_cast<MemRefType>();
    auto bType = b.getType().dyn_cast<MemRefType>();
    auto cType = c.getType().dyn_cast<MemRefType>();
    if (!aOrigType || !bType || !cType) return failure();

    bool hasScale = false;
    Value scaleVal = nullptr;
    Operation* scaleOpToErase = nullptr;

    bool hasBias = false;
    Value biasVal = nullptr;
    Operation* addOpToErase = nullptr;

    Value mainC = c;
    if (auto subview = c.getDefiningOp<memref::SubViewOp>()) {
      mainC = subview.source();
    }

    for (Operation* user : mainC.getUsers()) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
        auto &block = genericOp.getRegion().front();
        if (block.getOperations().size() == 2) {
          if (genericOp.getNumInputs() == 1) {
            if (auto mulfOp = dyn_cast<arith::MulFOp>(block.front())) {
              hasScale = true;
              scaleVal = (mulfOp.getLhs() == block.getArgument(0)) ? mulfOp.getRhs() : mulfOp.getLhs();
              scaleOpToErase = genericOp;
            }
          }
          else if (genericOp.getNumInputs() == 2) {
            if (auto addfOp = dyn_cast<arith::AddFOp>(block.front())) {
              hasBias = true;
              biasVal = genericOp.getInputOperand(1)->get();
              addOpToErase = genericOp;
            }
          }
        }
      }
    }

    OpBuilder builder(op);
    Location loc = op.getLoc();
    MLIRContext *ctx = builder.getContext();

    Value dimM_orig = builder.create<memref::DimOp>(loc, a_orig, 0);
    Value dimK_orig = builder.create<memref::DimOp>(loc, a_orig, 1);

    ArrayRef<int64_t> aOrigShape = aOrigType.getShape();
    auto aType = MemRefType::get(
        {aOrigShape[1], aOrigShape[0]},
        aOrigType.getElementType(),
        MemRefLayoutAttrInterface{},
        aOrigType.getMemorySpace());

    SmallVector<Value, 2> dynSizesA;
    if (aOrigShape[1] == ShapedType::kDynamicSize) dynSizesA.push_back(dimK_orig);
    if (aOrigShape[0] == ShapedType::kDynamicSize) dynSizesA.push_back(dimM_orig);

    Value a = builder.create<memref::AllocOp>(loc, aType, dynSizesA);

    AffineMap mapIn = AffineMap::getMultiDimIdentityMap(2, ctx);
    AffineMap mapOut = AffineMap::get(2, 0,
        {builder.getAffineDimExpr(1), builder.getAffineDimExpr(0)}, ctx);
    SmallVector<AffineMap, 2> indexingMaps = {mapIn, mapOut};

    SmallVector<StringRef, 2> iteratorTypes = {"parallel", "parallel"};

    builder.create<linalg::GenericOp>(
        loc,
        TypeRange{},
        ValueRange{a_orig},
        ValueRange{a},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        }
    );

    Type aElemType = aType.getElementType();
    Type bElemType = bType.getElementType();
    Type cElemType = cType.getElementType();

    int32_t dtA_val = getDataTypeCode(aElemType);
    int32_t dtB_val = getDataTypeCode(bElemType);
    int32_t dtC_val = getDataTypeCode(cElemType);

    if (auto attrA = op->getAttrOfType<StringAttr>("quadrilatero.type_a")) {
      if (attrA.getValue() == "fp8e4m3") dtA_val = 4;
      else if (attrA.getValue() == "fp8e5m2") dtA_val = 12;
    }
    if (auto attrB = op->getAttrOfType<StringAttr>("quadrilatero.type_b")) {
      if (attrB.getValue() == "fp8e4m3") dtB_val = 4;
      else if (attrB.getValue() == "fp8e5m2") dtB_val = 12;
    }
    if (auto attrC = op->getAttrOfType<StringAttr>("quadrilatero.type_c")) {
      if (attrC.getValue() == "fp32") dtC_val = 6;
    }

    unsigned bitWidth = aElemType.getIntOrFloatBitWidth();
    int64_t tileKVal = (bitWidth == 32) ? 64 : (bitWidth == 16) ? 128 : 256;
    int64_t tileMVal = 64, tileNVal = 64;
    int64_t shiftVal = (bitWidth == 32) ? 0 : (bitWidth == 16) ? 1 : 2;

    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = builder.getI32Type();
    auto indexTy = builder.getIndexType();
    SymbolTable symbolTable(module);

    auto getOrInsertFn = [&](StringRef name, FunctionType type) {
      if (auto fn = symbolTable.lookup<func::FuncOp>(name)) return fn;
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto fn = builder.create<func::FuncOp>(loc, name, type);
      fn.setPrivate();
      symbolTable.insert(fn);
      return fn;
    };

    auto getCoreIdxFn = getOrInsertFn("snrt_cluster_core_idx", builder.getFunctionType({}, {i32Ty}));
    auto hwBarrierFn = getOrInsertFn("snrt_cluster_hw_barrier", builder.getFunctionType({}, {}));
    auto l1ResetFn = getOrInsertFn("snrt_l1alloc_reset", builder.getFunctionType({}, {}));

    Value cid = builder.create<func::CallOp>(loc, getCoreIdxFn, ValueRange{}).getResult(0);
    Value isCore0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, builder.create<arith::ConstantIntOp>(loc, 0, 32));
    Value isCore1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, builder.create<arith::ConstantIntOp>(loc, 1, 32));

    auto l1SpaceAttr = IntegerAttr::get(builder.getI64Type(), 1);
    auto aL1Type = MemRefType::get({tileKVal, tileMVal}, aElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto bL1Type = MemRefType::get({tileKVal, tileNVal}, bElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto cAccType = MemRefType::get({tileMVal, tileNVal}, cElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);

    Value aL1_0 = builder.create<memref::AllocOp>(loc, aL1Type);
    Value aL1_1 = builder.create<memref::AllocOp>(loc, aL1Type);
    Value bL1_0 = builder.create<memref::AllocOp>(loc, bL1Type);
    Value bL1_1 = builder.create<memref::AllocOp>(loc, bL1Type);
    Value cBuf_0 = builder.create<memref::AllocOp>(loc, cAccType);
    Value cBuf_1 = builder.create<memref::AllocOp>(loc, cAccType);
    Value cBuf_2 = builder.create<memref::AllocOp>(loc, cAccType);

    auto tagType = MemRefType::get({1}, builder.getI32Type());
    Value tagA_0 = builder.create<memref::AllocaOp>(loc, tagType), tagB_0 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagA_1 = builder.create<memref::AllocaOp>(loc, tagType), tagB_1 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagC   = builder.create<memref::AllocaOp>(loc, tagType);

    Value c0_idx = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1_idx = builder.create<arith::ConstantIndexOp>(loc, 1);

    Value bias2D = nullptr;
    Value biasL1_1D = nullptr;
    Value biasL1_2D = nullptr;
    Value tagBias = nullptr;
    Attribute biasMemSpace = nullptr;

    if (hasBias) {
        auto biasType = biasVal.getType().cast<MemRefType>();
        biasMemSpace = biasType.getMemorySpace();

        SmallVector<ReassociationIndices> reassoc = {{0, 1}};
        auto bias2DType = MemRefType::get({1, biasType.getShape()[0]}, biasType.getElementType(), MemRefLayoutAttrInterface{}, biasMemSpace);
        bias2D = builder.create<memref::ExpandShapeOp>(loc, bias2DType, biasVal, reassoc);

        auto biasL1Type_1D = MemRefType::get({tileNVal}, cElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
        biasL1_1D = builder.create<memref::AllocOp>(loc, biasL1Type_1D);

        auto biasL1Type_2D = MemRefType::get({1, tileNVal}, cElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
        biasL1_2D = builder.create<memref::ExpandShapeOp>(loc, biasL1Type_2D, biasL1_1D, reassoc);

        tagBias = builder.create<memref::AllocaOp>(loc, tagType);
    }

    Value c0_i32 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value c1_i32 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    Value c2_i32 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
    Value tk_i32 = builder.create<arith::ConstantIntOp>(loc, tileKVal, 32);
    Value tm_i32 = builder.create<arith::ConstantIntOp>(loc, tileMVal, 32);
    Value tn_i32 = builder.create<arith::ConstantIntOp>(loc, tileNVal, 32);
    Value tm_idx = builder.create<arith::ConstantIndexOp>(loc, tileMVal);
    Value tn_idx = builder.create<arith::ConstantIndexOp>(loc, tileNVal);
    Value tk_idx = builder.create<arith::ConstantIndexOp>(loc, tileKVal);
    Value tk2_idx = builder.create<arith::ConstantIndexOp>(loc, tileKVal * 2);
    Value tk2_i32 = builder.create<arith::ConstantIntOp>(loc, tileKVal * 2, 32);

    Value dimK_idx = builder.create<memref::DimOp>(loc, a, 0);
    Value dimM_idx = builder.create<memref::DimOp>(loc, a, 1);
    Value dimN_idx = builder.create<memref::DimOp>(loc, b, 1);

    Value dimK = builder.create<arith::IndexCastOp>(loc, i32Ty, dimK_idx);
    Value dimM = builder.create<arith::IndexCastOp>(loc, i32Ty, dimM_idx);
    Value dimN = builder.create<arith::IndexCastOp>(loc, i32Ty, dimN_idx);

    auto ceilDivI32 = [&](Value dim, Value tile) {
        Value sub1 = builder.create<arith::SubIOp>(loc, dim, c1_i32);
        Value addTile = builder.create<arith::AddIOp>(loc, sub1, tile);
        return builder.create<arith::DivUIOp>(loc, addTile, tile);
    };
    Value itersK = ceilDivI32(dimK, tk_i32);
    Value itersN = ceilDivI32(dimN, tn_i32);
    Value itersM = ceilDivI32(dimM, tm_i32);

    Value itersKN = builder.create<arith::MulIOp>(loc, itersK, itersN);
    Value totalIters = builder.create<arith::MulIOp>(loc, itersKN, itersM);

    auto resolveBuf = [&](OpBuilder &b, Location l, Value id) -> Value {
        Value is0 = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq, id, c0_i32);
        Value is1 = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq, id, c1_i32);
        Value sel1 = b.create<arith::SelectOp>(l, is1, cBuf_1, cBuf_2);
        return b.create<arith::SelectOp>(l, is0, cBuf_0, sel1);
    };

    builder.create<scf::IfOp>(loc, isCore0, [&](OpBuilder &ifB, Location ifL) {
      Value mE_0_i32 = createMinI32(ifB, ifL, dimM, tm_i32);
      Value nE_0_i32 = createMinI32(ifB, ifL, dimN, tn_i32);
      Value kE_0_i32 = createMinI32(ifB, ifL, dimK, tk_i32);

      Value mE_0 = ifB.create<arith::IndexCastOp>(ifL, indexTy, mE_0_i32);
      Value nE_0 = ifB.create<arith::IndexCastOp>(ifL, indexTy, nE_0_i32);
      Value kE_0 = ifB.create<arith::IndexCastOp>(ifL, indexTy, kE_0_i32);

      Value aS_0 = createSubview2D(ifB, ifL, a, c0_idx, c0_idx, kE_0, mE_0, aElemType, aType.getMemorySpace());
      Value bS_0 = createSubview2D(ifB, ifL, b, c0_idx, c0_idx, kE_0, nE_0, bElemType, bType.getMemorySpace());
      Value aL_0 = createSubview2D(ifB, ifL, aL1_0, c0_idx, c0_idx, kE_0, mE_0, aElemType, l1SpaceAttr);
      Value bL_0 = createSubview2D(ifB, ifL, bL1_0, c0_idx, c0_idx, kE_0, nE_0, bElemType, l1SpaceAttr);

      Value szA_i32 = ifB.create<arith::MulIOp>(ifL, kE_0_i32, mE_0_i32);
      Value szB_i32 = ifB.create<arith::MulIOp>(ifL, kE_0_i32, nE_0_i32);
      Value szA_idx = ifB.create<arith::IndexCastOp>(ifL, indexTy, szA_i32);
      Value szB_idx = ifB.create<arith::IndexCastOp>(ifL, indexTy, szB_i32);

      ifB.create<memref::DmaStartOp>(ifL, aS_0, ValueRange{c0_idx, c0_idx}, aL_0, ValueRange{c0_idx, c0_idx}, szA_idx, tagA_0, ValueRange{c0_idx});
      ifB.create<memref::DmaStartOp>(ifL, bS_0, ValueRange{c0_idx, c0_idx}, bL_0, ValueRange{c0_idx, c0_idx}, szB_idx, tagB_0, ValueRange{c0_idx});
      ifB.create<memref::DmaWaitOp>(ifL, tagA_0, ValueRange{c0_idx}, szA_idx);
      ifB.create<memref::DmaWaitOp>(ifL, tagB_0, ValueRange{c0_idx}, szB_idx);
      ifB.create<scf::YieldOp>(ifL);
    });

    builder.create<func::CallOp>(loc, hwBarrierFn, ValueRange{});

    SmallVector<Value, 4> loopArgs = {c0_i32, c1_i32, c2_i32, c0_i32};

    builder.create<scf::IfOp>(loc, isCore0, [&](OpBuilder &b0, Location l0) {

        auto loopM = b0.create<scf::ForOp>(l0, c0_idx, dimM_idx, tm_idx, loopArgs,
            [&](OpBuilder &lbM, Location lM, Value m_idx, ValueRange argsM) {

                auto loopN = lbM.create<scf::ForOp>(lM, c0_idx, dimN_idx, tn_idx, argsM,
                    [&](OpBuilder &lbN, Location lN, Value n_idx, ValueRange argsN) {

                        auto loopK = lbN.create<scf::ForOp>(lN, c0_idx, dimK_idx, tk_idx, argsN,
                            [&](OpBuilder &lbK, Location ll, Value k_idx, ValueRange argsK) {

                                Value acc_id = argsK[0]; Value p1_id = argsK[1]; Value p2_id = argsK[2];
                                Value flatIdx = argsK[3];

                                Value mOff = lbK.create<arith::IndexCastOp>(ll, i32Ty, m_idx);
                                Value nOff = lbK.create<arith::IndexCastOp>(ll, i32Ty, n_idx);
                                Value kOff = lbK.create<arith::IndexCastOp>(ll, i32Ty, k_idx);

                                Value c_pong1 = resolveBuf(lbK, ll, p1_id);
                                Value c_pong2 = resolveBuf(lbK, ll, p2_id);

                                Value parity = lbK.create<arith::RemUIOp>(ll, flatIdx, c2_i32);
                                Value is_even = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::eq, parity, c0_i32);

                                Value a_next = lbK.create<arith::SelectOp>(ll, is_even, aL1_1, aL1_0);
                                Value b_next = lbK.create<arith::SelectOp>(ll, is_even, bL1_1, bL1_0);
                                Value tagA_next = lbK.create<arith::SelectOp>(ll, is_even, tagA_1, tagA_0);
                                Value tagB_next = lbK.create<arith::SelectOp>(ll, is_even, tagB_1, tagB_0);

                                Value mE = createMinI32(lbK, ll, lbK.create<arith::SubIOp>(ll, dimM, mOff), tm_i32);
                                Value nE = createMinI32(lbK, ll, lbK.create<arith::SubIOp>(ll, dimN, nOff), tn_i32);

                                Value mE_idx = lbK.create<arith::IndexCastOp>(ll, indexTy, mE);
                                Value nE_idx = lbK.create<arith::IndexCastOp>(ll, indexTy, nE);

                                Value is_k0 = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::eq, kOff, c0_i32);
                                Value not_first = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::ne, flatIdx, c0_i32);
                                Value is_k2_plus = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::sge, kOff, tk2_i32);
                                Value multiple_k = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::sgt, itersK, c1_i32);

                                lbK.create<scf::IfOp>(ll, is_k0, [&](OpBuilder &iB, Location iL) {
                                    iB.create<scf::IfOp>(iL, not_first, [&](OpBuilder &stB, Location stL) {
                                        Value prevFlat = stB.create<arith::SubIOp>(stL, flatIdx, c1_i32);
                                        Value p_m_n_idx = stB.create<arith::DivUIOp>(stL, prevFlat, itersK);
                                        Value p_n_iter = stB.create<arith::RemUIOp>(stL, p_m_n_idx, itersN);
                                        Value p_m_iter = stB.create<arith::DivUIOp>(stL, p_m_n_idx, itersN);
                                        Value p_nOff = stB.create<arith::MulIOp>(stL, p_n_iter, tn_i32);
                                        Value p_mOff = stB.create<arith::MulIOp>(stL, p_m_iter, tm_i32);
                                        Value p_mE = createMinI32(stB, stL, stB.create<arith::SubIOp>(stL, dimM, p_mOff), tm_i32);
                                        Value p_nE = createMinI32(stB, stL, stB.create<arith::SubIOp>(stL, dimN, p_nOff), tn_i32);

                                        Value p_mOff_idx = stB.create<arith::IndexCastOp>(stL, indexTy, p_mOff);
                                        Value p_nOff_idx = stB.create<arith::IndexCastOp>(stL, indexTy, p_nOff);
                                        Value p_mE_idx = stB.create<arith::IndexCastOp>(stL, indexTy, p_mE);
                                        Value p_nE_idx = stB.create<arith::IndexCastOp>(stL, indexTy, p_nE);

                                        stB.create<scf::IfOp>(stL, multiple_k, [&](OpBuilder &addB, Location addL) {
                                            Value cAccS = createSubview2D(addB, addL, c_pong1, c0_idx, c0_idx, p_mE_idx, p_nE_idx, cElemType, l1SpaceAttr);
                                            Value cPrevS = createSubview2D(addB, addL, c_pong2, c0_idx, c0_idx, p_mE_idx, p_nE_idx, cElemType, l1SpaceAttr);
                                            addB.create<spatz::MatrixAddOp>(addL, cAccS, cPrevS, p_mE_idx, p_nE_idx, builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
                                            addB.create<scf::YieldOp>(addL);
                                        });

                                        Value cOut = createSubview2D(stB, stL, c, p_mOff_idx, p_nOff_idx, p_mE_idx, p_nE_idx, cElemType, cType.getMemorySpace());
                                        Value cSrc = createSubview2D(stB, stL, c_pong1, c0_idx, c0_idx, p_mE_idx, p_nE_idx, cElemType, l1SpaceAttr);

                                        if (hasBias) {
                                            Value biasSub2D = createSubview2D(stB, stL, bias2D, c0_idx, p_nOff_idx, c1_idx, p_nE_idx, cElemType, biasMemSpace);
                                            Value biasL1Sub2D = createSubview2D(stB, stL, biasL1_2D, c0_idx, c0_idx, c1_idx, p_nE_idx, cElemType, l1SpaceAttr);
                                            Value szBias_idx = stB.create<arith::IndexCastOp>(stL, indexTy, p_nE);

                                            stB.create<memref::DmaStartOp>(stL, biasSub2D, ValueRange{c0_idx, c0_idx}, biasL1Sub2D, ValueRange{c0_idx, c0_idx}, szBias_idx, tagBias, ValueRange{c0_idx});
                                            stB.create<memref::DmaWaitOp>(stL, tagBias, ValueRange{c0_idx}, szBias_idx);

                                            SmallVector<OpFoldResult, 1> offsets1D = {c0_idx};
                                            SmallVector<OpFoldResult, 1> sizes1D = {p_nE_idx};
                                            SmallVector<OpFoldResult, 1> strides1D = {stB.getIndexAttr(1)};
                                            Value biasL1Sub1D = stB.create<memref::SubViewOp>(stL, biasL1_1D, offsets1D, sizes1D, strides1D);

                                            stB.create<spatz::MatrixVectorAddOp>(stL, cSrc, biasL1Sub1D, p_mE_idx, p_nE_idx,
                                                builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
                                        }

                                        if (hasScale) {
                                            stB.create<spatz::MatrixScalarMulOp>(stL, cSrc, scaleVal, p_mE_idx, p_nE_idx,
                                                builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
                                        }

                                        Value cSz_i32 = stB.create<arith::MulIOp>(stL, p_mE, p_nE);
                                        Value cSz_idx = stB.create<arith::IndexCastOp>(stL, indexTy, cSz_i32);

                                        stB.create<memref::DmaStartOp>(stL, cSrc, ValueRange{c0_idx, c0_idx}, cOut, ValueRange{c0_idx, c0_idx}, cSz_idx, tagC, ValueRange{c0_idx});
                                        stB.create<memref::DmaWaitOp>(stL, tagC, ValueRange{c0_idx}, cSz_idx);
                                        stB.create<scf::YieldOp>(stL);
                                    });
                                    iB.create<scf::YieldOp>(iL);
                                });

                                Value nextFlatIdx = lbK.create<arith::AddIOp>(ll, flatIdx, c1_i32);
                                Value hasNext = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::slt, nextFlatIdx, totalIters);
                                lbK.create<scf::IfOp>(ll, hasNext, [&](OpBuilder &fb, Location fl) {
                                    Value n_k = fb.create<arith::RemUIOp>(fl, nextFlatIdx, itersK);
                                    Value n_mn = fb.create<arith::DivUIOp>(fl, nextFlatIdx, itersK);
                                    Value n_n = fb.create<arith::RemUIOp>(fl, n_mn, itersN);
                                    Value n_m = fb.create<arith::DivUIOp>(fl, n_mn, itersN);
                                    Value nKOff = fb.create<arith::MulIOp>(fl, n_k, tk_i32);
                                    Value nNOff = fb.create<arith::MulIOp>(fl, n_n, tn_i32);
                                    Value nMOff = fb.create<arith::MulIOp>(fl, n_m, tm_i32);
                                    Value n_mE = createMinI32(fb, fl, fb.create<arith::SubIOp>(fl, dimM, nMOff), tm_i32);
                                    Value n_nE = createMinI32(fb, fl, fb.create<arith::SubIOp>(fl, dimN, nNOff), tn_i32);
                                    Value n_kE = createMinI32(fb, fl, fb.create<arith::SubIOp>(fl, dimK, nKOff), tk_i32);

                                    Value nKOff_idx = fb.create<arith::IndexCastOp>(fl, indexTy, nKOff);
                                    Value nNOff_idx = fb.create<arith::IndexCastOp>(fl, indexTy, nNOff);
                                    Value nMOff_idx = fb.create<arith::IndexCastOp>(fl, indexTy, nMOff);
                                    Value n_mE_idx = fb.create<arith::IndexCastOp>(fl, indexTy, n_mE);
                                    Value n_nE_idx = fb.create<arith::IndexCastOp>(fl, indexTy, n_nE);
                                    Value n_kE_idx = fb.create<arith::IndexCastOp>(fl, indexTy, n_kE);

                                    Value aS = createSubview2D(fb, fl, a, nKOff_idx, nMOff_idx, n_kE_idx, n_mE_idx, aElemType, aType.getMemorySpace());
                                    Value bS = createSubview2D(fb, fl, b, nKOff_idx, nNOff_idx, n_kE_idx, n_nE_idx, bElemType, bType.getMemorySpace());
                                    Value aL = createSubview2D(fb, fl, a_next, c0_idx, c0_idx, n_kE_idx, n_mE_idx, aElemType, l1SpaceAttr);
                                    Value bL = createSubview2D(fb, fl, b_next, c0_idx, c0_idx, n_kE_idx, n_nE_idx, bElemType, l1SpaceAttr);

                                    Value szA_i32 = fb.create<arith::MulIOp>(fl, n_kE, n_mE);
                                    Value szB_i32 = fb.create<arith::MulIOp>(fl, n_kE, n_nE);
                                    Value szA_idx = fb.create<arith::IndexCastOp>(fl, indexTy, szA_i32);
                                    Value szB_idx = fb.create<arith::IndexCastOp>(fl, indexTy, szB_i32);

                                    fb.create<memref::DmaStartOp>(fl, aS, ValueRange{c0_idx, c0_idx}, aL, ValueRange{c0_idx, c0_idx}, szA_idx, tagA_next, ValueRange{c0_idx});
                                    fb.create<memref::DmaStartOp>(fl, bS, ValueRange{c0_idx, c0_idx}, bL, ValueRange{c0_idx, c0_idx}, szB_idx, tagB_next, ValueRange{c0_idx});
                                    fb.create<memref::DmaWaitOp>(fl, tagA_next, ValueRange{c0_idx}, szA_idx);
                                    fb.create<memref::DmaWaitOp>(fl, tagB_next, ValueRange{c0_idx}, szB_idx);
                                    fb.create<scf::YieldOp>(fl);
                                });

                                lbK.create<scf::IfOp>(ll, is_k2_plus, [&](OpBuilder &ab, Location al) {
                                    Value c_acc = resolveBuf(ab, al, acc_id);
                                    Value cAccS = createSubview2D(ab, al, c_acc, c0_idx, c0_idx, mE_idx, nE_idx, cElemType, l1SpaceAttr);
                                    Value cPrevS = createSubview2D(ab, al, c_pong2, c0_idx, c0_idx, mE_idx, nE_idx, cElemType, l1SpaceAttr);
                                    ab.create<spatz::MatrixAddOp>(al, cAccS, cPrevS, mE_idx, nE_idx, builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
                                    ab.create<scf::YieldOp>(al);
                                });

                                lbK.create<func::CallOp>(ll, hwBarrierFn, ValueRange{});

                                Value next_acc_id = acc_id; Value next_p1_id = p2_id; Value next_p2_id = p1_id;

                                Value next_kOff = lbK.create<arith::AddIOp>(ll, kOff, tk_i32);
                                Value is_last_k = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::sge, next_kOff, dimK);

                                Value final_acc_id = lbK.create<arith::SelectOp>(ll, is_last_k, p2_id, next_acc_id);
                                Value final_p1_id  = lbK.create<arith::SelectOp>(ll, is_last_k, acc_id, next_p1_id);
                                Value final_p2_id  = lbK.create<arith::SelectOp>(ll, is_last_k, p1_id, next_p2_id);

                                lbK.create<scf::YieldOp>(ll, ValueRange{final_acc_id, final_p1_id, final_p2_id, nextFlatIdx});
                            });
                        lbN.create<scf::YieldOp>(lN, loopK.getResults());
                    });
                lbM.create<scf::YieldOp>(lM, loopN.getResults());
            });

        Value res_c_pong1 = resolveBuf(b0, l0, loopM.getResult(1));
        Value res_c_pong2 = resolveBuf(b0, l0, loopM.getResult(2));

        Value multiple_k = b0.create<arith::CmpIOp>(l0, arith::CmpIPredicate::sgt, itersK, c1_i32);
        Value m_last_iter = b0.create<arith::SubIOp>(l0, itersM, c1_i32);
        Value n_last_iter = b0.create<arith::SubIOp>(l0, itersN, c1_i32);
        Value m_lastOff = b0.create<arith::MulIOp>(l0, m_last_iter, tm_i32);
        Value n_lastOff = b0.create<arith::MulIOp>(l0, n_last_iter, tn_i32);
        Value mE_last = createMinI32(b0, l0, b0.create<arith::SubIOp>(l0, dimM, m_lastOff), tm_i32);
        Value nE_last = createMinI32(b0, l0, b0.create<arith::SubIOp>(l0, dimN, n_lastOff), tn_i32);

        Value m_lastOff_idx = b0.create<arith::IndexCastOp>(l0, indexTy, m_lastOff);
        Value n_lastOff_idx = b0.create<arith::IndexCastOp>(l0, indexTy, n_lastOff);
        Value mE_last_idx = b0.create<arith::IndexCastOp>(l0, indexTy, mE_last);
        Value nE_last_idx = b0.create<arith::IndexCastOp>(l0, indexTy, nE_last);

        b0.create<scf::IfOp>(l0, multiple_k, [&](OpBuilder &addB, Location addL) {
            Value cAccS = createSubview2D(addB, addL, res_c_pong1, c0_idx, c0_idx, mE_last_idx, nE_last_idx, cElemType, l1SpaceAttr);
            Value cPrevS = createSubview2D(addB, addL, res_c_pong2, c0_idx, c0_idx, mE_last_idx, nE_last_idx, cElemType, l1SpaceAttr);
            addB.create<spatz::MatrixAddOp>(addL, cAccS, cPrevS, mE_last_idx, nE_last_idx, builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
            addB.create<scf::YieldOp>(addL);
        });

        Value cOut = createSubview2D(b0, l0, c, m_lastOff_idx, n_lastOff_idx, mE_last_idx, nE_last_idx, cElemType, cType.getMemorySpace());
        Value cSrc = createSubview2D(b0, l0, res_c_pong1, c0_idx, c0_idx, mE_last_idx, nE_last_idx, cElemType, l1SpaceAttr);

        if (hasBias) {
            Value biasSub2D = createSubview2D(b0, l0, bias2D, c0_idx, n_lastOff_idx, c1_idx, nE_last_idx, cElemType, biasMemSpace);
            Value biasL1Sub2D = createSubview2D(b0, l0, biasL1_2D, c0_idx, c0_idx, c1_idx, nE_last_idx, cElemType, l1SpaceAttr);
            Value szBias_idx = b0.create<arith::IndexCastOp>(l0, indexTy, nE_last);

            b0.create<memref::DmaStartOp>(l0, biasSub2D, ValueRange{c0_idx, c0_idx}, biasL1Sub2D, ValueRange{c0_idx, c0_idx}, szBias_idx, tagBias, ValueRange{c0_idx});
            b0.create<memref::DmaWaitOp>(l0, tagBias, ValueRange{c0_idx}, szBias_idx);

            SmallVector<OpFoldResult, 1> offsets1D = {c0_idx};
            SmallVector<OpFoldResult, 1> sizes1D = {nE_last_idx};
            SmallVector<OpFoldResult, 1> strides1D = {b0.getIndexAttr(1)};
            Value biasL1Sub1D = b0.create<memref::SubViewOp>(l0, biasL1_1D, offsets1D, sizes1D, strides1D);

            b0.create<spatz::MatrixVectorAddOp>(l0, cSrc, biasL1Sub1D, mE_last_idx, nE_last_idx,
                builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
        }

        if (hasScale) {
            b0.create<spatz::MatrixScalarMulOp>(l0, cSrc, scaleVal, mE_last_idx, nE_last_idx,
                builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
        }

        Value cSz_i32 = b0.create<arith::MulIOp>(l0, mE_last, nE_last);
        Value cSz_idx = b0.create<arith::IndexCastOp>(l0, indexTy, cSz_i32);

        b0.create<memref::DmaStartOp>(l0, cSrc, ValueRange{c0_idx, c0_idx}, cOut, ValueRange{c0_idx, c0_idx}, cSz_idx, tagC, ValueRange{c0_idx});
        b0.create<memref::DmaWaitOp>(l0, tagC, ValueRange{c0_idx}, cSz_idx);

        b0.create<func::CallOp>(l0, l1ResetFn, ValueRange{});
        b0.create<scf::YieldOp>(l0);
    });

    builder.create<scf::IfOp>(loc, isCore1, [&](OpBuilder &b1, Location l1) {

        b1.create<scf::ForOp>(l1, c0_idx, dimM_idx, tm_idx, loopArgs,
            [&](OpBuilder &lbM, Location lM, Value m_idx, ValueRange argsM) {

                Value mOff = lbM.create<arith::IndexCastOp>(lM, i32Ty, m_idx);
                Value mE = createMinI32(lbM, lM, lbM.create<arith::SubIOp>(lM, dimM, mOff), tm_i32);
                Value mE_idx = lbM.create<arith::IndexCastOp>(lM, indexTy, mE);

                auto loopN = lbM.create<scf::ForOp>(lM, c0_idx, dimN_idx, tn_idx, argsM,
                    [&](OpBuilder &lbN, Location lN, Value n_idx, ValueRange argsN) {

                        Value nOff = lbN.create<arith::IndexCastOp>(lN, i32Ty, n_idx);
                        Value nE = createMinI32(lbN, lN, lbN.create<arith::SubIOp>(lN, dimN, nOff), tn_i32);
                        Value nE_idx = lbN.create<arith::IndexCastOp>(lN, indexTy, nE);

                        auto loopK = lbN.create<scf::ForOp>(lN, c0_idx, dimK_idx, tk2_idx, argsN,
                            [&](OpBuilder &lbK, Location ll, Value k_idx, ValueRange argsK) {

                                Value acc_id = argsK[0]; Value p1_id = argsK[1]; Value p2_id = argsK[2];
                                Value flatIdx = argsK[3];

                                Value kOff = lbK.create<arith::IndexCastOp>(ll, i32Ty, k_idx);

                                Value parity = lbK.create<arith::RemUIOp>(ll, flatIdx, c2_i32);
                                Value is_even = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::eq, parity, c0_i32);

                                Value aL1_curr = lbK.create<arith::SelectOp>(ll, is_even, aL1_0, aL1_1);
                                Value bL1_curr = lbK.create<arith::SelectOp>(ll, is_even, bL1_0, bL1_1);
                                Value aL1_next = lbK.create<arith::SelectOp>(ll, is_even, aL1_1, aL1_0);
                                Value bL1_next = lbK.create<arith::SelectOp>(ll, is_even, bL1_1, bL1_0);

                                Value c_acc_0 = resolveBuf(lbK, ll, acc_id);
                                Value c_pong1_0 = resolveBuf(lbK, ll, p1_id);

                                Value kE_0 = createMinI32(lbK, ll, lbK.create<arith::SubIOp>(ll, dimK, kOff), tk_i32);
                                Value kE_idx_0 = lbK.create<arith::IndexCastOp>(ll, indexTy, kE_0);

                                Value is_k0_0 = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::eq, kOff, c0_i32);

                                Value aSub_0 = createSubview2D(lbK, ll, aL1_curr, c0_idx, c0_idx, kE_idx_0, mE_idx, aElemType, l1SpaceAttr);
                                Value bSub_0 = createSubview2D(lbK, ll, bL1_curr, c0_idx, c0_idx, kE_idx_0, nE_idx, bElemType, l1SpaceAttr);

                                Value cTarget_0 = lbK.create<arith::SelectOp>(ll, is_k0_0, c_acc_0, c_pong1_0);
                                Value cSub_0 = createSubview2D(lbK, ll, cTarget_0, c0_idx, c0_idx, mE_idx, nE_idx, cElemType, l1SpaceAttr);

                                lbK.create<quadrilatero::TcdmMatmulMemRefOp>(ll, aSub_0, bSub_0, cSub_0, mE_idx, nE_idx, kE_idx_0,
                                    lbK.create<arith::ConstantIndexOp>(ll, shiftVal),
                                    builder.getI32IntegerAttr(dtC_val), builder.getI32IntegerAttr(dtA_val), builder.getI32IntegerAttr(dtB_val));

                                lbK.create<func::CallOp>(ll, hwBarrierFn, ValueRange{});

                                Value nextFlatIdx_0 = lbK.create<arith::AddIOp>(ll, flatIdx, c1_i32);
                                Value next_acc_id_0 = acc_id; Value next_p1_id_0 = p2_id; Value next_p2_id_0 = p1_id;

                                Value next_kOff_0 = lbK.create<arith::AddIOp>(ll, kOff, tk_i32);
                                Value is_last_k_0 = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::sge, next_kOff_0, dimK);

                                Value final_acc_id_0 = lbK.create<arith::SelectOp>(ll, is_last_k_0, p2_id, next_acc_id_0);
                                Value final_p1_id_0  = lbK.create<arith::SelectOp>(ll, is_last_k_0, acc_id, next_p1_id_0);
                                Value final_p2_id_0  = lbK.create<arith::SelectOp>(ll, is_last_k_0, p1_id, next_p2_id_0);

                                Value has_odd_tile = lbK.create<arith::CmpIOp>(ll, arith::CmpIPredicate::slt, next_kOff_0, dimK);

                                SmallVector<Type, 4> ifTypes = {i32Ty, i32Ty, i32Ty, i32Ty};
                                auto oddIf = lbK.create<scf::IfOp>(ll, ifTypes, has_odd_tile,
                                    [&](OpBuilder &ob, Location ol) {
                                        Value c_pong1_1 = resolveBuf(ob, ol, final_p1_id_0);

                                        Value kE_1 = createMinI32(ob, ol, ob.create<arith::SubIOp>(ol, dimK, next_kOff_0), tk_i32);
                                        Value kE_idx_1 = ob.create<arith::IndexCastOp>(ol, indexTy, kE_1);

                                        Value aSub_1 = createSubview2D(ob, ol, aL1_next, c0_idx, c0_idx, kE_idx_1, mE_idx, aElemType, l1SpaceAttr);
                                        Value bSub_1 = createSubview2D(ob, ol, bL1_next, c0_idx, c0_idx, kE_idx_1, nE_idx, bElemType, l1SpaceAttr);

                                        Value cSub_1 = createSubview2D(ob, ol, c_pong1_1, c0_idx, c0_idx, mE_idx, nE_idx, cElemType, l1SpaceAttr);

                                        ob.create<quadrilatero::TcdmMatmulMemRefOp>(ol, aSub_1, bSub_1, cSub_1, mE_idx, nE_idx, kE_idx_1,
                                            ob.create<arith::ConstantIndexOp>(ol, shiftVal),
                                            builder.getI32IntegerAttr(dtC_val), builder.getI32IntegerAttr(dtA_val), builder.getI32IntegerAttr(dtB_val));

                                        ob.create<func::CallOp>(ol, hwBarrierFn, ValueRange{});

                                        Value nextFlatIdx_1 = ob.create<arith::AddIOp>(ol, nextFlatIdx_0, c1_i32);
                                        Value next_acc_id_1 = final_acc_id_0; Value next_p1_id_1 = final_p2_id_0; Value next_p2_id_1 = final_p1_id_0;

                                        Value next_kOff_1 = ob.create<arith::AddIOp>(ol, next_kOff_0, tk_i32);
                                        Value is_last_k_1 = ob.create<arith::CmpIOp>(ol, arith::CmpIPredicate::sge, next_kOff_1, dimK);

                                        Value final_acc_id_1 = ob.create<arith::SelectOp>(ol, is_last_k_1, final_p2_id_0, next_acc_id_1);
                                        Value final_p1_id_1  = ob.create<arith::SelectOp>(ol, is_last_k_1, final_acc_id_0, next_p1_id_1);
                                        Value final_p2_id_1  = ob.create<arith::SelectOp>(ol, is_last_k_1, final_p1_id_0, next_p2_id_1);

                                        ob.create<scf::YieldOp>(ol, ValueRange{final_acc_id_1, final_p1_id_1, final_p2_id_1, nextFlatIdx_1});
                                    },
                                    [&](OpBuilder &ob, Location ol) {
                                        ob.create<scf::YieldOp>(ol, ValueRange{final_acc_id_0, final_p1_id_0, final_p2_id_0, nextFlatIdx_0});
                                    }
                                );

                                lbK.create<scf::YieldOp>(ll, oddIf.getResults());
                            });
                        lbN.create<scf::YieldOp>(lN, loopK.getResults());
                    });
                lbM.create<scf::YieldOp>(lM, loopN.getResults());
            });

        b1.create<scf::YieldOp>(l1);
    });

    builder.create<func::CallOp>(loc, hwBarrierFn, ValueRange{});

    if (scaleOpToErase) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(scaleOpToErase)) {
        if (linalgOp.getNumOutputs() == 1) {
          Value genericOut = linalgOp.getOutputOperand(0)->get();
          if (genericOut != mainC) { genericOut.replaceAllUsesWith(mainC); }
        }
      }
      scaleOpToErase->erase();
    }

    if (addOpToErase) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(addOpToErase)) {
        if (linalgOp.getNumOutputs() == 1) {
          Value genericOut = linalgOp.getOutputOperand(0)->get();
          if (genericOut != mainC) { genericOut.replaceAllUsesWith(mainC); }
        }
      }
      addOpToErase->erase();
    }

    op.erase();
    return success();
  }
};
}

namespace mlir {
std::unique_ptr<Pass> createLowerLinalgMatmulToQuadrilateroPass() {
  return std::make_unique<LowerLinalgMatmulToQuadrilateroPass>();
}
}
