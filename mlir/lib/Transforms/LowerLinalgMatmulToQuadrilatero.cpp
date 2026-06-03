//===- LowerLinalgMatmulToQuadrilatero.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
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

static Value createSubview2D(OpBuilder &builder, Location loc, Value base,
                             Value off0, Value off1, Value size0, Value size1,
                             Type elementType, Attribute memorySpace) {
  SmallVector<OpFoldResult, 2> offsets = {off0, off1};
  SmallVector<OpFoldResult, 2> sizes = {size0, size1};
  SmallVector<OpFoldResult, 2> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
  return builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides).getResult();
}

struct LowerLinalgMatmulToQuadrilateroPass
    : public LowerLinalgMatmulToQuadrilateroBase<
          LowerLinalgMatmulToQuadrilateroPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<linalg::MatmulOp, 4> matmuls;
    funcOp.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    for (linalg::MatmulOp op : matmuls)
      if (failed(lowerMatmul(op))) signalPassFailure();
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

    op.erase();
    return success();
  }
};
} // namespace

namespace mlir {
std::unique_ptr<Pass> createLowerLinalgMatmulToQuadrilateroPass() {
  return std::make_unique<LowerLinalgMatmulToQuadrilateroPass>();
}
} // namespace mlir