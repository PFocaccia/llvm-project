//===- LowerLinalgMatmulToQuadrilatero.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quadrilatero/QuadrilateroDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Spatz/SpatzDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/SymbolTable.h"  

using namespace mlir;

namespace {

static int32_t getDataTypeCode(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    bool isUnsigned = intType.isUnsigned();
    if (width == 8)  return isUnsigned ? 8 : 0;   // UINT8  : INT8
    if (width == 16) return isUnsigned ? 9 : 1;   // UINT16 : INT16
    if (width == 32) return isUnsigned ? 10 : 2;  // UINT32 : INT32
  } else if (type.isF32()) {
    return 6;  // FP32
  } else if (type.isF16()) {
    return 5;  // FP16
  } else if (type.isBF16()) {
    return 13; // BFP16
  } 
  return -1; 
}

static Value createMinIndex(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
  return builder.create<arith::SelectOp>(loc, cmp, lhs, rhs);
}

static Value createSubview2D(OpBuilder &builder, Location loc, Value base, Value off0, Value off1, Value size0, Value size1,
                             Type elementType, Attribute memorySpace) {
  SmallVector<OpFoldResult, 2> offsets = {off0, off1};
  SmallVector<OpFoldResult, 2> sizes   = {size0, size1};
  SmallVector<OpFoldResult, 2> strides = {builder.getIndexAttr(1),builder.getIndexAttr(1)};
  (void)elementType;
  (void)memorySpace;

  return builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides).getResult();
}

struct LowerLinalgMatmulToQuadrilateroPass : public LowerLinalgMatmulToQuadrilateroBase<LowerLinalgMatmulToQuadrilateroPass> {
  
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<linalg::MatmulOp, 4> matmuls;
    
    funcOp.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    
    for (linalg::MatmulOp op : matmuls) if (failed(lowerMatmul(op))) signalPassFailure();
  }

  LogicalResult lowerMatmul(linalg::MatmulOp op) {
    if (op.getNumInputs() != 2 || op.getNumOutputs() != 1) return failure();
    
    Value a = op.inputs()[0]; Value b = op.inputs()[1]; Value c = op.outputs()[0];
    auto aType = a.getType().dyn_cast<MemRefType>();
    auto bType = b.getType().dyn_cast<MemRefType>();
    auto cType = c.getType().dyn_cast<MemRefType>();
    
    if (!aType || !bType || !cType) return failure();

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
    int64_t tileMVal = 64; int64_t tileNVal = 64;

    int64_t shiftVal = 0;
    if (bitWidth == 32) shiftVal = 0;
    else if (bitWidth == 16) shiftVal = 1;
    else if (bitWidth == 8) shiftVal = 2;

    OpBuilder builder(op);
    Location loc = op.getLoc();

    auto aShape = aType.getShape();
    SmallVector<int64_t, 2> transShape = {aShape[1], aShape[0]};
    auto aTransType = MemRefType::get(transShape, aElemType, MemRefLayoutAttrInterface{}, aType.getMemorySpace());
    
    Value aTrans = builder.create<memref::AllocOp>(loc, aTransType);
    
    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto mapInput = AffineMap::get(2, 0, {d0, d1}, builder.getContext());
    auto mapOutput = AffineMap::get(2, 0, {d1, d0}, builder.getContext());
    SmallVector<AffineMap, 2> indexingMaps = {mapInput, mapOutput};
    
    SmallVector<StringRef, 2> iteratorTypes = {"parallel", "parallel"};


    builder.create<linalg::GenericOp>(
        loc,
        ValueRange{a},
        ValueRange{aTrans},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        }
    );

    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = builder.getI32Type();

    SymbolTable symbolTable(module);
    
    auto getCoreIdxFn = symbolTable.lookup<func::FuncOp>("snrt_cluster_core_idx");
    if (!getCoreIdxFn) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      getCoreIdxFn = builder.create<func::FuncOp>(loc, "snrt_cluster_core_idx", builder.getFunctionType({}, {i32Ty}));
      getCoreIdxFn.setPrivate();
      symbolTable.insert(getCoreIdxFn);
    }

    auto hwBarrierFn = symbolTable.lookup<func::FuncOp>("snrt_cluster_hw_barrier");
    if (!hwBarrierFn) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      hwBarrierFn = builder.create<func::FuncOp>(loc, "snrt_cluster_hw_barrier", builder.getFunctionType({}, {}));
      hwBarrierFn.setPrivate();
      symbolTable.insert(hwBarrierFn);
    }
    
    auto l1ResetFn = symbolTable.lookup<func::FuncOp>("snrt_l1alloc_reset");
    if (!l1ResetFn) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      l1ResetFn = builder.create<func::FuncOp>(loc, "snrt_l1alloc_reset", builder.getFunctionType({}, {}));
      l1ResetFn.setPrivate();
      symbolTable.insert(l1ResetFn);
    }

    Value cid = builder.create<func::CallOp>(loc, getCoreIdxFn, ValueRange{}).getResult(0);
    Value cid0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value cid1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    Value isCore0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, cid0); 
    Value isCore1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, cid1); 

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
    Value tagA_0 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagA_1 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagB_0 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagB_1 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagC_0 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagC_1 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagC_2 = builder.create<memref::AllocaOp>(loc, tagType);

    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c64 = builder.create<arith::ConstantIndexOp>(loc, 64);
    Value tk = builder.create<arith::ConstantIndexOp>(loc, tileKVal);

    Value dimK = builder.create<memref::DimOp>(loc, aTrans, 0);
    Value dimM = builder.create<memref::DimOp>(loc, aTrans, 1);
    Value dimN = builder.create<memref::DimOp>(loc, b, 1);
    
    Value true_val = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    Value false_val = builder.create<arith::ConstantIntOp>(loc, 0, 1);

    Value mE_0 = createMinIndex(builder, loc, dimM, c64);
    Value nE_0 = createMinIndex(builder, loc, dimN, c64);
    Value kE_0 = createMinIndex(builder, loc, dimK, tk);
    
    Value k1_global = builder.create<arith::ConstantIndexOp>(loc, tileKVal);
    Value has_k1_global = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, k1_global, dimK);

    builder.create<scf::IfOp>(loc, isCore0, [&](OpBuilder &ifB, Location ifL) {
      Value aS_0 = createSubview2D(ifB, ifL, aTrans, c0, c0, kE_0, mE_0, aElemType, aType.getMemorySpace());
      Value bS_0 = createSubview2D(ifB, ifL, b, c0, c0, kE_0, nE_0, bElemType, bType.getMemorySpace());
      Value aL_0 = createSubview2D(ifB, ifL, aL1_0, c0, c0, kE_0, mE_0, aElemType, l1SpaceAttr);
      Value bL_0 = createSubview2D(ifB, ifL, bL1_0, c0, c0, kE_0, nE_0, bElemType, l1SpaceAttr);
      ifB.create<memref::DmaStartOp>(ifL, aS_0, ValueRange{c0,c0}, aL_0, ValueRange{c0,c0}, ifB.create<arith::MulIOp>(ifL, kE_0, mE_0), tagA_0, ValueRange{c0});
      ifB.create<memref::DmaStartOp>(ifL, bS_0, ValueRange{c0,c0}, bL_0, ValueRange{c0,c0}, ifB.create<arith::MulIOp>(ifL, kE_0, nE_0), tagB_0, ValueRange{c0});
      
      ifB.create<scf::IfOp>(ifL, has_k1_global, [&](OpBuilder &innerB, Location innerL) {
         Value kE_1 = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimK, k1_global), tk);
         Value aS_1 = createSubview2D(innerB, innerL, aTrans, k1_global, c0, kE_1, mE_0, aElemType, aType.getMemorySpace());
         Value bS_1 = createSubview2D(innerB, innerL, b, k1_global, c0, kE_1, nE_0, bElemType, bType.getMemorySpace());
         Value aL_1 = createSubview2D(innerB, innerL, aL1_1, c0, c0, kE_1, mE_0, aElemType, l1SpaceAttr);
         Value bL_1 = createSubview2D(innerB, innerL, bL1_1, c0, c0, kE_1, nE_0, bElemType, l1SpaceAttr);
         innerB.create<memref::DmaStartOp>(innerL, aS_1, ValueRange{c0,c0}, aL_1, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, kE_1, mE_0), tagA_1, ValueRange{c0});
         innerB.create<memref::DmaStartOp>(innerL, bS_1, ValueRange{c0,c0}, bL_1, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, kE_1, nE_0), tagB_1, ValueRange{c0});
         innerB.create<scf::YieldOp>(innerL);
      });

      ifB.create<memref::DmaWaitOp>(ifL, tagA_0, ValueRange{c0}, ifB.create<arith::MulIOp>(ifL, kE_0, mE_0));
      ifB.create<memref::DmaWaitOp>(ifL, tagB_0, ValueRange{c0}, ifB.create<arith::MulIOp>(ifL, kE_0, nE_0));
      ifB.create<scf::YieldOp>(ifL);
    });
    builder.create<func::CallOp>(loc, hwBarrierFn, ValueRange{});

    SmallVector<Value, 18> mnArgs = {
        aL1_0, aL1_1, bL1_0, bL1_1, 
        tagA_0, tagA_1, tagB_0, tagB_1, 
        cBuf_0, cBuf_1, cBuf_2, 
        tagC_0, tagC_1, tagC_2, 
        false_val, c0, c0, true_val
    };

    auto loopM = builder.create<scf::ForOp>(loc, c0, dimM, c64, mnArgs, [&](OpBuilder &mB, Location mL, Value i, ValueRange mArgs) {
      auto loopN = mB.create<scf::ForOp>(mL, c0, dimN, c64, mArgs, [&](OpBuilder &nB, Location nL, Value j, ValueRange nArgs) {
        
        Value a_r = nArgs[0]; Value a_f = nArgs[1]; Value b_r = nArgs[2]; Value b_f = nArgs[3];
        Value tA_r = nArgs[4]; Value tA_f = nArgs[5]; Value tB_r = nArgs[6]; Value tB_f = nArgs[7];
        Value c_wb = nArgs[8]; Value c_acc = nArgs[9]; Value c_spare = nArgs[10];
        Value tC_wb = nArgs[11]; Value tC_acc = nArgs[12]; Value tC_spare = nArgs[13];
        Value prev_valid = nArgs[14]; Value prev_i = nArgs[15]; Value prev_j = nArgs[16];
        Value is_first_mn = nArgs[17]; 

        Value mE = createMinIndex(nB, nL, nB.create<arith::SubIOp>(nL, dimM, i), c64);
        Value nE = createMinIndex(nB, nL, nB.create<arith::SubIOp>(nL, dimN, j), c64);
        Value k0_eff = createMinIndex(nB, nL, dimK, tk);
        
        Value next_j = nB.create<arith::AddIOp>(nL, j, c64);
        Value has_next_j = nB.create<arith::CmpIOp>(nL, arith::CmpIPredicate::slt, next_j, dimN);
        Value next_i = nB.create<arith::AddIOp>(nL, i, c64);
        Value has_next_i = nB.create<arith::CmpIOp>(nL, arith::CmpIPredicate::slt, next_i, dimM);
        Value has_next_mn = nB.create<arith::OrIOp>(nL, has_next_j, has_next_i);

        Value fetch_i = nB.create<arith::SelectOp>(nL, has_next_j, i, next_i);
        Value fetch_j = nB.create<arith::SelectOp>(nL, has_next_j, next_j, c0);
        Value fetch_mE = createMinIndex(nB, nL, nB.create<arith::SubIOp>(nL, dimM, fetch_i), c64);
        Value fetch_nE = createMinIndex(nB, nL, nB.create<arith::SubIOp>(nL, dimN, fetch_j), c64);
        Value fetch_k0_eff = createMinIndex(nB, nL, dimK, tk);

        Value k1 = nB.create<arith::ConstantIndexOp>(nL, tileKVal);
        Value has_k1 = nB.create<arith::CmpIOp>(nL, arith::CmpIPredicate::slt, k1, dimK);

        nB.create<scf::IfOp>(nL, isCore0, [&](OpBuilder &ifB, Location ifL) {
          ifB.create<scf::IfOp>(ifL, prev_valid, [&](OpBuilder &innerB, Location innerL) {
             Value prev_mE = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimM, prev_i), c64);
             Value prev_nE = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimN, prev_j), c64);
             Value cSubOrig = createSubview2D(innerB, innerL, c, prev_i, prev_j, prev_mE, prev_nE, cElemType, cType.getMemorySpace());
             Value cWbS = createSubview2D(innerB, innerL, c_wb, c0, c0, prev_mE, prev_nE, cElemType, l1SpaceAttr);
             innerB.create<memref::DmaStartOp>(innerL, cWbS, ValueRange{c0,c0}, cSubOrig, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, prev_mE, prev_nE), tC_wb, ValueRange{c0});
             innerB.create<scf::YieldOp>(innerL);
          });

          Value not_first = ifB.create<arith::CmpIOp>(ifL, arith::CmpIPredicate::eq, is_first_mn, false_val);
          Value do_k1_fetch = ifB.create<arith::AndIOp>(ifL, not_first, has_k1);

          ifB.create<scf::IfOp>(ifL, do_k1_fetch, [&](OpBuilder &innerB, Location innerL) {
             Value k1_eff = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimK, k1), tk);
             Value aS1 = createSubview2D(innerB, innerL, aTrans, k1, i, k1_eff, mE, aElemType, aType.getMemorySpace());
             Value bS1 = createSubview2D(innerB, innerL, b, k1, j, k1_eff, nE, bElemType, bType.getMemorySpace());
             Value aL1 = createSubview2D(innerB, innerL, a_f, c0, c0, k1_eff, mE, aElemType, l1SpaceAttr);
             Value bL1 = createSubview2D(innerB, innerL, b_f, c0, c0, k1_eff, nE, bElemType, l1SpaceAttr);
             innerB.create<memref::DmaStartOp>(innerL, aS1, ValueRange{c0,c0}, aL1, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, k1_eff, mE), tA_f, ValueRange{c0});
             innerB.create<memref::DmaStartOp>(innerL, bS1, ValueRange{c0,c0}, bL1, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, k1_eff, nE), tB_f, ValueRange{c0});
             innerB.create<scf::YieldOp>(innerL);
          });

          Value do_next_mn = ifB.create<arith::AndIOp>(ifL, 
                ifB.create<arith::CmpIOp>(ifL, arith::CmpIPredicate::eq, has_k1, false_val), has_next_mn);

          ifB.create<scf::IfOp>(ifL, do_next_mn, [&](OpBuilder &innerB, Location innerL) {
             Value aS1 = createSubview2D(innerB, innerL, aTrans, c0, fetch_i, fetch_k0_eff, fetch_mE, aElemType, aType.getMemorySpace());
             Value bS1 = createSubview2D(innerB, innerL, b, c0, fetch_j, fetch_k0_eff, fetch_nE, bElemType, bType.getMemorySpace());
             Value aL1 = createSubview2D(innerB, innerL, a_f, c0, c0, fetch_k0_eff, fetch_mE, aElemType, l1SpaceAttr);
             Value bL1 = createSubview2D(innerB, innerL, b_f, c0, c0, fetch_k0_eff, fetch_nE, bElemType, l1SpaceAttr);
             innerB.create<memref::DmaStartOp>(innerL, aS1, ValueRange{c0,c0}, aL1, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, fetch_k0_eff, fetch_mE), tA_f, ValueRange{c0});
             innerB.create<memref::DmaStartOp>(innerL, bS1, ValueRange{c0,c0}, bL1, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, fetch_k0_eff, fetch_nE), tB_f, ValueRange{c0});
             innerB.create<scf::YieldOp>(innerL);
          });
          ifB.create<scf::YieldOp>(ifL);
        });

        nB.create<scf::IfOp>(nL, isCore1, [&](OpBuilder &ifB, Location ifL) {
            Value aL0 = createSubview2D(ifB, ifL, a_r, c0, c0, k0_eff, mE, aElemType, l1SpaceAttr);
            Value bL0 = createSubview2D(ifB, ifL, b_r, c0, c0, k0_eff, nE, bElemType, l1SpaceAttr);
            Value cAccS = createSubview2D(ifB, ifL, c_acc, c0, c0, mE, nE, cElemType, l1SpaceAttr);
            ifB.create<quadrilatero::TcdmMatmulMemRefOp>(ifL, aL0, bL0, cAccS, mE, nE, k0_eff, 
                ifB.create<arith::ConstantIndexOp>(ifL, shiftVal), builder.getI32IntegerAttr(dtC_val), builder.getI32IntegerAttr(dtA_val), builder.getI32IntegerAttr(dtB_val));
            ifB.create<scf::YieldOp>(ifL);
        });

        nB.create<func::CallOp>(nL, hwBarrierFn, ValueRange{});

        nB.create<scf::IfOp>(nL, isCore0, [&](OpBuilder &ifB, Location ifL) {
            Value do_next_mn = ifB.create<arith::AndIOp>(ifL, 
                ifB.create<arith::CmpIOp>(ifL, arith::CmpIPredicate::eq, has_k1, false_val), has_next_mn);
            Value wait_k1 = ifB.create<arith::OrIOp>(ifL, has_k1, do_next_mn);
            
            ifB.create<scf::IfOp>(ifL, wait_k1, [&](OpBuilder &innerB, Location innerL) {
               Value wait_k_eff = innerB.create<arith::SelectOp>(innerL, has_k1, 
                    createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimK, k1), tk), fetch_k0_eff);
               Value wait_mE = innerB.create<arith::SelectOp>(innerL, has_k1, mE, fetch_mE);
               Value wait_nE = innerB.create<arith::SelectOp>(innerL, has_k1, nE, fetch_nE);
               innerB.create<memref::DmaWaitOp>(innerL, tA_f, ValueRange{c0}, innerB.create<arith::MulIOp>(innerL, wait_k_eff, wait_mE));
               innerB.create<memref::DmaWaitOp>(innerL, tB_f, ValueRange{c0}, innerB.create<arith::MulIOp>(innerL, wait_k_eff, wait_nE));
               innerB.create<scf::YieldOp>(innerL);
            });
            ifB.create<scf::YieldOp>(ifL);
        });
        nB.create<func::CallOp>(nL, hwBarrierFn, ValueRange{});

        SmallVector<Value, 13> kArgs = {a_f, a_r, b_f, b_r, tA_f, tA_r, tB_f, tB_r, c_acc, c_spare, c_wb, false_val, prev_valid};
        
        auto loopK = nB.create<scf::ForOp>(nL, k1, dimK, tk, kArgs, [&](OpBuilder &kB, Location kL, Value k, ValueRange kR) {
           Value a_c = kR[0]; Value a_f_nxt = kR[1]; Value b_c = kR[2]; Value b_f_nxt = kR[3];
           Value tA_c = kR[4]; Value tA_f_nxt = kR[5]; Value tB_c = kR[6]; Value tB_f_nxt = kR[7];
           Value c_acc_L = kR[8]; Value c_free_L = kR[9]; Value c_last_L = kR[10]; Value needs_add = kR[11];
           Value wait_c_wb_L = kR[12]; 

           Value k_eff = createMinIndex(kB, kL, kB.create<arith::SubIOp>(kL, dimK, k), tk);
           Value next_k = kB.create<arith::AddIOp>(kL, k, tk);
           Value has_next_k = kB.create<arith::CmpIOp>(kL, arith::CmpIPredicate::slt, next_k, dimK);
           Value do_next_mn_k = kB.create<arith::AndIOp>(kL, kB.create<arith::CmpIOp>(kL, arith::CmpIPredicate::eq, has_next_k, false_val), has_next_mn);

           kB.create<scf::IfOp>(kL, isCore0, [&](OpBuilder &ifB, Location ifL) {

               ifB.create<scf::IfOp>(ifL, needs_add, [&](OpBuilder &innerB, Location innerL) {
                   Value cAccS = createSubview2D(innerB, innerL, c_acc_L, c0, c0, mE, nE, cElemType, l1SpaceAttr);
                   Value cTmpS = createSubview2D(innerB, innerL, c_last_L, c0, c0, mE, nE, cElemType, l1SpaceAttr);
                   innerB.create<spatz::MatrixAddOp>(innerL, cAccS, cTmpS, mE, nE, builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
                   innerB.create<scf::YieldOp>(innerL);
               });

               ifB.create<scf::IfOp>(ifL, wait_c_wb_L, [&](OpBuilder &innerB, Location innerL) {
                   Value prev_mE = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimM, prev_i), c64);
                   Value prev_nE = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimN, prev_j), c64);
                   innerB.create<memref::DmaWaitOp>(innerL, tC_wb, ValueRange{c0}, innerB.create<arith::MulIOp>(innerL, prev_mE, prev_nE));
                   innerB.create<scf::YieldOp>(innerL);
               });

               ifB.create<scf::IfOp>(ifL, has_next_k, [&](OpBuilder &innerB, Location innerL) {
                   Value nk_eff = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimK, next_k), tk);
                   Value aSn = createSubview2D(innerB, innerL, aTrans, next_k, i, nk_eff, mE, aElemType, aType.getMemorySpace());
                   Value bSn = createSubview2D(innerB, innerL, b, next_k, j, nk_eff, nE, bElemType, bType.getMemorySpace());
                   Value aLn = createSubview2D(innerB, innerL, a_f_nxt, c0, c0, nk_eff, mE, aElemType, l1SpaceAttr);
                   Value bLn = createSubview2D(innerB, innerL, b_f_nxt, c0, c0, nk_eff, nE, bElemType, l1SpaceAttr);
                   innerB.create<memref::DmaStartOp>(innerL, aSn, ValueRange{c0,c0}, aLn, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, nk_eff, mE), tA_f_nxt, ValueRange{c0});
                   innerB.create<memref::DmaStartOp>(innerL, bSn, ValueRange{c0,c0}, bLn, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, nk_eff, nE), tB_f_nxt, ValueRange{c0});
                   innerB.create<scf::YieldOp>(innerL);
               });

               ifB.create<scf::IfOp>(ifL, do_next_mn_k, [&](OpBuilder &innerB, Location innerL) {
                   Value aS1 = createSubview2D(innerB, innerL, aTrans, c0, fetch_i, fetch_k0_eff, fetch_mE, aElemType, aType.getMemorySpace());
                   Value bS1 = createSubview2D(innerB, innerL, b, c0, fetch_j, fetch_k0_eff, fetch_nE, bElemType, bType.getMemorySpace());
                   Value aL1 = createSubview2D(innerB, innerL, a_f_nxt, c0, c0, fetch_k0_eff, fetch_mE, aElemType, l1SpaceAttr);
                   Value bL1 = createSubview2D(innerB, innerL, b_f_nxt, c0, c0, fetch_k0_eff, fetch_nE, bElemType, l1SpaceAttr);
                   innerB.create<memref::DmaStartOp>(innerL, aS1, ValueRange{c0,c0}, aL1, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, fetch_k0_eff, fetch_mE), tA_f_nxt, ValueRange{c0});
                   innerB.create<memref::DmaStartOp>(innerL, bS1, ValueRange{c0,c0}, bL1, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, fetch_k0_eff, fetch_nE), tB_f_nxt, ValueRange{c0});
                   innerB.create<scf::YieldOp>(innerL);
               });
               ifB.create<scf::YieldOp>(ifL);
           });

           kB.create<scf::IfOp>(kL, isCore1, [&](OpBuilder &ifB, Location ifL) {
               Value aSubC = createSubview2D(ifB, ifL, a_c, c0, c0, k_eff, mE, aElemType, l1SpaceAttr);
               Value bSubC = createSubview2D(ifB, ifL, b_c, c0, c0, k_eff, nE, bElemType, l1SpaceAttr);
               Value cTmpS = createSubview2D(ifB, ifL, c_free_L, c0, c0, mE, nE, cElemType, l1SpaceAttr);
               ifB.create<quadrilatero::TcdmMatmulMemRefOp>(ifL, aSubC, bSubC, cTmpS, mE, nE, k_eff, 
                   ifB.create<arith::ConstantIndexOp>(ifL, shiftVal), builder.getI32IntegerAttr(dtC_val), builder.getI32IntegerAttr(dtA_val), builder.getI32IntegerAttr(dtB_val));
               ifB.create<scf::YieldOp>(ifL);
           });

           kB.create<func::CallOp>(kL, hwBarrierFn, ValueRange{});

           kB.create<scf::IfOp>(kL, isCore0, [&](OpBuilder &ifB, Location ifL) {
               Value wait_k = ifB.create<arith::OrIOp>(ifL, has_next_k, do_next_mn_k);
               ifB.create<scf::IfOp>(ifL, wait_k, [&](OpBuilder &innerB, Location innerL) {
                   Value wait_k_eff = innerB.create<arith::SelectOp>(innerL, has_next_k, 
                        createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimK, next_k), tk), fetch_k0_eff);
                   Value wait_mE = innerB.create<arith::SelectOp>(innerL, has_next_k, mE, fetch_mE);
                   Value wait_nE = innerB.create<arith::SelectOp>(innerL, has_next_k, nE, fetch_nE);
                   innerB.create<memref::DmaWaitOp>(innerL, tA_f_nxt, ValueRange{c0}, innerB.create<arith::MulIOp>(innerL, wait_k_eff, wait_mE));
                   innerB.create<memref::DmaWaitOp>(innerL, tB_f_nxt, ValueRange{c0}, innerB.create<arith::MulIOp>(innerL, wait_k_eff, wait_nE));
                   innerB.create<scf::YieldOp>(innerL);
               });
               ifB.create<scf::YieldOp>(ifL);
           });
           kB.create<func::CallOp>(kL, hwBarrierFn, ValueRange{});

           kB.create<scf::YieldOp>(kL, ValueRange{
               a_f_nxt, a_c, b_f_nxt, b_c, 
               tA_f_nxt, tA_c, tB_f_nxt, tB_c, 
               c_acc_L, c_last_L, c_free_L, 
               true_val, false_val
           });
        });

        Value k_a_c = loopK.getResult(0); Value k_a_f = loopK.getResult(1);
        Value k_b_c = loopK.getResult(2); Value k_b_f = loopK.getResult(3);
        Value k_tA_c = loopK.getResult(4); Value k_tA_f = loopK.getResult(5);
        Value k_tB_c = loopK.getResult(6); Value k_tB_f = loopK.getResult(7);
        Value k_c_acc = loopK.getResult(8); Value k_c_free = loopK.getResult(9); Value k_c_last = loopK.getResult(10);
        Value k_needs_add = loopK.getResult(11);
        Value wait_c_wb_final = loopK.getResult(12);

        nB.create<scf::IfOp>(nL, isCore0, [&](OpBuilder &ifB, Location ifL) {
            ifB.create<scf::IfOp>(ifL, k_needs_add, [&](OpBuilder &innerB, Location innerL) {
               Value cAccS = createSubview2D(innerB, innerL, k_c_acc, c0, c0, mE, nE, cElemType, l1SpaceAttr);
               Value cTmpS = createSubview2D(innerB, innerL, k_c_last, c0, c0, mE, nE, cElemType, l1SpaceAttr);
               innerB.create<spatz::MatrixAddOp>(innerL, cAccS, cTmpS, mE, nE, builder.getI64IntegerAttr(64), builder.getI32IntegerAttr(dtC_val));
               innerB.create<scf::YieldOp>(innerL);
            });
            
            ifB.create<scf::IfOp>(ifL, wait_c_wb_final, [&](OpBuilder &innerB, Location innerL) {
               Value prev_mE = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimM, prev_i), c64);
               Value prev_nE = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimN, prev_j), c64);
               innerB.create<memref::DmaWaitOp>(innerL, tC_wb, ValueRange{c0}, innerB.create<arith::MulIOp>(innerL, prev_mE, prev_nE));
               innerB.create<scf::YieldOp>(innerL);
            });

            ifB.create<scf::YieldOp>(ifL);
        });
        nB.create<func::CallOp>(nL, hwBarrierFn, ValueRange{});

        nB.create<scf::YieldOp>(nL, ValueRange{
            k_a_c, k_a_f, k_b_c, k_b_f, 
            k_tA_c, k_tA_f, k_tB_c, k_tB_f, 
            k_c_acc, k_c_free, k_c_last, 
            tC_acc, tC_spare, tC_wb, 
            true_val, i, j, false_val
        });
      });
      mB.create<scf::YieldOp>(mL, loopN.getResults());
    });
    
    Value final_c_wb = loopM.getResult(8);
    Value final_tC_wb = loopM.getResult(11);
    Value final_prev_valid = loopM.getResult(14);
    Value final_prev_i = loopM.getResult(15);
    Value final_prev_j = loopM.getResult(16);

    builder.create<scf::IfOp>(loc, isCore0, [&](OpBuilder &ifB, Location ifL) {
       ifB.create<scf::IfOp>(ifL, final_prev_valid, [&](OpBuilder &innerB, Location innerL) {
           Value prev_mE = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimM, final_prev_i), c64);
           Value prev_nE = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimN, final_prev_j), c64);
           Value cSubOrig = createSubview2D(innerB, innerL, c, final_prev_i, final_prev_j, prev_mE, prev_nE, cElemType, cType.getMemorySpace());
           Value cWbS = createSubview2D(innerB, innerL, final_c_wb, c0, c0, prev_mE, prev_nE, cElemType, l1SpaceAttr);
           
           innerB.create<memref::DmaStartOp>(innerL, cWbS, ValueRange{c0,c0}, cSubOrig, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, prev_mE, prev_nE), final_tC_wb, ValueRange{c0});
           innerB.create<memref::DmaWaitOp>(innerL, final_tC_wb, ValueRange{c0}, innerB.create<arith::MulIOp>(innerL, prev_mE, prev_nE));
           
           innerB.create<scf::YieldOp>(innerL);
       });
       ifB.create<scf::YieldOp>(ifL);
    });
    
    builder.create<func::CallOp>(loc, hwBarrierFn, ValueRange{});

    builder.create<scf::IfOp>(loc, isCore0, [&](OpBuilder &ifB, Location ifL) {
        ifB.create<func::CallOp>(ifL, l1ResetFn, ValueRange{});
        ifB.create<scf::YieldOp>(ifL);
    });

    builder.create<func::CallOp>(loc, hwBarrierFn, ValueRange{});

    builder.create<memref::DeallocOp>(loc, aTrans);

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