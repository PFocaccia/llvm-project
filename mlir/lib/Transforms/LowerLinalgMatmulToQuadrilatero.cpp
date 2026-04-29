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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"

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

static Value createMinIndex(OpBuilder &builder, Location loc, Value lhs,Value rhs) {
  
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
  
  LowerLinalgMatmulToQuadrilateroPass() = default;

  void runOnOperation() override {
    
    func::FuncOp funcOp = getOperation();

    SmallVector<linalg::MatmulOp, 4> matmuls;
    
    funcOp.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

    for (linalg::MatmulOp op : matmuls) if (failed(lowerMatmul(op))) signalPassFailure();
  
  }

  LogicalResult lowerMatmul(linalg::MatmulOp op) {
    if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
      return failure();

    Value a = op.inputs()[0];
    Value b = op.inputs()[1];
    Value c = op.outputs()[0];

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

    if (dtA_val == -1 || dtB_val == -1 || dtC_val == -1) {
      op.emitError("Data type not supported by Quadrilatero");
      return failure();
    }
  
    int64_t tileMVal = 64;
    int64_t tileNVal = 64;
    int64_t tileKVal = 64;

    OpBuilder builder(op);
    Location loc = op.getLoc();

    auto l1SpaceAttr = IntegerAttr::get(builder.getI64Type(), l1MemorySpace);

    SmallVector<int64_t, 2> aShape{tileKVal, tileMVal}; 
    SmallVector<int64_t, 2> bShape{tileKVal, tileNVal};
    SmallVector<int64_t, 2> cShape{tileMVal, tileNVal};

    auto aL1Type = MemRefType::get(aShape, aElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto bL1Type = MemRefType::get(bShape, bElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto cAccType = MemRefType::get(cShape, cElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto cTmpType = MemRefType::get(cShape, cElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);

    Value aL1 = builder.create<memref::AllocOp>(loc, aL1Type);
    Value bL1 = builder.create<memref::AllocOp>(loc, bL1Type);
    Value cAcc = builder.create<memref::AllocOp>(loc, cAccType);
    Value cTmp = builder.create<memref::AllocOp>(loc, cTmpType);

    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value tileM_val = builder.create<arith::ConstantIndexOp>(loc, tileMVal);
    Value tileN_val = builder.create<arith::ConstantIndexOp>(loc, tileNVal);
    Value tileK_val = builder.create<arith::ConstantIndexOp>(loc, tileKVal);

    unsigned bitWidth = aElemType.getIntOrFloatBitWidth();
    int64_t shift_amount = 0;
    
    if (bitWidth == 32) {
      shift_amount = 2; // 32-bit -> 4 byte -> shift di 2
    } else if (bitWidth == 16) {
      shift_amount = 1; // 16-bit -> 2 byte -> shift di 1
    } else if (bitWidth == 8) {
      shift_amount = 0; // 8-bit  -> 1 byte -> shift di 0
    } else {
      op.emitError("Bit width not supported");
      return failure();
    }

    Value shiftVal = builder.create<arith::ConstantIndexOp>(loc, shift_amount);

    Value dimK = builder.create<memref::DimOp>(loc, a, 0); 
    Value dimM = builder.create<memref::DimOp>(loc, a, 1); 
    Value dimN = builder.create<memref::DimOp>(loc, b, 1);

    SmallVector<Value, 2> ijLbs{c0, c0};
    SmallVector<Value, 2> ijUbs{dimM, dimN};
    SmallVector<Value, 2> ijSteps{tileM_val, tileN_val};

    scf::buildLoopNest(
        builder, loc, ijLbs, ijUbs, ijSteps,
        [&](OpBuilder &ijBuilder, Location ijLoc, ValueRange ijIvs) {
          Value i = ijIvs[0];
          Value j = ijIvs[1];

          Value remM = ijBuilder.create<arith::SubIOp>(ijLoc, dimM, i);
          Value remN = ijBuilder.create<arith::SubIOp>(ijLoc, dimN, j);
          Value mEff = createMinIndex(ijBuilder, ijLoc, remM, tileM_val);
          Value nEff = createMinIndex(ijBuilder, ijLoc, remN, tileN_val);

          Value cSub = createSubview2D(ijBuilder, ijLoc, c, i, j, mEff, nEff,
                                       cElemType, cType.getMemorySpace());
          Value cAccSub = createSubview2D(ijBuilder, ijLoc, cAcc, c0, c0, mEff, nEff,
                                          cElemType, l1SpaceAttr);

          ijBuilder.create<memref::CopyOp>(ijLoc, cSub, cAccSub);

          scf::buildLoopNest(
              ijBuilder, ijLoc, {c0}, {dimK}, {tileK_val},
              [&](OpBuilder &kBuilder, Location kLoc, ValueRange kIvs) {
                Value k = kIvs[0];

                Value remK = kBuilder.create<arith::SubIOp>(kLoc, dimK, k);
                Value kEff = createMinIndex(kBuilder, kLoc, remK, tileK_val);

                Value aSub = createSubview2D(kBuilder, kLoc, a, k, i, kEff, mEff,
                                             aElemType, aType.getMemorySpace());

                Value bSub = createSubview2D(kBuilder, kLoc, b, k, j, kEff, nEff,
                                             bElemType, bType.getMemorySpace());

                Value aL1Sub = createSubview2D(kBuilder, kLoc, aL1, c0, c0, kEff, mEff,
                                               aElemType, l1SpaceAttr);
                
                Value bL1Sub = createSubview2D(kBuilder, kLoc, bL1, c0, c0, kEff, nEff,
                                               bElemType, l1SpaceAttr);
                Value cTmpSub = createSubview2D(kBuilder, kLoc, cTmp, c0, c0, mEff, nEff,
                                                cElemType, l1SpaceAttr);

                kBuilder.create<memref::CopyOp>(kLoc, aSub, aL1Sub);
                kBuilder.create<memref::CopyOp>(kLoc, bSub, bL1Sub);

                kBuilder.create<quadrilatero::TcdmMatmulMemRefOp>(
                    kLoc, aL1Sub, bL1Sub, cTmpSub, mEff, nEff, kEff,
                    shiftVal, 
                    kBuilder.getI32IntegerAttr(dtC_val),
                    kBuilder.getI32IntegerAttr(dtA_val),
                    kBuilder.getI32IntegerAttr(dtB_val));

                kBuilder.create<spatz::MatrixAddOp>(
                  kLoc, cAccSub, cTmpSub, mEff, nEff,
                  kBuilder.getI64IntegerAttr(tileNVal));
              });

          ijBuilder.create<memref::CopyOp>(ijLoc, cAccSub, cSub);
        });

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