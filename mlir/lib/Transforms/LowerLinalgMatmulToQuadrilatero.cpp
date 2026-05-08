//===- LowerLinalgMatmulToQuadrilatero.cpp -------------------------------===//
// Ottimizzato con Paradigma DAE (Decoupled Access-Execute)
// Core 0: DMA in/out (snrt_sdma) + Calcolo Vettoriale (spatz.matrix_add)
// Core 1: Calcolo Matriciale (quadrilatero)
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
#include "llvm/ADT/SmallVector.h"
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

    unsigned bitWidth = aElemType.getIntOrFloatBitWidth();
    
    int64_t tileKVal = (bitWidth == 32) ? 64 : (bitWidth == 16) ? 128 : 256;
    int64_t tileMVal = 64; int64_t tileNVal = 64;

    OpBuilder builder(op);
    Location loc = op.getLoc();
    
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

    Value cid = builder.create<func::CallOp>(loc, getCoreIdxFn, ValueRange{}).getResult(0);
    Value cid0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value cid1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    Value isCore0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, cid0); 
    Value isCore1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cid, cid1); 

    auto l1SpaceAttr = IntegerAttr::get(builder.getI64Type(), l1MemorySpace);
    auto aL1Type = MemRefType::get({tileKVal, tileMVal}, aElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto bL1Type = MemRefType::get({tileKVal, tileNVal}, bElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);
    auto cAccType = MemRefType::get({tileMVal, tileNVal}, cElemType, MemRefLayoutAttrInterface{}, l1SpaceAttr);

    Value aL1_0 = builder.create<memref::AllocOp>(loc, aL1Type);
    Value aL1_1 = builder.create<memref::AllocOp>(loc, aL1Type);
    Value bL1_0 = builder.create<memref::AllocOp>(loc, bL1Type);
    Value bL1_1 = builder.create<memref::AllocOp>(loc, bL1Type);
    Value cAcc = builder.create<memref::AllocOp>(loc, cAccType);
    Value cTmp = builder.create<memref::AllocOp>(loc, cAccType);

    auto tagType = MemRefType::get({1}, builder.getI32Type());
    Value tagA_0 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagA_1 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagB_0 = builder.create<memref::AllocaOp>(loc, tagType);
    Value tagB_1 = builder.create<memref::AllocaOp>(loc, tagType);

    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value tk = builder.create<arith::ConstantIndexOp>(loc, tileKVal);
    Value dimK = builder.create<memref::DimOp>(loc, a, 0);
    Value dimM = builder.create<memref::DimOp>(loc, a, 1);
    Value dimN = builder.create<memref::DimOp>(loc, b, 1);

    scf::buildLoopNest(builder, loc, {c0, c0}, {dimM, dimN}, {builder.create<arith::ConstantIndexOp>(loc, 64), builder.create<arith::ConstantIndexOp>(loc, 64)},
        [&](OpBuilder &ijB, Location ijL, ValueRange ijIvs) {
          Value i = ijIvs[0]; Value j = ijIvs[1];
          Value mE = createMinIndex(ijB, ijL, ijB.create<arith::SubIOp>(ijL, dimM, i), builder.create<arith::ConstantIndexOp>(ijL, 64));
          Value nE = createMinIndex(ijB, ijL, ijB.create<arith::SubIOp>(ijL, dimN, j), builder.create<arith::ConstantIndexOp>(ijL, 64));

          Value k0_eff = createMinIndex(ijB, ijL, dimK, tk);
          Value aS0 = createSubview2D(ijB, ijL, a, c0, i, k0_eff, mE, aElemType, aType.getMemorySpace());
          Value bS0 = createSubview2D(ijB, ijL, b, c0, j, k0_eff, nE, bElemType, bType.getMemorySpace());
          Value aL0 = createSubview2D(ijB, ijL, aL1_0, c0, c0, k0_eff, mE, aElemType, l1SpaceAttr);
          Value bL0 = createSubview2D(ijB, ijL, bL1_0, c0, c0, k0_eff, nE, bElemType, l1SpaceAttr);
          
          ijB.create<scf::IfOp>(ijL, isCore0, [&](OpBuilder &ifB, Location ifL) {
            ifB.create<memref::DmaStartOp>(ifL, aS0, ValueRange{c0,c0}, aL0, ValueRange{c0,c0}, ifB.create<arith::MulIOp>(ifL, k0_eff, mE), tagA_0, ValueRange{c0});
            ifB.create<memref::DmaStartOp>(ifL, bS0, ValueRange{c0,c0}, bL0, ValueRange{c0,c0}, ifB.create<arith::MulIOp>(ifL, k0_eff, nE), tagB_0, ValueRange{c0});
            ifB.create<scf::YieldOp>(ifL);
          });

          SmallVector<Value, 8> iterArgs = {aL1_0, aL1_1, bL1_0, bL1_1, tagA_0, tagA_1, tagB_0, tagB_1};
          
          ijB.create<scf::ForOp>(ijL, c0, dimK, tk, iterArgs,
            [&](OpBuilder &kB, Location kL, Value k, ValueRange rArgs) {
              Value curA = rArgs[0]; Value nxtA = rArgs[1]; Value curB = rArgs[2]; Value nxtB = rArgs[3];
              Value curTA = rArgs[4]; Value nxtTA = rArgs[5]; Value curTB = rArgs[6]; Value nxtTB = rArgs[7];

              Value k_eff = createMinIndex(kB, kL, kB.create<arith::SubIOp>(kL, dimK, k), tk);
              Value nEA = kB.create<arith::MulIOp>(kL, k_eff, mE);
              Value nEB = kB.create<arith::MulIOp>(kL, k_eff, nE);

              Value next_k = kB.create<arith::AddIOp>(kL, k, tk);
              Value has_next = kB.create<arith::CmpIOp>(kL, arith::CmpIPredicate::slt, next_k, dimK);
              
              kB.create<scf::IfOp>(kL, isCore0, [&](OpBuilder &ifB, Location ifL) {
                ifB.create<scf::IfOp>(ifL, has_next, [&](OpBuilder &innerB, Location innerL) {
                  Value nk_eff = createMinIndex(innerB, innerL, innerB.create<arith::SubIOp>(innerL, dimK, next_k), tk);
                  Value aSn = createSubview2D(innerB, innerL, a, next_k, i, nk_eff, mE, aElemType, aType.getMemorySpace());
                  Value bSn = createSubview2D(innerB, innerL, b, next_k, j, nk_eff, nE, bElemType, bType.getMemorySpace());
                  Value aLn = createSubview2D(innerB, innerL, nxtA, c0, c0, nk_eff, mE, aElemType, l1SpaceAttr);
                  Value bLn = createSubview2D(innerB, innerL, nxtB, c0, c0, nk_eff, nE, bElemType, l1SpaceAttr);
                  innerB.create<memref::DmaStartOp>(innerL, aSn, ValueRange{c0,c0}, aLn, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, nk_eff, mE), nxtTA, ValueRange{c0});
                  innerB.create<memref::DmaStartOp>(innerL, bSn, ValueRange{c0,c0}, bLn, ValueRange{c0,c0}, innerB.create<arith::MulIOp>(innerL, nk_eff, nE), nxtTB, ValueRange{c0});
                  innerB.create<scf::YieldOp>(innerL);
                });
                ifB.create<scf::YieldOp>(ifL);
              });

              kB.create<scf::IfOp>(kL, isCore0, [&](OpBuilder &ifB, Location ifL) {
                ifB.create<memref::DmaWaitOp>(ifL, curTA, ValueRange{c0}, nEA);
                ifB.create<memref::DmaWaitOp>(ifL, curTB, ValueRange{c0}, nEB);
                ifB.create<scf::YieldOp>(ifL);
              });

              kB.create<func::CallOp>(kL, hwBarrierFn, ValueRange{});

              Value aSubC = createSubview2D(kB, kL, curA, c0, c0, k_eff, mE, aElemType, l1SpaceAttr);
              Value bSubC = createSubview2D(kB, kL, curB, c0, c0, k_eff, nE, bElemType, l1SpaceAttr);
              Value cAccS = createSubview2D(kB, kL, cAcc, c0, c0, mE, nE, cElemType, l1SpaceAttr);
              Value is_first = kB.create<arith::CmpIOp>(kL, arith::CmpIPredicate::eq, k, c0);

              kB.create<scf::IfOp>(kL, isCore1, [&](OpBuilder &ifB, Location ifL) {
                ifB.create<scf::IfOp>(ifL, is_first, [&](OpBuilder &firstB, Location firstL) {
                  firstB.create<quadrilatero::TcdmMatmulMemRefOp>(firstL, aSubC, bSubC, cAccS, mE, nE, k_eff, 
                    builder.create<arith::ConstantIndexOp>(firstL, 0), builder.getI32IntegerAttr(dtC_val), builder.getI32IntegerAttr(dtA_val), builder.getI32IntegerAttr(dtB_val));
                  firstB.create<scf::YieldOp>(firstL);
                }, [&](OpBuilder &elseB, Location elseL) {
                  Value cTmpS = createSubview2D(elseB, elseL, cTmp, c0, c0, mE, nE, cElemType, l1SpaceAttr);
                  elseB.create<quadrilatero::TcdmMatmulMemRefOp>(elseL, aSubC, bSubC, cTmpS, mE, nE, k_eff, 
                    builder.create<arith::ConstantIndexOp>(elseL, 0), builder.getI32IntegerAttr(dtC_val), builder.getI32IntegerAttr(dtA_val), builder.getI32IntegerAttr(dtB_val));
                  elseB.create<scf::YieldOp>(elseL);
                });
                ifB.create<scf::YieldOp>(ifL);
              });

              kB.create<func::CallOp>(kL, hwBarrierFn, ValueRange{});

              kB.create<scf::IfOp>(kL, isCore0, [&](OpBuilder &ifB, Location ifL) {
                Value not_first = ifB.create<arith::CmpIOp>(ifL, arith::CmpIPredicate::ne, k, c0);
                ifB.create<scf::IfOp>(ifL, not_first, [&](OpBuilder &innerB, Location innerL) {
                  Value cTmpS = createSubview2D(innerB, innerL, cTmp, c0, c0, mE, nE, cElemType, l1SpaceAttr);
                  innerB.create<spatz::MatrixAddOp>(innerL, cAccS, cTmpS, mE, nE, builder.getI64IntegerAttr(64));
                  innerB.create<scf::YieldOp>(innerL);
                });
                ifB.create<scf::YieldOp>(ifL);
              });

              kB.create<scf::YieldOp>(kL, ValueRange{nxtA, curA, nxtB, curB, nxtTA, curTA, nxtTB, curTB});
            });

          ijB.create<scf::IfOp>(ijL, isCore0, [&](OpBuilder &ifB, Location ifL) {
            Value cSubOrig = createSubview2D(ifB, ifL, c, i, j, mE, nE, cElemType, cType.getMemorySpace());
            Value cAccFinal = createSubview2D(ifB, ifL, cAcc, c0, c0, mE, nE, cElemType, l1SpaceAttr);
            ifB.create<memref::CopyOp>(ifL, cAccFinal, cSubOrig);
            ifB.create<scf::YieldOp>(ifL);
          });

          ijB.create<func::CallOp>(ijL, hwBarrierFn, ValueRange{});
        });
    
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