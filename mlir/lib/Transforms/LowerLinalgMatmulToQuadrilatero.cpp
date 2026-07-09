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
#include "llvm/ADT/SmallPtrSet.h"
#include <cmath>

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
  
  Value subview3D = builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides).getResult();

  SmallVector<ReassociationIndices, 2> reassociation = {{0, 1}, {2}};
  
  return builder.create<memref::CollapseShapeOp>(loc, subview3D, reassociation).getResult();

}

static Value createSubview1D(OpBuilder &builder, Location loc, Value base, Value off0, Value size0,
                             Type elementType, Attribute memorySpace) {

  SmallVector<OpFoldResult, 1> offsets = {off0};
  SmallVector<OpFoldResult, 1> sizes = {size0};
  SmallVector<OpFoldResult, 1> strides = {builder.getIndexAttr(1)};
  
  return builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides).getResult();

}

struct LowerLinalgMatmulToQuadrilateroPass : public LowerLinalgMatmulToQuadrilateroBase<LowerLinalgMatmulToQuadrilateroPass> {

  void runOnOperation() override {

    func::FuncOp funcOp = getOperation();
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

    SmallVector<linalg::MatmulOp, 4> matmuls;
    funcOp.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    for (linalg::MatmulOp op : matmuls) {
      if (failed(lowerMatmul(op))) signalPassFailure();
    }
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
    auto expFn = getOrInsertFn("baremetal_exp", builder.getFunctionType({builder.getF32Type()}, {builder.getF32Type()}));
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
                    
                    b0.create<memref::DmaStartOp>(l0, kSub_0, ValueRange{c0_idx, c0_idx}, kL1Sub_0, ValueRange{c0_idx, c0_idx}, szK_0_idx, tagK_0, ValueRange{c0_idx}, hw_stride_idx, nE_0_idx);
                    b0.create<memref::DmaStartOp>(l0, vSub_0, ValueRange{c0_idx, c0_idx}, vL1Sub_0, ValueRange{c0_idx, c0_idx}, szK_0_idx, tagV_0, ValueRange{c0_idx}, hw_stride_idx, nE_0_idx);
                    
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
                                
                                bN.create<memref::DmaStartOp>(lN, kSub_n, ValueRange{c0_idx, c0_idx}, kL1Sub_n, ValueRange{c0_idx, c0_idx}, szN_idx, tagK_n, ValueRange{c0_idx}, hw_stride_idx, n_nE_idx);
                                bN.create<memref::DmaStartOp>(lN, vSub_n, ValueRange{c0_idx, c0_idx}, vL1Sub_n, ValueRange{c0_idx, c0_idx}, szN_idx, tagV_n, ValueRange{c0_idx}, hw_stride_idx, n_nE_idx);
                                
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
                            
                            b1.create<quadrilatero::TcdmMatmulMemRefOp>(l1, kSub, qTSub, sTSub, mE_idx, nE_idx, D_idx,
                                        shift_idx, builder.getI32IntegerAttr(dt_val), builder.getI32IntegerAttr(dt_val), builder.getI32IntegerAttr(dt_val));
                            
                            b1.create<scf::YieldOp>(l1);
                        });

                        bKV.create<func::CallOp>(lKV, hwBarrierFn, ValueRange{});

                        bKV.create<scf::IfOp>(lKV, isCore0, [&](OpBuilder &b0, Location l0) {
                            
                            b0.create<spatz::MatrixScalarMulOp>(l0, s_buf, m_scale, nE_idx, mE_idx, b0.getI64IntegerAttr(HW_Stride_Elements), b0.getI32IntegerAttr(dt_val));
                            
                            b0.create<spatz::MatrixColumnMaxOp>(l0, s_buf, m_block, nE_idx, mE_idx, b0.getI64IntegerAttr(HW_Stride_Elements), b0.getI32IntegerAttr(dt_val));
                            
                            b0.create<scf::ForOp>(l0, c0_idx, mE_idx, c1_idx, ValueRange{},
                                [&](OpBuilder &bR, Location lR, Value r_idx, ValueRange) {
                                    
                                    Value m_blk = bR.create<memref::LoadOp>(lR, m_block, ValueRange{r_idx});
                                    Value m_prev = bR.create<memref::LoadOp>(lR, m_val, ValueRange{r_idx});
                                
                                    Value is_gt = bR.create<arith::CmpFOp>(lR, arith::CmpFPredicate::UGT, m_blk, m_prev);
                                    Value m_new = bR.create<arith::SelectOp>(lR, is_gt, m_blk, m_prev);
                                    bR.create<memref::StoreOp>(lR, m_new, m_val, ValueRange{r_idx});
                                
                                    Value m_diff = bR.create<arith::SubFOp>(lR, m_prev, m_new);
                                    Value diff_prev = bR.create<func::CallOp>(lR, expFn, ValueRange{m_diff}).getResult(0);
                                
                                    Value l_prev = bR.create<memref::LoadOp>(lR, l_val, ValueRange{r_idx});
                                    Value l_scaled = bR.create<arith::MulFOp>(lR, l_prev, diff_prev);
                                
                                    bR.create<scf::ForOp>(lR, c0_idx, D_idx, c1_idx, ValueRange{},
                                        [&](OpBuilder &bC, Location lC, Value d_idx, ValueRange) {
                                            Value o = bC.create<memref::LoadOp>(lC, o_buf, ValueRange{r_idx, d_idx});
                                            Value o_rescaled = bC.create<arith::MulFOp>(lC, o, diff_prev);
                                            bC.create<memref::StoreOp>(lC, o_rescaled, o_buf, ValueRange{r_idx, d_idx});
                                            bC.create<scf::YieldOp>(lC);
                                        });
                                    
                                    Value local_sum = bR.create<memref::AllocaOp>(lR, MemRefType::get({1}, builder.getF32Type()));
                                    bR.create<memref::StoreOp>(lR, f_zero, local_sum, ValueRange{c0_idx});
                                    
                                    bR.create<scf::ForOp>(lR, c0_idx, nE_idx, c1_idx, ValueRange{},
                                        [&](OpBuilder &bC, Location lC, Value c_idx, ValueRange) {
                                            Value s = bC.create<memref::LoadOp>(lC, s_buf, ValueRange{c_idx, r_idx});
                                            if (hasBias) {
                                                Value b_val = bC.create<memref::LoadOp>(lC, bias_curr, ValueRange{c_idx});
                                                s = bC.create<arith::AddFOp>(lC, s, b_val);
                                            }
                                            Value sub_m = bC.create<arith::SubFOp>(lC, s, m_new);
                                            Value exp_s = bC.create<func::CallOp>(lC, expFn, ValueRange{sub_m}).getResult(0);
                                        
                                            Value curr_sum = bC.create<memref::LoadOp>(lC, local_sum, ValueRange{c0_idx});
                                            Value new_sum = bC.create<arith::AddFOp>(lC, curr_sum, exp_s);
                                            bC.create<memref::StoreOp>(lC, new_sum, local_sum, ValueRange{c0_idx});
                                        
                                            bC.create<memref::StoreOp>(lC, exp_s, p_T, ValueRange{c_idx, r_idx});
                                            bC.create<scf::YieldOp>(lC);
                                        });
                                    
                                    Value loc_sum_val = bR.create<memref::LoadOp>(lR, local_sum, ValueRange{c0_idx});
                                    Value l_new = bR.create<arith::AddFOp>(lR, l_scaled, loc_sum_val);
                                    bR.create<memref::StoreOp>(lR, l_new, l_val, ValueRange{r_idx});
                                    
                                    bR.create<scf::YieldOp>(lR);
                                });
                            
                            b0.create<scf::YieldOp>(l0);
                        });

                        bKV.create<func::CallOp>(lKV, hwBarrierFn, ValueRange{});

                        bKV.create<scf::IfOp>(lKV, isCore1, [&](OpBuilder &b1, Location l1) {
                            Value pTSub = createSubview2D(b1, l1, p_T, c0_idx, c0_idx, nE_idx, mE_idx, elemType, l1SpaceAttr);
                            Value vSub  = createSubview2D(b1, l1, v_curr, c0_idx, c0_idx, nE_idx, D_idx, elemType, l1SpaceAttr);
                            Value oTileSub = createSubview2D(b1, l1, o_tile, c0_idx, c0_idx, mE_idx, D_idx, elemType, l1SpaceAttr);
                            b1.create<quadrilatero::TcdmMatmulMemRefOp>(l1, pTSub, vSub, oTileSub, mE_idx, D_idx, nE_idx,
                                        shift_idx, builder.getI32IntegerAttr(dt_val), builder.getI32IntegerAttr(dt_val), builder.getI32IntegerAttr(dt_val));
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
                    b0.create<scf::ForOp>(l0, c0_idx, mE_idx, c1_idx, ValueRange{}, [&](OpBuilder &bR, Location lR, Value r_idx, ValueRange) {
                        Value l_final = bR.create<memref::LoadOp>(lR, l_val, ValueRange{r_idx});
                        bR.create<scf::ForOp>(lR, c0_idx, D_idx, c1_idx, ValueRange{}, [&](OpBuilder &bC, Location lC, Value d_idx, ValueRange) {
                            Value o = bC.create<memref::LoadOp>(lC, o_buf, ValueRange{r_idx, d_idx});
                            Value o_norm = bC.create<arith::DivFOp>(lC, o, l_final);
                            bC.create<memref::StoreOp>(lC, o_norm, o_buf, ValueRange{r_idx, d_idx});
                            bC.create<scf::YieldOp>(lC);
                        });
                        bR.create<scf::YieldOp>(lR);
                    });

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
} // namespace

namespace mlir {
std::unique_ptr<Pass> createLowerLinalgMatmulToQuadrilateroPass() {
  return std::make_unique<LowerLinalgMatmulToQuadrilateroPass>();
}
} // namespace mlir