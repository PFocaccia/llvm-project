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

using namespace mlir;

namespace {

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

struct MatrixExpLowering : public OpRewritePattern<spatz::MatrixExpOp> {
  using OpRewritePattern<spatz::MatrixExpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spatz::MatrixExpOp op, PatternRewriter &rewriter) const override {
    auto accType = op.matrix().getType().dyn_cast<MemRefType>();
    if (!accType || !accType.getElementType().isa<FloatType>()) return failure();

    auto elemType = accType.getElementType();
    auto vecLenAttr = op->getAttrOfType<IntegerAttr>("vecLen");
    if (!vecLenAttr || vecLenAttr.getInt() <= 0) return failure();

    const int64_t vectorSize = 8;   // come nel codice C (LMUL=8)
    auto floatVecType = VectorType::get({vectorSize}, elemType, {true});
    auto intVecType   = VectorType::get({vectorSize}, rewriter.getI32Type(), {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, vectorSize);

    auto f32Type = rewriter.getF32Type();
    Value B = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(1064866805.0f), f32Type);
    Value C = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(12102203.0f), f32Type);

    Value rows = op.rows();
    Value cols = op.cols();

    rewriter.create<scf::ForOp>(loc, c0, rows, c1, ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, Value rowIdx, ValueRange) {
            builder.create<scf::ForOp>(bodyLoc, c0, cols, step, ValueRange{}, [&](OpBuilder &innerBuilder, Location innerLoc, Value colIdx, ValueRange) {
                    
                    Value rem = innerBuilder.create<arith::SubIOp>(innerLoc, cols, colIdx);
                    Value vectorSizeVal = innerBuilder.create<arith::ConstantIndexOp>(innerLoc, vectorSize);
                    Value cond = innerBuilder.create<arith::CmpIOp>(innerLoc, arith::CmpIPredicate::slt, rem, vectorSizeVal);
                    Value vlIndex = innerBuilder.create<arith::SelectOp>(innerLoc, cond, rem, vectorSizeVal);
                    Value vl = innerBuilder.create<arith::IndexCastOp>(innerLoc, rewriter.getI32Type(), vlIndex);

                    Value indices[] = {rowIdx, colIdx};

                    Value xVec = innerBuilder.create<spatz::VLEOp>(innerLoc, floatVecType, op.matrix(), indices, vl);
                    
                    Value bVec = innerBuilder.create<spatz::VFMvVFOp>(innerLoc, floatVecType, innerBuilder.create<LLVM::UndefOp>(innerLoc, floatVecType), B, vl);

                    Value expFloat = innerBuilder.create<spatz::VFMaccVFOp>(innerLoc, floatVecType, bVec, xVec, C, vl);

                    Value expInt = innerBuilder.create<spatz::VFCvtRtzXUFVOp>(innerLoc, intVecType, innerBuilder.create<LLVM::UndefOp>(innerLoc, intVecType), expFloat, vl);

                    Value expFinal = innerBuilder.create<spatz::VBitcastOp>(innerLoc, floatVecType, expInt);

                    innerBuilder.create<spatz::VSEOp>(innerLoc, expFinal, op.matrix(), indices, vl);
                    innerBuilder.create<scf::YieldOp>(innerLoc);
                });
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

    int64_t baseElements = 8;
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
    auto elemType = op.vec_in().getType().cast<MemRefType>().getElementType();
    if (!elemType.isa<FloatType>()) return failure();

    int64_t baseElements = 8;
    auto vecType = VectorType::get({baseElements}, elemType, {true});
    auto intVecType = VectorType::get({baseElements}, rewriter.getI32Type(), {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, baseElements);
    Value len = op.len();

    auto singleElemMemType = MemRefType::get({1}, elemType);
    Value memB = rewriter.create<memref::AllocaOp>(loc, singleElemMemType);
    Value memC = rewriter.create<memref::AllocaOp>(loc, singleElemMemType);
    
    Value constB = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1064866805.0f));
    Value constC = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 12102203.0f));
    
    Value zeroIdx[] = {c0};
    rewriter.create<memref::StoreOp>(loc, constB, memB, zeroIdx);
    rewriter.create<memref::StoreOp>(loc, constC, memC, zeroIdx);

    rewriter.create<scf::ForOp>(loc, c0, len, step, ValueRange{}, 
      [&](OpBuilder &builder, Location bodyLoc, Value idx, ValueRange) {
        
        Value rem = builder.create<arith::SubIOp>(bodyLoc, len, idx);
        Value cond = builder.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::slt, rem, step);
        Value vlIndex = builder.create<arith::SelectOp>(bodyLoc, cond, rem, step);
        Value vl = builder.create<arith::IndexCastOp>(bodyLoc, rewriter.getI32Type(), vlIndex);

        Value indices[] = {idx};
        Value vIn = builder.create<spatz::VLEOp>(bodyLoc, vecType, op.vec_in(), indices, vl);
        Value scalarB = builder.create<memref::LoadOp>(bodyLoc, memB, zeroIdx);
        Value scalarC = builder.create<memref::LoadOp>(bodyLoc, memC, zeroIdx);
        Value undef = builder.create<LLVM::UndefOp>(bodyLoc, vecType);
        Value bVec = builder.create<spatz::VFMvVFOp>(bodyLoc, vecType, undef, scalarB, vl);
        Value macVec = builder.create<spatz::VFMaccVFOp>(bodyLoc, vecType, bVec, vIn, scalarC, vl);
        
        Value undefInt = builder.create<LLVM::UndefOp>(bodyLoc, intVecType);
        Value intVec = builder.create<spatz::VFCvtRtzXUFVOp>(bodyLoc, intVecType, undefInt, macVec, vl);
        Value expVec = builder.create<spatz::VBitcastOp>(bodyLoc, vecType, intVec);

        builder.create<spatz::VSEOp>(bodyLoc, expVec, op.vec_out(), indices, vl);
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
    int64_t baseElements = 8;
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

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    auto elemType = op.matrix().getType().template cast<MemRefType>().getElementType();
    int64_t baseElements = 8;
    auto vecType = VectorType::get({baseElements}, elemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, baseElements);
    
    Value rows = op.rows();
    Value cols = op.cols();
    Value scalarArr;
    
    if constexpr (IsScale) scalarArr = op.scaleVec();
    else scalarArr = op.divVec();

    rewriter.create<scf::ForOp>(loc, c0, rows, c1, ValueRange{}, 
      [&](OpBuilder &builder, Location bodyLoc, Value rIdx, ValueRange) {
        
        builder.create<scf::ForOp>(bodyLoc, c0, cols, step, ValueRange{}, 
          [&](OpBuilder &innerB, Location innerLoc, Value cIdx, ValueRange) {
            
            Value scalar = innerB.create<memref::LoadOp>(innerLoc, scalarArr, ValueRange{rIdx});

            Value rem = innerB.create<arith::SubIOp>(innerLoc, cols, cIdx);
            Value cond = innerB.create<arith::CmpIOp>(innerLoc, arith::CmpIPredicate::slt, rem, step);
            Value vlIndex = innerB.create<arith::SelectOp>(innerLoc, cond, rem, step);
            Value vl = innerB.create<arith::IndexCastOp>(innerLoc, rewriter.getI32Type(), vlIndex);

            Value matIndices[] = {rIdx, cIdx};
            Value matVec = innerB.create<spatz::VLEOp>(innerLoc, vecType, op.matrix(), matIndices, vl);
            
            Value undef = innerB.create<LLVM::UndefOp>(innerLoc, vecType);
            Value bcastVec = innerB.create<spatz::VFMvVFOp>(innerLoc, vecType, undef, scalar, vl);
            
            Value resVec;
            if constexpr (IsScale) {
                resVec = innerB.create<spatz::VFMulVVOp>(innerLoc, vecType, undef, matVec, bcastVec, vl);
            } else {
                resVec = innerB.create<spatz::VFDivVVOp>(innerLoc, vecType, undef, matVec, bcastVec, vl);
            }
            
            innerB.create<spatz::VSEOp>(innerLoc, resVec, op.matrix(), matIndices, vl);
            innerB.create<scf::YieldOp>(innerLoc);
        });
        builder.create<scf::YieldOp>(bodyLoc);
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
    patterns.add<MatrixAddLowering, MatrixVectorAddLowering, MatrixScalarMulLowering, MatrixSubLowering, MatrixExpLowering, MatrixExpReduceLowering, MatrixColumnMaxLowering,
    VectorMaxUpdateLowering, VectorExpLowering, VectorAddLowering, VectorMulLowering, MatrixRowScaleLowering, MatrixRowDivLowering>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createLowerSpatzMatrixAddPass() {
  return std::make_unique<LowerSpatzMatrixAddPass>();
}
} // namespace mlir