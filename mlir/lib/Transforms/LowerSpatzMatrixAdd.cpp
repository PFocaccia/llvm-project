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

struct LowerSpatzMatrixAddPass : public LowerSpatzMatrixAddBase<LowerSpatzMatrixAddPass> {
  LowerSpatzMatrixAddPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<MatrixAddLowering, MatrixVectorAddLowering, MatrixScalarMulLowering>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createLowerSpatzMatrixAddPass() {
  return std::make_unique<LowerSpatzMatrixAddPass>();
}
} // namespace mlir