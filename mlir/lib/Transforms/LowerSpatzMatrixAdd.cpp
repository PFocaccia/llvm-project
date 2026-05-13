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
    auto vecLenAttr = op->getAttrOfType<IntegerAttr>("vecLen");
    
    if (!vecLenAttr || vecLenAttr.getInt() <= 0) return failure();

    auto vectorType = VectorType::get({8}, elemType, {true});

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Type i32Type = rewriter.getI32Type();
    Value vlI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, op.cols());

    rewriter.create<scf::ForOp>(loc, c0, op.rows(), c1, ValueRange{}, [&](OpBuilder &builder, Location bodyLoc, Value iv, ValueRange) {
          
          Value rowIdx = iv;
          Value indices[] = {rowIdx, c0};

          Value accVec = builder.create<spatz::VLEOp>(bodyLoc, vectorType, op.accMatrix(), indices, vlI32);
          Value tmpVec = builder.create<spatz::VLEOp>(bodyLoc, vectorType, op.tmpMatrix(), indices, vlI32);
          
          Value undefVec = builder.create<LLVM::UndefOp>(bodyLoc, vectorType);

          Value sumVec;
          if (elemType.isa<FloatType>()) {
            sumVec = builder.create<spatz::VFAddVVOp>(bodyLoc, vectorType, undefVec, accVec, tmpVec, vlI32);
          } else {
            sumVec = builder.create<spatz::VAddVVOp>(bodyLoc, vectorType, undefVec, accVec, tmpVec, vlI32);
          }

          builder.create<spatz::VSEOp>(bodyLoc, sumVec, op.accMatrix(), indices, vlI32);
          
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
    patterns.add<MatrixAddLowering>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createLowerSpatzMatrixAddPass() {
  return std::make_unique<LowerSpatzMatrixAddPass>();
}
} // namespace mlir