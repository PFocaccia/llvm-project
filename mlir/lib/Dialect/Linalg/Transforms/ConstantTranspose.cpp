//===- ConstantTranspose.cpp - Fold constant linalg.transpose --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that folds constant linalg.generic transpose
// operations into a transposed constant at compile time.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
static bool getInputToLoopPermutation(AffineMap map,
                                      SmallVectorImpl<unsigned> &inputToLoop) {
  if (map.getNumSymbols() != 0)
    return false;
  unsigned numDims = map.getNumDims();
  if (map.getNumResults() != numDims)
    return false;

  llvm::SmallBitVector seen(numDims, false);
  inputToLoop.clear();
  inputToLoop.reserve(numDims);
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = expr.dyn_cast<AffineDimExpr>();
    if (!dimExpr)
      return false;
    unsigned loopDim = dimExpr.getPosition();
    if (loopDim >= numDims || seen[loopDim])
      return false;
    seen[loopDim] = true;
    inputToLoop.push_back(loopDim);
  }
  return true;
}

static SmallVector<int64_t> computeStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 0);
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

static int64_t linearizeIndex(ArrayRef<int64_t> indices,
                              ArrayRef<int64_t> strides) {
  int64_t linear = 0;
  for (size_t i = 0; i < indices.size(); ++i)
    linear += indices[i] * strides[i];
  return linear;
}

static void delinearizeIndex(int64_t linear, ArrayRef<int64_t> shape,
                             ArrayRef<int64_t> strides,
                             SmallVectorImpl<int64_t> &indices) {
  indices.resize(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    indices[i] = linear / strides[i];
    linear = linear % strides[i];
  }
}

class FoldConstantTranspose : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasTensorSemantics())
      return failure();
    if (genericOp.getNumInputs() != 1 || genericOp.getNumOutputs() != 1)
      return failure();
        ArrayRef<Attribute> iteratorTypes = genericOp.iterator_types().getValue();
        if (!llvm::all_of(iteratorTypes,
                  [](Attribute it) { return isParallelIterator(it); }))
      return failure();

    DenseElementsAttr denseAttr;
    Value input = genericOp.getInputOperand(0)->get();
    if (!matchPattern(input, m_Constant<DenseElementsAttr>(&denseAttr)))
      return failure();

    auto inputType = denseAttr.getType().dyn_cast<RankedTensorType>();
    auto outputType = genericOp.getResult(0)
                          .getType()
                          .dyn_cast<RankedTensorType>();
    if (!inputType || !outputType)
      return failure();
    if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
      return failure();
    if (inputType.getRank() != outputType.getRank())
      return failure();

    AffineMap inputMap =
        genericOp.getTiedIndexingMap(genericOp.getInputOperand(0));
    AffineMap outputMap =
        genericOp.getTiedIndexingMap(genericOp.getOutputOperand(0));
    if (!outputMap.isIdentity())
      return failure();

    SmallVector<unsigned> inputToLoop;
    if (!getInputToLoopPermutation(inputMap, inputToLoop))
      return failure();
    if (inputToLoop.size() != static_cast<size_t>(inputType.getRank()))
      return failure();

    for (int64_t i = 0, e = inputType.getRank(); i < e; ++i) {
      int64_t expected = outputType.getDimSize(inputToLoop[i]);
      if (inputType.getDimSize(i) != expected)
        return failure();
    }

    Block &body = genericOp.getRegion().front();
    if (body.getNumArguments() != 2)
      return failure();
    if (!llvm::hasSingleElement(body))
      return failure();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp || yieldOp.values().size() != 1)
      return failure();
    if (yieldOp.values()[0] != body.getArgument(0))
      return failure();

    int64_t numElems = denseAttr.getNumElements();
    if (outputType.getNumElements() != numElems)
      return failure();

    SmallVector<Attribute> inputValues =
        llvm::to_vector(denseAttr.getValues<Attribute>());
    SmallVector<Attribute> outputValues(numElems);

    SmallVector<int64_t> inputShape(inputType.getShape().begin(),
                                    inputType.getShape().end());
    SmallVector<int64_t> outputShape(outputType.getShape().begin(),
                                     outputType.getShape().end());
    SmallVector<int64_t> inputStrides = computeStrides(inputShape);
    SmallVector<int64_t> outputStrides = computeStrides(outputShape);

    SmallVector<int64_t> outIdx;
    SmallVector<int64_t> inIdx(inputType.getRank(), 0);
    for (int64_t linear = 0; linear < numElems; ++linear) {
      delinearizeIndex(linear, outputShape, outputStrides, outIdx);
      for (int64_t i = 0, e = inputType.getRank(); i < e; ++i)
        inIdx[i] = outIdx[inputToLoop[i]];
      int64_t inLinear = linearizeIndex(inIdx, inputStrides);
      outputValues[linear] = inputValues[inLinear];
    }

    auto transposedAttr = DenseElementsAttr::get(outputType, outputValues);
    auto constantOp =
        rewriter.create<arith::ConstantOp>(genericOp.getLoc(), outputType,
                                           transposedAttr);
    rewriter.replaceOp(genericOp, constantOp.getResult());
    return success();
  }
};

struct LinalgFoldConstantTransposePass
    : public LinalgFoldConstantTransposeBase<LinalgFoldConstantTransposePass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldConstantTranspose>(context);
    (void)applyPatternsAndFoldGreedily(funcOp.getBody(), std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgFoldConstantTransposePass() {
  return std::make_unique<LinalgFoldConstantTransposePass>();
}
