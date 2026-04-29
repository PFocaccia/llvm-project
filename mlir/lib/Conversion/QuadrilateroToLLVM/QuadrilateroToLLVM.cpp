#include "mlir/Conversion/QuadrilateroToLLVM/QuadrilateroToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Quadrilatero/QuadrilateroDialect.h"

using namespace mlir;

namespace {

struct QuadrilateroTcdmMatmulMemRefLowering : public ConvertOpToLLVMPattern<quadrilatero::TcdmMatmulMemRefOp> {
  using ConvertOpToLLVMPattern<quadrilatero::TcdmMatmulMemRefOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(quadrilatero::TcdmMatmulMemRefOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();

    MemRefDescriptor aDesc(adaptor.getOperands()[0]);
    MemRefDescriptor bDesc(adaptor.getOperands()[1]);
    MemRefDescriptor cDesc(adaptor.getOperands()[2]);

    Value aBase = aDesc.alignedPtr(rewriter, loc);
    Value aOffset = aDesc.offset(rewriter, loc);
    Value aPtr = rewriter.create<LLVM::GEPOp>(loc, aBase.getType(), aBase, aOffset);

    Value bBase = bDesc.alignedPtr(rewriter, loc);
    Value bOffset = bDesc.offset(rewriter, loc);
    Value bPtr = rewriter.create<LLVM::GEPOp>(loc, bBase.getType(), bBase, bOffset);

    Value cBase = cDesc.alignedPtr(rewriter, loc);
    Value cOffset = cDesc.offset(rewriter, loc);
    Value cPtr = rewriter.create<LLVM::GEPOp>(loc, cBase.getType(), cBase, cOffset);

    auto i8Ty = IntegerType::get(rewriter.getContext(), 8);
    
    auto getI8PtrType = [&](Value value) {
      auto ptrType = value.getType().cast<LLVM::LLVMPointerType>();
      return LLVM::LLVMPointerType::get(i8Ty, ptrType.getAddressSpace());
    };

    Value aI8 = rewriter.create<LLVM::BitcastOp>(loc, getI8PtrType(aPtr), aPtr);
    Value bI8 = rewriter.create<LLVM::BitcastOp>(loc, getI8PtrType(bPtr), bPtr);
    Value cI8 = rewriter.create<LLVM::BitcastOp>(loc, getI8PtrType(cPtr), cPtr);

    auto dtC = op->getAttrOfType<IntegerAttr>("dtC");
    auto dtA = op->getAttrOfType<IntegerAttr>("dtA");
    auto dtB = op->getAttrOfType<IntegerAttr>("dtB");
    
    if (!dtC || !dtA || !dtB) return failure();

    rewriter.create<quadrilatero::TcdmMatmulOp>( loc, aI8, bI8, cI8, adaptor.getOperands()[3], adaptor.getOperands()[4], adaptor.getOperands()[5],
        adaptor.getOperands()[6], dtC, dtA, dtB);
    
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuadrilateroToLLVMPass : public ConvertQuadrilateroToLLVMBase<ConvertQuadrilateroToLLVMPass> {
  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<quadrilatero::TcdmMatmulOp>();
    target.addIllegalOp<quadrilatero::TcdmMatmulMemRefOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<QuadrilateroTcdmMatmulMemRefLowering>(typeConverter);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertQuadrilateroToLLVMPass() {
  return std::make_unique<ConvertQuadrilateroToLLVMPass>();
}
} // namespace mlir