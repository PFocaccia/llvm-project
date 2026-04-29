//===- SnitchMemoryToLLVM.cpp - Snitch memory ops to LLVM conversion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace {

static bool isL1DefaultCopy(memref::CopyOp op, unsigned l1MemorySpace) {
  
    auto srcType = op.source().getType().dyn_cast<MemRefType>();
    auto dstType = op.target().getType().dyn_cast<MemRefType>();
    
    if (!srcType || !dstType) return false;

    unsigned srcSpace = srcType.getMemorySpaceAsInt();
    unsigned dstSpace = dstType.getMemorySpaceAsInt();
    
    return (srcSpace == l1MemorySpace && dstSpace == 0) || (srcSpace == 0 && dstSpace == l1MemorySpace);
}

static bool isSupportedSdmaTwodCopy(memref::CopyOp op, unsigned l1MemorySpace) {
    
    if(!isL1DefaultCopy(op, l1MemorySpace)) return false;

    auto srcType = op.source().getType().dyn_cast<MemRefType>();
    auto dstType = op.target().getType().dyn_cast<MemRefType>();
  
    if(!srcType || !dstType) return false;

    if (srcType.getRank() != 2 || dstType.getRank() != 2) return false;

    return srcType.getElementType() == dstType.getElementType();
}

struct SnitchL1AllocOpLowering : public AllocLikeOpLLVMLowering {
    
    SnitchL1AllocOpLowering(LLVMTypeConverter &converter) : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(), converter) {}

    std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter, Location loc, Value sizeBytes, Operation *op) const override {
    
    auto allocOp = cast<memref::AllocOp>(op);
    auto memRefType = allocOp.getType();
    
    Type elementPtrType = getElementPtrType(memRefType);

    auto allocFn = LLVM::lookupOrCreateFn(allocOp->getParentOfType<ModuleOp>(), "snrt_l1alloc", {getIndexType()}, getVoidPtrType());
    auto results = createLLVMCall(rewriter, loc, allocFn, {sizeBytes}, getVoidPtrType());
    
    Value allocatedPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, elementPtrType, results[0]);

    return std::make_tuple(allocatedPtr, allocatedPtr);
  }
};

struct SnitchL1DeallocOpLowering : public ConvertOpToLLVMPattern<memref::DeallocOp> {
  
  using ConvertOpToLLVMPattern<memref::DeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
  
};

struct SnitchSdmaTwodCopyOpLowering : public ConvertOpToLLVMPattern<memref::CopyOp> {
    
    SnitchSdmaTwodCopyOpLowering(LLVMTypeConverter &converter, unsigned l1MemorySpace)
      : ConvertOpToLLVMPattern<memref::CopyOp>(converter), l1MemorySpace(l1MemorySpace) {}

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    if (!isSupportedSdmaTwodCopy(op, l1MemorySpace)) return failure();

    auto loc = op.getLoc();
    auto srcType = op.source().getType().cast<MemRefType>();

    MemRefDescriptor srcDesc(adaptor.source());
    MemRefDescriptor dstDesc(adaptor.target());

    Value srcBasePtr =  srcDesc.alignedPtr(rewriter, loc);
    Value srcOffset =   srcDesc.offset(rewriter, loc);
    Value srcPtr =      rewriter.create<LLVM::GEPOp>(loc, srcBasePtr.getType(), srcBasePtr, srcOffset);

    Value dstBasePtr =  dstDesc.alignedPtr(rewriter, loc);
    Value dstOffset =   dstDesc.offset(rewriter, loc);
    Value dstPtr =      rewriter.create<LLVM::GEPOp>(loc, dstBasePtr.getType(), dstBasePtr, dstOffset);

    Type srcIntPtrTy =  getIntPtrType(srcBasePtr.getType().cast<LLVM::LLVMPointerType>().getAddressSpace());
    Type dstIntPtrTy =  getIntPtrType(dstBasePtr.getType().cast<LLVM::LLVMPointerType>().getAddressSpace());

    Value srcAddr = rewriter.create<LLVM::PtrToIntOp>(loc, srcIntPtrTy, srcPtr);
    Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(loc, dstIntPtrTy, dstPtr);

    Value srcAddr64 = castIntToWidth(rewriter, loc, srcAddr, 64);
    Value dstAddr64 = castIntToWidth(rewriter, loc, dstAddr, 64);

    Value elemSizeIdx = getSizeInBytes(loc, srcType.getElementType(), rewriter);
    Value rowElems =    srcDesc.size(rewriter, loc, 1);
    Value nReps =       srcDesc.size(rewriter, loc, 0);

    Value srcStrideElems = srcDesc.stride(rewriter, loc, 0);
    Value dstStrideElems = dstDesc.stride(rewriter, loc, 0);

    Value rowBytes =        rewriter.create<LLVM::MulOp>(loc, rowElems, elemSizeIdx);
    Value srcStrideBytes =  rewriter.create<LLVM::MulOp>(loc, srcStrideElems, elemSizeIdx);
    Value dstStrideBytes =  rewriter.create<LLVM::MulOp>(loc, dstStrideElems, elemSizeIdx);

    Value size32 =          castIntToWidth(rewriter, loc, rowBytes, 32);
    Value srcStride32 =     castIntToWidth(rewriter, loc, srcStrideBytes, 32);
    Value dstStride32 =     castIntToWidth(rewriter, loc, dstStrideBytes, 32);
    Value nReps32 =         castIntToWidth(rewriter, loc, nReps, 32);

    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
    Value cfg32 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 0));

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto sdmaStartFn = LLVM::lookupOrCreateFn( module, "llvm.riscv.sdma.start.twod", {i64Ty, i64Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty}, i32Ty);
    (void)createLLVMCall(rewriter, loc, sdmaStartFn, {srcAddr64, dstAddr64, size32, srcStride32, dstStride32, nReps32, cfg32}, i32Ty);

    auto sdmaWaitFn = LLVM::lookupOrCreateFn( module, "llvm.riscv.sdma.wait.for.idle", {}, LLVM::LLVMVoidType::get(rewriter.getContext()));
    (void)createLLVMCall(rewriter, loc, sdmaWaitFn);

    rewriter.eraseOp(op);
    return success();
  }

private:
  static Value castIntToWidth(ConversionPatternRewriter &rewriter, Location loc, Value value, unsigned targetWidth) {
    
    auto valueTy = value.getType().cast<IntegerType>();
    
    if (valueTy.getWidth() == targetWidth) return value;

    auto targetTy = IntegerType::get(rewriter.getContext(), targetWidth);

    if (valueTy.getWidth() < targetWidth) return rewriter.create<LLVM::ZExtOp>(loc, targetTy, value);

    return rewriter.create<LLVM::TruncOp>(loc, targetTy, value);
  }

  unsigned l1MemorySpace;
};

struct ConvertSnitchMemoryToLLVMPass : public ConvertSnitchMemoryToLLVMBase<ConvertSnitchMemoryToLLVMPass> {
  ConvertSnitchMemoryToLLVMPass() = default;

  void runOnOperation() override {

    Operation *op = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(op));
    
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout) options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);

    RewritePatternSet patterns(&getContext());
    patterns.add<SnitchL1AllocOpLowering>(typeConverter);
    patterns.add<SnitchL1DeallocOpLowering>(typeConverter);
    patterns.add<SnitchSdmaTwodCopyOpLowering>(typeConverter, l1MemorySpace);
    
    LLVMConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp allocOp) {
      return allocOp.getType().getMemorySpaceAsInt() != l1MemorySpace;
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>(
        [&](memref::DeallocOp deallocOp) {

          auto memRefType = deallocOp.memref().getType().dyn_cast<MemRefType>();
          return !memRefType || memRefType.getMemorySpaceAsInt() != l1MemorySpace;

        });

    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp copyOp) {
      return !isSupportedSdmaTwodCopy(copyOp, l1MemorySpace);
    });

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertSnitchMemoryToLLVMPass() {
  return std::make_unique<ConvertSnitchMemoryToLLVMPass>();
}
} // namespace mlir
