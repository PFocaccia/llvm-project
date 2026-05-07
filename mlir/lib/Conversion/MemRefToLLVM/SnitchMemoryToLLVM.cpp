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

static Value castIntToWidth(ConversionPatternRewriter &rewriter, Location loc, Value value, unsigned targetWidth) {
    
  auto valueTy = value.getType().cast<IntegerType>();
  if (valueTy.getWidth() == targetWidth) return value;
  
  auto targetTy = IntegerType::get(rewriter.getContext(), targetWidth);
  if (valueTy.getWidth() < targetWidth) return rewriter.create<LLVM::ZExtOp>(loc, targetTy, value);
  
  return rewriter.create<LLVM::TruncOp>(loc, targetTy, value);
}

static bool isSupportedSdmaTwodCopy(memref::CopyOp op, unsigned l1MemorySpace) {
  
  auto srcType = op.getOperands()[0].getType().dyn_cast<MemRefType>();
  auto dstType = op.getOperands()[1].getType().dyn_cast<MemRefType>();
  if (!srcType || !dstType) return false;

  unsigned srcSpace = srcType.getMemorySpaceAsInt();
  unsigned dstSpace = dstType.getMemorySpaceAsInt();
    
  bool isL1DefaultCopy = (srcSpace == l1MemorySpace && dstSpace == 0) || (srcSpace == 0 && dstSpace == l1MemorySpace);
  if (!isL1DefaultCopy) return false;

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

  LogicalResult matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct SnitchSdmaTwodCopyOpLowering : public ConvertOpToLLVMPattern<memref::CopyOp> {
  
  SnitchSdmaTwodCopyOpLowering(LLVMTypeConverter &converter, unsigned l1MemorySpace)
      : ConvertOpToLLVMPattern<memref::CopyOp>(converter), l1MemorySpace(l1MemorySpace) {}

  LogicalResult matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    if (!isSupportedSdmaTwodCopy(op, l1MemorySpace)) return failure();

    auto loc = op.getLoc();
    auto srcType = op.getOperands()[0].getType().cast<MemRefType>();

    MemRefDescriptor srcDesc(adaptor.getOperands()[0]);
    MemRefDescriptor dstDesc(adaptor.getOperands()[1]);

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
    unsigned bitWidth = srcType.getElementType().getIntOrFloatBitWidth();
    
    if (bitWidth < 32) {
        unsigned shift = (bitWidth == 16) ? 1 : 2;
        unsigned multiplier = 1 << shift;
        
        Value shiftVal = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, shift));
        Value multVal =  rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, multiplier));
        Value epsVal =   rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, multiplier - 1));

        Value nRepsPlusEps = rewriter.create<LLVM::AddOp>(loc, nReps32, epsVal);
        nReps32 = rewriter.create<LLVM::LShrOp>(loc, nRepsPlusEps, shiftVal);
        
        size32 = rewriter.create<LLVM::MulOp>(loc, size32, multVal);
        srcStride32 = rewriter.create<LLVM::MulOp>(loc, srcStride32, multVal);
        dstStride32 = rewriter.create<LLVM::MulOp>(loc, dstStride32, multVal);
        
        auto indexTy = srcOffset.getType();
        Value srcHorizElems = rewriter.create<LLVM::URemOp>(loc, srcOffset, srcStrideElems);
        Value dstHorizElems = rewriter.create<LLVM::URemOp>(loc, dstOffset, dstStrideElems);
        
        Value multMinusOne = rewriter.create<LLVM::ConstantOp>(loc, indexTy, rewriter.getIntegerAttr(indexTy, multiplier - 1));
        Value srcExtraElems = rewriter.create<LLVM::MulOp>(loc, srcHorizElems, multMinusOne);
        Value dstExtraElems = rewriter.create<LLVM::MulOp>(loc, dstHorizElems, multMinusOne);
        
        Value srcExtraBytes = rewriter.create<LLVM::MulOp>(loc, srcExtraElems, elemSizeIdx);
        Value dstExtraBytes = rewriter.create<LLVM::MulOp>(loc, dstExtraElems, elemSizeIdx);
        
        Value srcExtraBytes64 = castIntToWidth(rewriter, loc, srcExtraBytes, 64);
        Value dstExtraBytes64 = castIntToWidth(rewriter, loc, dstExtraBytes, 64);
        
        srcAddr64 = rewriter.create<LLVM::AddOp>(loc, srcAddr64, srcExtraBytes64);
        dstAddr64 = rewriter.create<LLVM::AddOp>(loc, dstAddr64, dstExtraBytes64);
    }

    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
    Value cfg32 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 0));
    ModuleOp module = op->getParentOfType<ModuleOp>();
    
    auto sdmaStartFn = LLVM::lookupOrCreateFn(module, "llvm.riscv.sdma.start.twod", {i64Ty, i64Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty}, i32Ty);
    auto startCallResults = createLLVMCall(rewriter, loc, sdmaStartFn, {srcAddr64, dstAddr64, size32, srcStride32, dstStride32, nReps32, cfg32}, i32Ty);
    Value tid = startCallResults[0];

    Block *currentBlock = rewriter.getBlock();
    Block *exitBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.createBlock(currentBlock->getParent(), exitBlock->getIterator());

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange(), loopBlock);
    rewriter.setInsertionPointToEnd(loopBlock);
    
    Value statSelector = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(0));
    auto sdmaStatFn = LLVM::lookupOrCreateFn(module, "llvm.riscv.sdma.stat", {i32Ty}, i32Ty);
    auto statCallResults = createLLVMCall(rewriter, loc, sdmaStatFn, {statSelector}, i32Ty);
    Value completedId = statCallResults[0];

    Value isBusy = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, completedId, tid);
    rewriter.create<LLVM::CondBrOp>(loc, isBusy, loopBlock, exitBlock);
    rewriter.setInsertionPointToStart(exitBlock);

    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned l1MemorySpace;
};

struct SnitchDmaStartOpLowering : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
    SnitchDmaStartOpLowering(LLVMTypeConverter &converter, unsigned l1MemorySpace)
      : ConvertOpToLLVMPattern<memref::DmaStartOp>(converter), l1MemorySpace(l1MemorySpace) {}

  LogicalResult matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    auto srcType = op.getSrcMemRef().getType().dyn_cast<MemRefType>();
    auto dstType = op.getDstMemRef().getType().dyn_cast<MemRefType>();
    if (!srcType || !dstType || srcType.getRank() != 2 || dstType.getRank() != 2) return failure();

    unsigned srcSpace = srcType.getMemorySpaceAsInt();
    unsigned dstSpace = dstType.getMemorySpaceAsInt();
    if (!((srcSpace == l1MemorySpace && dstSpace == 0) || (srcSpace == 0 && dstSpace == l1MemorySpace))) return failure();

    auto loc = op.getLoc();
    unsigned srcRank = srcType.getRank();
    unsigned dstRank = dstType.getRank();
    
    Value srcAdaptor = adaptor.getOperands()[0];
    Value dstAdaptor = adaptor.getOperands()[1 + srcRank];
    Value tagAdaptor = adaptor.getOperands()[1 + srcRank + 1 + dstRank + 1];

    MemRefDescriptor srcDesc(srcAdaptor);
    MemRefDescriptor dstDesc(dstAdaptor);
    MemRefDescriptor tagDesc(tagAdaptor);

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

    unsigned bitWidth = srcType.getElementType().getIntOrFloatBitWidth();
    
    if (bitWidth < 32) {
        unsigned shift = (bitWidth == 16) ? 1 : 2;
        unsigned multiplier = 1 << shift;
        
        Value shiftVal = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, shift));
        Value multVal =  rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, multiplier));
        Value epsVal =   rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, multiplier - 1));

        Value nRepsPlusEps = rewriter.create<LLVM::AddOp>(loc, nReps32, epsVal);
        nReps32 = rewriter.create<LLVM::LShrOp>(loc, nRepsPlusEps, shiftVal);
        
        size32 = rewriter.create<LLVM::MulOp>(loc, size32, multVal);
        srcStride32 = rewriter.create<LLVM::MulOp>(loc, srcStride32, multVal);
        dstStride32 = rewriter.create<LLVM::MulOp>(loc, dstStride32, multVal);
        
        auto indexTy = srcOffset.getType();
        Value srcHorizElems = rewriter.create<LLVM::URemOp>(loc, srcOffset, srcStrideElems);
        Value dstHorizElems = rewriter.create<LLVM::URemOp>(loc, dstOffset, dstStrideElems);
        
        Value multMinusOne = rewriter.create<LLVM::ConstantOp>(loc, indexTy, rewriter.getIntegerAttr(indexTy, multiplier - 1));
        Value srcExtraElems = rewriter.create<LLVM::MulOp>(loc, srcHorizElems, multMinusOne);
        Value dstExtraElems = rewriter.create<LLVM::MulOp>(loc, dstHorizElems, multMinusOne);
        
        Value srcExtraBytes = rewriter.create<LLVM::MulOp>(loc, srcExtraElems, elemSizeIdx);
        Value dstExtraBytes = rewriter.create<LLVM::MulOp>(loc, dstExtraElems, elemSizeIdx);
        
        Value srcExtraBytes64 = castIntToWidth(rewriter, loc, srcExtraBytes, 64);
        Value dstExtraBytes64 = castIntToWidth(rewriter, loc, dstExtraBytes, 64);
        
        srcAddr64 = rewriter.create<LLVM::AddOp>(loc, srcAddr64, srcExtraBytes64);
        dstAddr64 = rewriter.create<LLVM::AddOp>(loc, dstAddr64, dstExtraBytes64);
    }

    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
    Value cfg32 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 0));

    ModuleOp module = op->getParentOfType<ModuleOp>();
    
    auto sdmaStartFn = LLVM::lookupOrCreateFn(module, "llvm.riscv.sdma.start.twod", {i64Ty, i64Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty}, i32Ty);
    auto startCallResults = createLLVMCall(rewriter, loc, sdmaStartFn, {srcAddr64, dstAddr64, size32, srcStride32, dstStride32, nReps32, cfg32}, i32Ty);
    Value tid = startCallResults[0];

    Value tagBasePtr = tagDesc.alignedPtr(rewriter, loc);
    Value tagOffset = tagDesc.offset(rewriter, loc);
    Value tagPtr = rewriter.create<LLVM::GEPOp>(loc, tagBasePtr.getType(), tagBasePtr, tagOffset);
    rewriter.create<LLVM::StoreOp>(loc, tid, tagPtr);

    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned l1MemorySpace;
};


struct SnitchDmaWaitOpLowering : public ConvertOpToLLVMPattern<memref::DmaWaitOp> {
  
  SnitchDmaWaitOpLowering(LLVMTypeConverter &converter) : ConvertOpToLLVMPattern<memref::DmaWaitOp>(converter) {}

  LogicalResult matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    MemRefDescriptor tagDesc(adaptor.getOperands()[0]);

    Value tagBasePtr = tagDesc.alignedPtr(rewriter, loc);
    Value tagOffset = tagDesc.offset(rewriter, loc);
    Value tagPtr = rewriter.create<LLVM::GEPOp>(loc, tagBasePtr.getType(), tagBasePtr, tagOffset);
    Value tid = rewriter.create<LLVM::LoadOp>(loc, tagPtr);

    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    ModuleOp module = op->getParentOfType<ModuleOp>();

    Block *currentBlock = rewriter.getBlock();
    Block *exitBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.createBlock(currentBlock->getParent(), exitBlock->getIterator());

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange(), loopBlock);
    rewriter.setInsertionPointToEnd(loopBlock);
    
    Value statSelector = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(0));
    auto sdmaStatFn = LLVM::lookupOrCreateFn(module, "llvm.riscv.sdma.stat", {i32Ty}, i32Ty);
    auto statCallResults = createLLVMCall(rewriter, loc, sdmaStatFn, {statSelector}, i32Ty);
    Value completedId = statCallResults[0];

    Value isBusy = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, completedId, tid);
    rewriter.create<LLVM::CondBrOp>(loc, isBusy, loopBlock, exitBlock);
    rewriter.setInsertionPointToStart(exitBlock);

    rewriter.eraseOp(op);
    return success();
  }
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
    patterns.add<SnitchDmaStartOpLowering>(typeConverter, l1MemorySpace);
    patterns.add<SnitchDmaWaitOpLowering>(typeConverter);
    
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

    target.addDynamicallyLegalOp<memref::DmaStartOp>([&](memref::DmaStartOp startOp) {
      
      auto srcType = startOp.getSrcMemRef().getType().dyn_cast<MemRefType>();
      auto dstType = startOp.getDstMemRef().getType().dyn_cast<MemRefType>();
      
      if (!srcType || !dstType) return true;
      
      unsigned srcSpace = srcType.getMemorySpaceAsInt();
      unsigned dstSpace = dstType.getMemorySpaceAsInt();
      bool isL1Transfer = (srcSpace == l1MemorySpace && dstSpace == 0) || (srcSpace == 0 && dstSpace == l1MemorySpace);
      
      return !isL1Transfer;
    });

    target.addDynamicallyLegalOp<memref::DmaWaitOp>([](memref::DmaWaitOp waitOp) { return false; });

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