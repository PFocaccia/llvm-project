#include "../PassDetail.h" // Include fondamentale per i pass TableGen
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Spatz/SpatzDialect.h"

using namespace mlir;

namespace {

static void ensureIntrinsicDeclared(PatternRewriter &rewriter, ModuleOp module, 
                                    StringRef name, Type resultType, 
                                    ArrayRef<Type> argTypes) {
  
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) return;
    
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto fnType = LLVM::LLVMFunctionType::get(resultType, argTypes);
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);

}

static std::string getVectorMangledName(Type elemTy, unsigned elements = 8) {
  std::string result = "nxv" + std::to_string(elements);
  if (elemTy.isInteger(8)) return result + "i8";
  if (elemTy.isInteger(16)) return result + "i16";
  if (elemTy.isInteger(32)) return result + "i32";
  if (elemTy.isInteger(64)) return result + "i64";
  if (elemTy.isF16()) return result + "f16";
  if (elemTy.isBF16()) return result + "bf16";
  if (elemTy.isF32()) return result + "f32";
  if (elemTy.isF64()) return result + "f64";
  return result + "unknown";
}

struct SpatzVleLowering : public ConvertOpToLLVMPattern<spatz::VLEOp> {
  
  using ConvertOpToLLVMPattern<spatz::VLEOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VLEOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    Type baseType = op.base().getType();
    auto memRefType = baseType.dyn_cast<MemRefType>();
    if (!memRefType) return failure(); 

    Value ptr = getStridedElementPtr(op.getLoc(), memRefType, adaptor.base(), adaptor.indices(), rewriter);

    Type llvmResultType = typeConverter->convertType(op.getResult().getType());

    Type elemType = typeConverter->convertType(memRefType.getElementType());
    auto ptrTypeSpace0 = LLVM::LLVMPointerType::get(elemType, 0); 
    Value ptrSpc0 = rewriter.create<LLVM::AddrSpaceCastOp>(op.getLoc(), ptrTypeSpace0, ptr);

    auto vecPtrType = LLVM::LLVMPointerType::get(llvmResultType, 0);
    Value vecPtr = rewriter.create<LLVM::BitcastOp>(op.getLoc(), vecPtrType, ptrSpc0);

    Value passthru = rewriter.create<LLVM::UndefOp>(op.getLoc(), llvmResultType);
    
    ModuleOp module = op->getParentOfType<ModuleOp>();
    std::string mangledName = getVectorMangledName(memRefType.getElementType());
    std::string intrinsicName = "llvm.riscv.vle." + mangledName + ".i32";
    
    SmallVector<Type, 3> argTypes = {passthru.getType(), vecPtr.getType(), adaptor.vl().getType()};
    
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{llvmResultType},  SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{passthru, vecPtr, adaptor.vl()}); 

    return success();
  }
};


struct SpatzVseLowering : public ConvertOpToLLVMPattern<spatz::VSEOp> {
  
  using ConvertOpToLLVMPattern<spatz::VSEOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VSEOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    Type baseType = op.base().getType();
    auto memRefType = baseType.dyn_cast<MemRefType>();
    if (!memRefType) return failure();

    Value ptr = getStridedElementPtr(op.getLoc(), memRefType, adaptor.base(), adaptor.indices(), rewriter);

    Type llvmVecType = adaptor.value().getType();

    Type elemType = typeConverter->convertType(memRefType.getElementType());
    auto ptrTypeSpace0 = LLVM::LLVMPointerType::get(elemType, 0);
    Value ptrSpc0 = rewriter.create<LLVM::AddrSpaceCastOp>(op.getLoc(), ptrTypeSpace0, ptr);

    auto vecPtrType = LLVM::LLVMPointerType::get(llvmVecType, 0);
    Value vecPtr = rewriter.create<LLVM::BitcastOp>(op.getLoc(), vecPtrType, ptrSpc0);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    
    std::string mangledName = getVectorMangledName(memRefType.getElementType());
    std::string intrinsicName = "llvm.riscv.vse." + mangledName + ".i32";
    
    SmallVector<Type, 3> argTypes = {adaptor.value().getType(), vecPtr.getType(), adaptor.vl().getType()};
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());

    ensureIntrinsicDeclared(rewriter, module, intrinsicName, voidType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>( op, TypeRange{}, SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.value(), vecPtr, adaptor.vl()});

    return success();
  }
};

struct SpatzVFAddVVLowering : public ConvertOpToLLVMPattern<spatz::VFAddVVOp> {
  
  using ConvertOpToLLVMPattern<spatz::VFAddVVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFAddVVOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    
    std::string mangledName = getVectorMangledName(vecTy.getElementType());
    std::string intrinsicName = "llvm.riscv.vfadd." + mangledName + "." + mangledName + ".i32";

    SmallVector<Type, 4> argTypes = {adaptor.passthru().getType(), adaptor.lhs().getType(), adaptor.rhs().getType(), adaptor.vl().getType()};
    
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>( op, TypeRange{llvmResultType}, SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.rhs(), adaptor.vl()});
        
    return success();
  }   
};

struct SpatzVAddVVLowering : public ConvertOpToLLVMPattern<spatz::VAddVVOp> {
    
  using ConvertOpToLLVMPattern<spatz::VAddVVOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(spatz::VAddVVOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
      
      Type llvmResultType = typeConverter->convertType(op.getResult().getType());
      auto vecTy = op.getResult().getType().cast<VectorType>();
                                  
      ModuleOp module = op->getParentOfType<ModuleOp>();
                                  
      std::string mangledName = getVectorMangledName(vecTy.getElementType());
      std::string intrinsicName = "llvm.riscv.vadd." + mangledName + "." + mangledName + ".i32";
                                  
      SmallVector<Type, 4> argTypes = {adaptor.passthru().getType(), adaptor.lhs().getType(), adaptor.rhs().getType(), adaptor.vl().getType()};
      
      ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{llvmResultType}, SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
          ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.rhs(), adaptor.vl()});

      return success();
    }
};

} // namespace

namespace mlir {
void populateSpatzToLLVMConversionPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  
  patterns.add<SpatzVleLowering, SpatzVseLowering, SpatzVFAddVVLowering, SpatzVAddVVLowering>(converter);

}
} // namespace mlir

namespace {

  struct ConvertSpatzToLLVMPass : public ConvertSpatzToLLVMBase<ConvertSpatzToLLVMPass> {
    
    void runOnOperation() override {
      
      LLVMConversionTarget target(getContext());
      target.addLegalDialect<LLVM::LLVMDialect>();

      target.addDynamicallyLegalOp<spatz::VLEOp>([](spatz::VLEOp op) {
        return op.base().getType().isa<LLVM::LLVMPointerType>();
      });
      target.addDynamicallyLegalOp<spatz::VSEOp>([](spatz::VSEOp op) {
        return op.base().getType().isa<LLVM::LLVMPointerType>();
      });

      LLVMTypeConverter typeConverter(&getContext());
      RewritePatternSet patterns(&getContext());
      populateSpatzToLLVMConversionPatterns(typeConverter, patterns);

      if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) signalPassFailure();
    }
  };
} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertSpatzToLLVMPass() {
  return std::make_unique<ConvertSpatzToLLVMPass>();
}
} // namespace mlir