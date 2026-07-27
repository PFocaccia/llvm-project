#include "../PassDetail.h"
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

static std::string getVectorMangledName(Type elemTy, unsigned numElements) {
  
  std::string prefix = "nxv" + std::to_string(numElements);
  
  if (elemTy.isInteger(8))  return prefix + "i8";
  if (elemTy.isInteger(16)) return prefix + "i16";
  if (elemTy.isInteger(32)) return prefix + "i32";
  if (elemTy.isInteger(64)) return prefix + "i64";
  if (elemTy.isF16())       return prefix + "f16";
  if (elemTy.isBF16())      return prefix + "bf16";
  if (elemTy.isF32())       return prefix + "f32";
  if (elemTy.isF64())       return prefix + "f64";
  
  return prefix + "unknown";
}

static std::string getScalarMangledName(Type elemTy) {
  if (elemTy.isInteger(8))  return "i8";
  if (elemTy.isInteger(16)) return "i16";
  if (elemTy.isInteger(32)) return "i32";
  if (elemTy.isInteger(64)) return "i64";
  if (elemTy.isF16())       return "f16";
  if (elemTy.isBF16())      return "bf16";
  if (elemTy.isF32())       return "f32";
  if (elemTy.isF64())       return "f64";
  return "unknown";
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
    
    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];

    ModuleOp module = op->getParentOfType<ModuleOp>();
    std::string mangledName = getVectorMangledName(memRefType.getElementType(), numElements);
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
    
    auto vecTy = adaptor.value().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];

    std::string mangledName = getVectorMangledName(memRefType.getElementType(), numElements);
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
    
    Type llvmResultType =typeConverter->convertType( op.getResult().getType());

    if (!llvmResultType) return failure();

    ModuleOp module = op->getParentOfType<ModuleOp>();

    if (!module) return failure();

    auto vecTy = op.getResult().getType().dyn_cast<VectorType>();

    if (!vecTy) return failure();

    unsigned numElements = vecTy.getShape()[0];

    std::string mangledName = getVectorMangledName( vecTy.getElementType(), numElements);
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
      
      unsigned numElements = vecTy.getShape()[0];
      std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);
      std::string intrinsicName = "llvm.riscv.vadd." + mangledName + "." + mangledName + ".i32";
                                  
      SmallVector<Type, 4> argTypes = {adaptor.passthru().getType(), adaptor.lhs().getType(), adaptor.rhs().getType(), adaptor.vl().getType()};
      ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{llvmResultType}, SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
          ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.rhs(), adaptor.vl()});

      return success();
    }
};

struct SpatzVMulVFLowering : public ConvertOpToLLVMPattern<spatz::VMulVFOp> {
  using ConvertOpToLLVMPattern<spatz::VMulVFOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VMulVFOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);
    std::string scalarMangled = getScalarMangledName(op.scalar().getType());
    std::string intrinsicName = "llvm.riscv.vmul." + mangledName + "." + scalarMangled + ".i32";

    SmallVector<Type, 4> argTypes = {adaptor.passthru().getType(), adaptor.lhs().getType(), adaptor.scalar().getType(), adaptor.vl().getType()};
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{llvmResultType}, SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.scalar(), adaptor.vl()});

    return success();
  }
};

struct SpatzVFMulVFLowering : public ConvertOpToLLVMPattern<spatz::VFMulVFOp> {
  using ConvertOpToLLVMPattern<spatz::VFMulVFOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFMulVFOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);
    std::string scalarMangled = getScalarMangledName(op.scalar().getType());
    std::string intrinsicName = "llvm.riscv.vfmul." + mangledName + "." + scalarMangled + ".i32";

    SmallVector<Type, 4> argTypes = {adaptor.passthru().getType(), adaptor.lhs().getType(), adaptor.scalar().getType(), adaptor.vl().getType()};
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{llvmResultType}, SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.scalar(), adaptor.vl()});

    return success();
  }
};

struct SpatzVFMvVFLowering : public ConvertOpToLLVMPattern<spatz::VFMvVFOp> {
  using ConvertOpToLLVMPattern<spatz::VFMvVFOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFMvVFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    if (!llvmResultType || !llvmResultType.isa<VectorType>())
      return failure();

    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledVec = getVectorMangledName(vecTy.getElementType(), numElements);

    std::string intrinsicName = "llvm.riscv.vfmv.v.f." + mangledVec + ".i32";

    SmallVector<Type, 3> argTypes = {
      adaptor.passthru().getType(),
      adaptor.scalar().getType(),
      adaptor.vl().getType()
    };
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.scalar(), adaptor.vl()});

    return success();
  }
};

struct SpatzVFMaxVVLowering : public ConvertOpToLLVMPattern<spatz::VFMaxVVOp> {
  using ConvertOpToLLVMPattern<spatz::VFMaxVVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFMaxVVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);
    std::string intrinsicName = "llvm.riscv.vfmax." + mangledName + "." + mangledName + ".i32";

    SmallVector<Type, 4> argTypes = {
      adaptor.passthru().getType(),
      adaptor.lhs().getType(),
      adaptor.rhs().getType(),
      adaptor.vl().getType()
    };
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.rhs(), adaptor.vl()});
    return success();
  }
};

struct SpatzVFSubVVLowering : public ConvertOpToLLVMPattern<spatz::VFSubVVOp> {
  using ConvertOpToLLVMPattern<spatz::VFSubVVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFSubVVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);
    std::string intrinsicName = "llvm.riscv.vfsub." + mangledName + "." + mangledName + ".i32";

    SmallVector<Type, 4> argTypes = {
      adaptor.passthru().getType(),
      adaptor.lhs().getType(),
      adaptor.rhs().getType(),
      adaptor.vl().getType()
    };
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.rhs(), adaptor.vl()});
    return success();
  }
};

struct SpatzVFMulVVLowering : public ConvertOpToLLVMPattern<spatz::VFMulVVOp> {
  using ConvertOpToLLVMPattern<spatz::VFMulVVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFMulVVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);
    std::string intrinsicName = "llvm.riscv.vfmul." + mangledName + "." + mangledName + ".i32";

    SmallVector<Type, 4> argTypes = {
      adaptor.passthru().getType(),
      adaptor.lhs().getType(),
      adaptor.rhs().getType(),
      adaptor.vl().getType()
    };
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.rhs(), adaptor.vl()});
    return success();
  }
};

struct SpatzVFDivVVLowering : public ConvertOpToLLVMPattern<spatz::VFDivVVOp> {
  using ConvertOpToLLVMPattern<spatz::VFDivVVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFDivVVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);
    std::string intrinsicName = "llvm.riscv.vfdiv." + mangledName + "." + mangledName + ".i32";

    SmallVector<Type, 4> argTypes = {
      adaptor.passthru().getType(),
      adaptor.lhs().getType(),
      adaptor.rhs().getType(),
      adaptor.vl().getType()
    };
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.rhs(), adaptor.vl()});
    return success();
  }
};

struct SpatzVFMaccVFLowering : public ConvertOpToLLVMPattern<spatz::VFMaccVFOp> {
  using ConvertOpToLLVMPattern<spatz::VFMaccVFOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFMaccVFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string vecMangled = getVectorMangledName(vecTy.getElementType(), numElements);
    std::string scalarMangled = getScalarMangledName(op.scalar().getType());

    std::string intrinsicName = "llvm.riscv.vfmacc." + vecMangled + "." + scalarMangled + ".i32";

    auto i32Type = IntegerType::get(rewriter.getContext(), 32);
    Value policyTA = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), 
        i32Type, 
        rewriter.getI32IntegerAttr(3)
    );

    SmallVector<Type, 5> argTypes = { adaptor.passthru().getType(), adaptor.scalar().getType(), adaptor.src().getType(), adaptor.vl().getType(), policyTA.getType()};

    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{
            adaptor.passthru(),
            adaptor.scalar(),
            adaptor.src(),
            adaptor.vl(),
            policyTA     
        });

    return success();
  }
};

struct SpatzVFCvtRtzXUFVLowering : public ConvertOpToLLVMPattern<spatz::VFCvtRtzXUFVOp> {
  using ConvertOpToLLVMPattern<spatz::VFCvtRtzXUFVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFCvtRtzXUFVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto outVecTy = op.getResult().getType().cast<VectorType>();
    auto inVecTy  = op.src().getType().cast<VectorType>();

    unsigned numElements = outVecTy.getShape()[0];
    std::string outMangled = getVectorMangledName(outVecTy.getElementType(), numElements);
    std::string inMangled  = getVectorMangledName(inVecTy.getElementType(), numElements);

    std::string intrinsicName = "llvm.riscv.vfcvt.rtz.xu.f.v." + outMangled + "." + inMangled + ".i32";

    SmallVector<Type, 3> argTypes = {
      adaptor.passthru().getType(),
      adaptor.src().getType(),
      adaptor.vl().getType()
    };
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.src(), adaptor.vl()});
    return success();
  }
};

struct SpatzVFCvtRtzXFVLowering : public ConvertOpToLLVMPattern<spatz::VFCvtRtzXFVOp> {
  using ConvertOpToLLVMPattern<spatz::VFCvtRtzXFVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFCvtRtzXFVOp op,OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();
    
    if (!module) return failure();

    auto outVecTy = op.getResult().getType().cast<VectorType>();

    auto inVecTy = op.src().getType().cast<VectorType>();

    unsigned numElements = outVecTy.getShape()[0];

    std::string outMangled = getVectorMangledName(outVecTy.getElementType(), numElements);
    std::string inMangled = getVectorMangledName(inVecTy.getElementType(), numElements);

    std::string intrinsicName = "llvm.riscv.vfcvt.rtz.x.f.v." + outMangled + "." + inMangled + ".i32";

    SmallVector<Type, 3> argTypes = {adaptor.passthru().getType(), adaptor.src().getType(), adaptor.vl().getType()};

    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op,
        TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.src(), adaptor.vl()});
    
    return success();
  }
};

struct SpatzVBitcastLowering : public ConvertOpToLLVMPattern<spatz::VBitcastOp> {
  using ConvertOpToLLVMPattern<spatz::VBitcastOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VBitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op,
        typeConverter->convertType(op.getResult().getType()),
        adaptor.input());
    return success();
  }
};

struct SpatzVFSubVFLowering : public ConvertOpToLLVMPattern<spatz::VFSubVFOp> {
  using ConvertOpToLLVMPattern<spatz::VFSubVFOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFSubVFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto vecTy = op.getResult().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);
    std::string scalarMangled = getScalarMangledName(op.scalar().getType());
    
    std::string intrinsicName = "llvm.riscv.vfsub." + mangledName + "." + scalarMangled + ".i32";

    SmallVector<Type, 4> argTypes = {
      adaptor.passthru().getType(),
      adaptor.lhs().getType(),
      adaptor.scalar().getType(),
      adaptor.vl().getType()
    };
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.passthru(), adaptor.lhs(), adaptor.scalar(), adaptor.vl()});
    return success();
  }
};

struct SpatzVFredUSumOpLowering : public ConvertOpToLLVMPattern<spatz::VFredUSumOp> {
  using ConvertOpToLLVMPattern<spatz::VFredUSumOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFredUSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto inVecTy = op.vector().getType().cast<VectorType>();
    Type elemTy = inVecTy.getElementType();
    unsigned numElementsIn = inVecTy.getShape()[0];
    std::string inMangled = getVectorMangledName(elemTy, numElementsIn);

    unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
    unsigned m1Elements = 64 / bitWidth;

    auto m1VecTy = VectorType::get({m1Elements}, elemTy, {true}); 
    Type llvmM1Type = typeConverter->convertType(m1VecTy);
    std::string outMangled = getVectorMangledName(elemTy, m1Elements);

    std::string intrinsicName = "llvm.riscv.vfredusum." + outMangled + "." + inMangled + ".i32";

    auto toM1Type = [&](Value val) -> Value {
      if (val.getType() == llvmM1Type) return val;
      
      if (!val.getType().isa<VectorType>()) {
          Value undef = rewriter.create<LLVM::UndefOp>(loc, llvmM1Type);
          Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
          return rewriter.create<LLVM::InsertElementOp>(loc, llvmM1Type, undef, val, zero);
      }
      
      Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value extracted = rewriter.create<LLVM::ExtractElementOp>(loc, val, zero);
      Value undef = rewriter.create<LLVM::UndefOp>(loc, llvmM1Type);
      return rewriter.create<LLVM::InsertElementOp>(loc, llvmM1Type, undef, extracted, zero);
    };

    Value m1Passthru = toM1Type(adaptor.passthru());
    Value m1Scalar = toM1Type(adaptor.scalar());

    SmallVector<Type, 4> argTypes = {
      llvmM1Type,                 
      adaptor.vector().getType(), 
      llvmM1Type,                 
      adaptor.vl().getType()      
    };
    
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmM1Type, argTypes);

    Value callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{llvmM1Type},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{m1Passthru, adaptor.vector(), m1Scalar, adaptor.vl()}).getResult(0);

    if (llvmResultType == llvmM1Type) {
        rewriter.replaceOp(op, callOp);
    } else if (!llvmResultType.isa<VectorType>()) {
        Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(op, callOp, zero);
    } else {
        Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        Value extracted = rewriter.create<LLVM::ExtractElementOp>(loc, callOp, zero);
        Value undefOut = rewriter.create<LLVM::UndefOp>(loc, llvmResultType);
        rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(op, llvmResultType, undefOut, extracted, zero);
    }

    return success();
  }
};

struct SpatzVFMvFSLowering : public ConvertOpToLLVMPattern<spatz::VFMvFSOp> {
  using ConvertOpToLLVMPattern<spatz::VFMvFSOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(spatz::VFMvFSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type llvmResultType = typeConverter->convertType(op.getResult().getType());
    if (!llvmResultType || !llvmResultType.isa<FloatType>())
      return failure();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto vecTy = op.vector().getType().cast<VectorType>();
    unsigned numElements = vecTy.getShape()[0];
    std::string mangledName = getVectorMangledName(vecTy.getElementType(), numElements);

    std::string intrinsicName = "llvm.riscv.vfmv.f.s." + mangledName;

    SmallVector<Type, 1> argTypes = {adaptor.vector().getType()};
    ensureIntrinsicDeclared(rewriter, module, intrinsicName, llvmResultType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{llvmResultType},
        SymbolRefAttr::get(rewriter.getContext(), intrinsicName),
        ValueRange{adaptor.vector()});

    return success();
  }
};

} // namespace

namespace mlir {
void populateSpatzToLLVMConversionPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
    SpatzVleLowering, 
    SpatzVseLowering, 
    SpatzVFAddVVLowering,
    SpatzVAddVVLowering,
    SpatzVMulVFLowering,
    SpatzVFMulVFLowering,
    SpatzVFMvVFLowering,
    SpatzVFMaxVVLowering,
    SpatzVFSubVVLowering,
    SpatzVFMulVVLowering,
    SpatzVFDivVVLowering,
    SpatzVFMaccVFLowering,        
    SpatzVFCvtRtzXUFVLowering,
    SpatzVFCvtRtzXFVLowering,    
    SpatzVBitcastLowering,
    SpatzVFSubVFLowering,
    SpatzVFredUSumOpLowering,
    SpatzVFMvFSLowering >(converter);
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