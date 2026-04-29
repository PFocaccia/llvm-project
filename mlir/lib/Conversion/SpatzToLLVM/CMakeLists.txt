add_mlir_conversion_library(MLIRSpatzToLLVM
  SpatzToLLVM.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRSpatz
  MLIRPass
  MLIRTransforms
)