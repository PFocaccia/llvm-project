add_mlir_conversion_library(MLIRQuadrilateroToLLVM
  QuadrilateroToLLVM.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRQuadrilatero
  MLIRPass
  MLIRTransforms
)