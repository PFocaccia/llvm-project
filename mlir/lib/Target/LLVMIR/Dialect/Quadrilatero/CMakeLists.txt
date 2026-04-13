add_mlir_translation_library(MLIRQuadrilateroToLLVMIRTranslation
  QuadrilateroToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRQuadrilatero
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
