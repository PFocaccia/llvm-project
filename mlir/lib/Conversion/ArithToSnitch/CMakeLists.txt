add_mlir_conversion_library(MLIRArithToSnitch
  ArithToSnitch.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRFunc
  MLIRIR
  MLIRPass
  MLIRTransforms
)