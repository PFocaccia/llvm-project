add_mlir_dialect_library(MLIRQuadrilatero
  QuadrilateroDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Quadrilatero

  DEPENDS
  MLIRQuadrilateroIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSideEffectInterfaces
  )
