add_subdirectory(Utils)

add_mlir_library(MLIRTransforms
  Canonicalizer.cpp
  ControlFlowSink.cpp
  CSE.cpp
  Inliner.cpp
  LocationSnapshot.cpp
  LoopInvariantCodeMotion.cpp
  LowerLinalgMatmulToQuadrilatero.cpp
  LowerLinalgMatmulToQuadrilateroNoTranspose.cpp
  LowerSpatzMatrixAdd.cpp
  OpStats.cpp
  SCCP.cpp
  StripDebugInfo.cpp
  SymbolDCE.cpp
  SymbolPrivatize.cpp
  ViewOpGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithmetic
  MLIRCopyOpInterface
  MLIRFunc
  MLIRLinalg
  MLIRLoopLikeInterface
  MLIRMemRef
  MLIRPass
  MLIRQuadrilatero
  MLIRSpatz
  MLIRSCF
  MLIRSupport
  MLIRTransformUtils
  )
