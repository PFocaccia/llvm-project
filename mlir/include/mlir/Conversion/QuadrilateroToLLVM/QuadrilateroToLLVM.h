//===- QuadrilateroToLLVM.h - Convert Quadrilatero to LLVM ------*- C++ -*-===//
#ifndef MLIR_CONVERSION_QUADRILATEROTOLLVM_QUADRILATEROTOLLVM_H
#define MLIR_CONVERSION_QUADRILATEROTOLLVM_QUADRILATEROTOLLVM_H

#include <memory>

namespace mlir {
class Pass;

// Hook per TableGen
#define GEN_PASS_DECL_CONVERTQUADRILATEROTOLLVM
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createConvertQuadrilateroToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_QUADRILATEROTOLLVM_QUADRILATEROTOLLVM_H