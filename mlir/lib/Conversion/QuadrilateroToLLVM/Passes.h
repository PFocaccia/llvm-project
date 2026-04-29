#ifndef MLIR_CONVERSION_QUADRILATEROTOLLVM_PASSES_H
#define MLIR_CONVERSION_QUADRILATEROTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace Quadrilatero {

std::unique_ptr<Pass> createConvertQuadrilateroToLLVMPass();

void registerQuadrilateroToLLVMPass();

} // namespace quadrilatero
} // namespace mlir

#endif