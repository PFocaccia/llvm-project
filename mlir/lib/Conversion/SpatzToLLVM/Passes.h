#ifndef MLIR_CONVERSION_SPATZTOLLVM_PASSES_H
#define MLIR_CONVERSION_SPATZTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace spatz {

std::unique_ptr<Pass> createConvertSpatzToLLVMPass();

void registerSpatzToLLVMPass();

} // namespace spatz
} // namespace mlir

#endif