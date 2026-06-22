#ifndef MLIR_CONVERSION_ARITHTOSNITCH_ARITHTOSNITCH_H
#define MLIR_CONVERSION_ARITHTOSNITCH_ARITHTOSNITCH_H

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOSNITCH
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createConvertArithToSnitchPass();

} // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOSNITCH_ARITHTOSNITCH_H