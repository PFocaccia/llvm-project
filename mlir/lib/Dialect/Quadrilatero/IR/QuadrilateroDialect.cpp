//===- QuadrilateroDialect.cpp - MLIR Quadrilatero ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Quadrilatero dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quadrilatero/QuadrilateroDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "mlir/Dialect/Quadrilatero/QuadrilateroDialect.cpp.inc"

void quadrilatero::QuadrilateroDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quadrilatero/Quadrilatero.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Quadrilatero/QuadrilateroTypes.cpp.inc"
      >();
}

static LogicalResult verifyImm2Range(Operation *op) {
  auto imm2 = op->getAttrOfType<IntegerAttr>("imm2");
  if (!imm2)
    return op->emitOpError("requires 'imm2' attribute");

  int64_t value = imm2.getInt();
  if (value < 0 || value > 3)
    return op->emitOpError("requires 'imm2' in range [0, 3], got ") << value;

  return success();
}

LogicalResult quadrilatero::McfgmOp::verify() { return verifyImm2Range(*this); }

LogicalResult quadrilatero::McfgnOp::verify() { return verifyImm2Range(*this); }

#define GET_OP_CLASSES
#include "mlir/Dialect/Quadrilatero/Quadrilatero.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Quadrilatero/QuadrilateroTypes.cpp.inc"
