//===- QuadrilateroDialect.h - MLIR Dialect for Quadrilatero ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Quadrilatero dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUADRILATERO_QUADRILATERODIALECT_H_
#define MLIR_DIALECT_QUADRILATERO_QUADRILATERODIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Quadrilatero/QuadrilateroDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Quadrilatero/QuadrilateroTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Quadrilatero/Quadrilatero.h.inc"

#endif // MLIR_DIALECT_QUADRILATERO_QUADRILATERODIALECT_H_
