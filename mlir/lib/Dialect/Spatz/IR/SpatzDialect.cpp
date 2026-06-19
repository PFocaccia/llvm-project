//===- SpatzDialect.cpp - MLIR Spatz ops -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Spatz/SpatzDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"             
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;

#include "mlir/Dialect/Spatz/SpatzDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Spatz/Spatz.cpp.inc"

void spatz::SpatzDialect::initialize() { 
addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Spatz/Spatz.cpp.inc"
      >();
}

static LogicalResult verifyIntegerVectorElemType(Type elemType, Operation *op, StringRef opName) {
  
  auto intType = elemType.dyn_cast<IntegerType>();
  
  if (!intType || !intType.isSignless()) return op->emitOpError(opName) << " requires signless integer vectors";

  unsigned width = intType.getWidth();
  if (width != 8 && width != 16 && width != 32) return op->emitOpError(opName) << " supports only i8/i16/i32 element types";

  return success();
}

static LogicalResult verifyFloatVectorElemType(Type elemType, Operation *op, StringRef opName) {
  
  if (!(elemType.isF16() || elemType.isF32() || elemType.isBF16())) return op->emitOpError(opName) << " supports only f16/f32/bf16 element types";

  return success();
}

static LogicalResult verifySupportedVectorElemType(Type elemType, Operation *op, StringRef opName) {
  
  if (elemType.isa<IntegerType>()) return verifyIntegerVectorElemType(elemType, op, opName);
  if (elemType.isa<FloatType>()) return verifyFloatVectorElemType(elemType, op, opName);

  return op->emitOpError(opName) << " unsupported vector element type";
}

static LogicalResult verifyMemRefIndices(Operation *op, MemRefType memrefType, ValueRange indices, StringRef opName) {
  
  if ((int64_t)indices.size() != memrefType.getRank()) return op->emitOpError(opName) << " incorrect number of indices for memref";
  return success();

}

LogicalResult spatz::VAddVVOp::verify() {
  auto vecTy = getResult().getType().dyn_cast<VectorType>();
  
  if (!vecTy) return emitOpError("requires vector result type");

  return verifyIntegerVectorElemType(vecTy.getElementType(), *this, "vadd_vv");
}

LogicalResult spatz::VFAddVVOp::verify() {
  auto vecTy = getResult().getType().dyn_cast<VectorType>();
  
  if (!vecTy) return emitOpError("requires vector result type");

  return verifyFloatVectorElemType(vecTy.getElementType(), *this, "vfadd_vv");
}

LogicalResult spatz::VLEOp::verify() {
  
  if (base().getType().isa<LLVM::LLVMPointerType>()) {
    if (!indices().empty())
      return emitOpError("indices must be empty for LLVM pointer base");
    return success();
  }

  auto memrefType = base().getType().dyn_cast<MemRefType>();
  
  if (!memrefType) return emitOpError("requires memref or LLVM pointer base operand");

  auto vecTy = getResult().getType().dyn_cast<VectorType>();
  
  if (!vecTy || vecTy.getRank() != 1) return emitOpError("requires 1-D vector result type");

  if (failed(verifyMemRefIndices(*this, memrefType, indices(), "vle"))) return failure();

  if (memrefType.getElementType() != vecTy.getElementType())
    return emitOpError("element type mismatch between memref and vector");

  return verifySupportedVectorElemType(vecTy.getElementType(), *this, "vle");
}

LogicalResult spatz::VSEOp::verify() {

  if (base().getType().isa<LLVM::LLVMPointerType>()) {
    if (!indices().empty())
      return emitOpError("indices must be empty for LLVM pointer base");
    return success();
  }

  auto memrefType = base().getType().dyn_cast<MemRefType>();
  
  if (!memrefType) return emitOpError("requires memref or LLVM pointer base operand");

  auto vecTy = value().getType().dyn_cast<VectorType>();
  
  if (!vecTy || vecTy.getRank() != 1) return emitOpError("requires 1-D vector value type");

  if (failed(verifyMemRefIndices(*this, memrefType, indices(), "vse"))) return failure();

  if (memrefType.getElementType() != vecTy.getElementType()) return emitOpError("element type mismatch between memref and vector");

  return verifySupportedVectorElemType(vecTy.getElementType(), *this, "vse");
}

LogicalResult spatz::VMulVFOp::verify() {
  auto vecTy = getResult().getType().dyn_cast<VectorType>();
  if (!vecTy) return emitOpError("requires vector result type");

  if (vecTy.getRank() != 1) return emitOpError("requires 1-D vector result type");

  if (failed(verifyIntegerVectorElemType(vecTy.getElementType(), *this, "vmul_vf"))) return failure();

  Type scalarTy = scalar().getType();
  if (!scalarTy.isSignlessInteger()) return emitOpError("scalar must be signless integer");

  return success();
}

LogicalResult spatz::VFMulVFOp::verify() {
  auto vecTy = getResult().getType().dyn_cast<VectorType>();
  if (!vecTy) return emitOpError("requires vector result type");

  if (vecTy.getRank() != 1) return emitOpError("requires 1-D vector result type");

  if (failed(verifyFloatVectorElemType(vecTy.getElementType(), *this, "vfmul_vf"))) return failure();

  Type scalarTy = scalar().getType();
  if (!scalarTy.isa<FloatType>()) return emitOpError("scalar must be floating-point");

  return success();
}

LogicalResult spatz::MatrixAddOp::verify() {
  
  auto accType = accMatrix().getType().dyn_cast<MemRefType>();
  auto tmpType = tmpMatrix().getType().dyn_cast<MemRefType>();
  
  if (!accType || !tmpType) return emitOpError("requires memref operands");

  if (accType.getRank() != 2 || tmpType.getRank() != 2) return emitOpError("requires rank-2 memrefs");

  if (accType.getElementType() != tmpType.getElementType()) return emitOpError("requires matching element types");

  auto vecLenAttr = (*this)->getAttrOfType<IntegerAttr>("vecLen");
  
  if (!vecLenAttr || vecLenAttr.getInt() <= 0) return emitOpError("requires positive 'vecLen' attribute");

  return success();
}

LogicalResult spatz::MatrixVectorAddOp::verify() {
  auto accType = accMatrix().getType().dyn_cast<MemRefType>();
  auto vecType = addVector().getType().dyn_cast<MemRefType>();
  
  if (!accType || !vecType) return emitOpError("requires memref operands");
  if (accType.getRank() != 2) return emitOpError("requires rank-2 memref for matrix");
  if (vecType.getRank() != 1) return emitOpError("requires rank-1 memref for vector");
  if (accType.getElementType() != vecType.getElementType()) 
    return emitOpError("requires matching element types between matrix and vector");

  auto vecLenAttr = (*this)->getAttrOfType<IntegerAttr>("vecLen");
  if (!vecLenAttr || vecLenAttr.getInt() <= 0) return emitOpError("requires positive 'vecLen' attribute");

  return success();
}

LogicalResult spatz::MatrixScalarMulOp::verify() {
  auto accType = accMatrix().getType().dyn_cast<MemRefType>();
  if (!accType) return emitOpError("requires memref for matrix");
  if (accType.getRank() != 2) return emitOpError("requires rank-2 memref for matrix");

  Type scalarTy = scalar().getType();
  if (accType.getElementType() != scalarTy) 
    return emitOpError("scalar type must match matrix element type");

  auto vecLenAttr = (*this)->getAttrOfType<IntegerAttr>("vecLen");
  if (!vecLenAttr || vecLenAttr.getInt() <= 0) return emitOpError("requires positive 'vecLen' attribute");

  return success();
}
