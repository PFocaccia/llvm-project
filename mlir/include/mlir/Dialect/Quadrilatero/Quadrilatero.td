//===-- Quadrilatero.td - Quadrilatero dialect ops -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MLIR dialect for the Quadrilatero matrix extension.
// The ops are 1:1 mappings to the corresponding RISC-V intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef QUADRILATERO
#define QUADRILATERO

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Quadrilatero dialect definition.
//===----------------------------------------------------------------------===//

def Quadrilatero_Dialect : Dialect {
  let name = "quadrilatero";
  let cppNamespace = "::mlir::quadrilatero";
  let description = [{
    The Quadrilatero dialect models a custom RISC-V matrix coprocessor
    extension. The operations in this dialect map 1:1 to the
    `llvm.riscv.quadrilatero.*` intrinsics.
  }];
}

//===----------------------------------------------------------------------===//
// Quadrilatero type definitions.
//===----------------------------------------------------------------------===//

class Quadrilatero_Type<string name, string typeMnemonic>
    : TypeDef<Quadrilatero_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Quadrilatero_MatrixType : Quadrilatero_Type<"Matrix", "matrix"> {
  let summary = "Quadrilatero matrix register type";
}

def Quadrilatero_AccumulatorType : Quadrilatero_Type<"Accumulator", "acc"> {
  let summary = "Quadrilatero accumulator register type";
}

def Quadrilatero_RegType : AnyTypeOf<[Quadrilatero_MatrixType,
                                      Quadrilatero_AccumulatorType]>;

//===----------------------------------------------------------------------===//
// Quadrilatero op base.
//===----------------------------------------------------------------------===//

class Quadrilatero_Op<string mnemonic, list<Trait> traits = []>
    : Op<Quadrilatero_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Quadrilatero ops.
//===----------------------------------------------------------------------===//

// Matrix memory ops.

def MldLhsOp : Quadrilatero_Op<"mld_lhs"> {
  let summary = "matrix load (lhs)";
  let arguments = (ins Arg<LLVM_AnyPointer, "load base", [MemRead]>:$base,
                       AnyInteger:$stride);
  let results = (outs Quadrilatero_MatrixType:$res);
  let assemblyFormat = "$base `,` $stride attr-dict `:` type($base) `,` type($stride) `->` type($res)";
}

def MldRhsOp : Quadrilatero_Op<"mld_rhs"> {
  let summary = "matrix load (rhs)";
  let arguments = (ins Arg<LLVM_AnyPointer, "load base", [MemRead]>:$base,
                       AnyInteger:$stride);
  let results = (outs Quadrilatero_MatrixType:$res);
  let assemblyFormat = "$base `,` $stride attr-dict `:` type($base) `,` type($stride) `->` type($res)";
}

def MstOp : Quadrilatero_Op<"mst"> {
  let summary = "matrix store";
  let arguments = (ins Quadrilatero_MatrixType:$src,
                       Arg<LLVM_AnyPointer, "store base", [MemWrite]>:$base,
                       AnyInteger:$stride);
  let assemblyFormat = "$src `,` $base `,` $stride attr-dict `:` type($src) `,` type($base) `,` type($stride)";
}

// Matrix arithmetic ops.

def MmaccOp : Quadrilatero_Op<"mmacc", [NoSideEffect]> {
  let summary = "matrix multiply-accumulate";
  let arguments = (ins Quadrilatero_MatrixType:$lhs,
                       Quadrilatero_MatrixType:$rhs);
  let results = (outs Quadrilatero_AccumulatorType:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

def MzeroMOp : Quadrilatero_Op<"mzero_m", [NoSideEffect]> {
  let summary = "matrix zero (matrix register)";
  let results = (outs Quadrilatero_MatrixType:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def MzeroAOp : Quadrilatero_Op<"mzero_a", [NoSideEffect]> {
  let summary = "matrix zero (accumulator register)";
  let results = (outs Quadrilatero_AccumulatorType:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

// Matrix move ops.

def MmovMmOp : Quadrilatero_Op<"mmov_mm", [NoSideEffect]> {
  let summary = "matrix move (matrix -> matrix)";
  let arguments = (ins Quadrilatero_MatrixType:$src);
  let results = (outs Quadrilatero_MatrixType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

def MmovMaOp : Quadrilatero_Op<"mmov_ma", [NoSideEffect]> {
  let summary = "matrix move (matrix -> accumulator)";
  let arguments = (ins Quadrilatero_MatrixType:$src);
  let results = (outs Quadrilatero_AccumulatorType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

def MmovAmOp : Quadrilatero_Op<"mmov_am", [NoSideEffect]> {
  let summary = "matrix move (accumulator -> matrix)";
  let arguments = (ins Quadrilatero_AccumulatorType:$src);
  let results = (outs Quadrilatero_MatrixType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

def MmovAaOp : Quadrilatero_Op<"mmov_aa", [NoSideEffect]> {
  let summary = "matrix move (accumulator -> accumulator)";
  let arguments = (ins Quadrilatero_AccumulatorType:$src);
  let results = (outs Quadrilatero_AccumulatorType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

// Matrix configuration ops.

def McfgkOp : Quadrilatero_Op<"mcfgk", [AllTypesMatch<["rs1", "res"]>]> {
  let summary = "matrix config (k)";
  let arguments = (ins AnyInteger:$rs1);
  let results = (outs AnyInteger:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($rs1) `->` type($res)";
}

def McfgmOp : Quadrilatero_Op<"mcfgm", [AllTypesMatch<["rs1", "res"]>]> {
  let summary = "matrix config (m)";
  let arguments = (ins AnyInteger:$rs1, I32Attr:$imm2);
  let results = (outs AnyInteger:$res);
  let assemblyFormat = "$rs1 `,` $imm2 attr-dict `:` type($rs1) `->` type($res)";
  let hasVerifier = 1;
}

def McfgnOp : Quadrilatero_Op<"mcfgn", [AllTypesMatch<["rs1", "res"]>]> {
  let summary = "matrix config (n)";
  let arguments = (ins AnyInteger:$rs1, I32Attr:$imm2);
  let results = (outs AnyInteger:$res);
  let assemblyFormat = "$rs1 `,` $imm2 attr-dict `:` type($rs1) `->` type($res)";
  let hasVerifier = 1;
}

def MmacDtOp : Quadrilatero_Op<"mmac_dt"> {
  let summary = "matrix mac data types";
  let arguments = (ins I32Attr:$dtC, I32Attr:$dtA, I32Attr:$dtB);
  let assemblyFormat = "$dtC `,` $dtA `,` $dtB attr-dict";
}

def TcdmMatmulOp : Quadrilatero_Op<"tcdm_matmul"> {
  let summary = "highly optimized TCDM matmul micro-kernel";
  let arguments = (ins Arg<LLVM_AnyPointer, "A base", [MemRead]>:$addrA,
                       Arg<LLVM_AnyPointer, "B base", [MemRead]>:$addrB,
                       Arg<LLVM_AnyPointer, "C base", [MemWrite]>:$addrC,
                       AnyInteger:$m, AnyInteger:$n, AnyInteger:$k,
                       AnyInteger:$shift,
                       I32Attr:$dtC, I32Attr:$dtA, I32Attr:$dtB);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

#endif // QUADRILATERO
