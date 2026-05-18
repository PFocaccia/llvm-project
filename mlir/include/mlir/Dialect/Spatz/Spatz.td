//===-- Spatz.td - Spatz dialect ops ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPATZ
#define SPATZ

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Spatz dialect definition.
//===----------------------------------------------------------------------===//

def Spatz_Dialect : Dialect {
  let name = "spatz";
  let cppNamespace = "::mlir::spatz";
  let description = [{
    The Spatz dialect models vector operations that are intentionally restricted
    to instructions supported by the Spatz hardware decoder.
  }];
}

class Spatz_Op<string mnemonic, list<Trait> traits = []>
    : Op<Spatz_Dialect, mnemonic, traits>;

// Maps to llvm.riscv.vadd.* (unmasked form).
def VAddVVOp : Spatz_Op<"vadd_vv", [AllTypesMatch<["passthru", "lhs", "rhs", "res"]>]> {
  let summary = "Spatz integer vector add (vv)";
  let arguments = (ins
    VectorOf<[AnySignlessInteger]>:$passthru,
    VectorOf<[AnySignlessInteger]>:$lhs,
    VectorOf<[AnySignlessInteger]>:$rhs,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnySignlessInteger]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $rhs `,` $vl attr-dict `:` type($passthru) `,` type($vl)";
  let hasVerifier = 1;
}

// Maps to llvm.riscv.vfadd.* (unmasked form).
def VFAddVVOp : Spatz_Op<"vfadd_vv", [AllTypesMatch<["passthru", "lhs", "rhs", "res"]>]> {
  let summary = "Spatz floating-point vector add (vv)";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    VectorOf<[AnyFloat]>:$lhs,
    VectorOf<[AnyFloat]>:$rhs,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $rhs `,` $vl attr-dict `:` type($passthru) `,` type($vl)";
  let hasVerifier = 1;
}

def VLEOp : Spatz_Op<"vle"> {
  let summary = "Spatz vector load (vle*.v) without masks";
  let arguments = (ins AnyType:$base, Variadic<Index>:$indices,
                       AnySignlessIntegerOrIndex:$vl);
  let results = (outs VectorOf<[AnyType]>:$res);
  let assemblyFormat = "$base `[` $indices `]` `,` $vl attr-dict `:` type($base) `,` type($res) `,` type($vl)";
  let hasVerifier = 1;
}

def VSEOp : Spatz_Op<"vse"> {
  let summary = "Spatz vector store (vse*.v) without masks";
  // Cambiato AnyMemRef in AnyType per supportare !llvm.ptr
  let arguments = (ins VectorOf<[AnyType]>:$value, AnyType:$base,
                       Variadic<Index>:$indices, AnySignlessIntegerOrIndex:$vl);
  let assemblyFormat = "$value `,` $base `[` $indices `]` `,` $vl attr-dict `:` type($value) `,` type($base) `,` type($vl)";
  let hasVerifier = 1;
}


def MatrixAddOp : Spatz_Op<"matrix_add"> {
  let summary = "Add two matrices in TCDM using Spatz";
  let arguments = (ins AnyMemRef:$accMatrix, AnyMemRef:$tmpMatrix,
                       Index:$rows, Index:$cols, I64Attr:$vecLen, 
                       I32Attr:$dataType); // <-- Aggiunto il tipo di dato per propagare l'fp8/fp32
  let assemblyFormat = "$accMatrix `,` $tmpMatrix `,` $rows `,` $cols attr-dict `:` type($accMatrix) `,` type($tmpMatrix) `,` type($rows) `,` type($cols)";
  let hasVerifier = 1;
}

#endif // SPATZ
