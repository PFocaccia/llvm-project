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

def VMulVFOp : Spatz_Op<"vmul_vf", [AllTypesMatch<["passthru", "lhs", "res"]>]> {
  let summary = "Spatz integer vector multiply by scalar (vf)";
  let arguments = (ins
    VectorOf<[AnySignlessInteger]>:$passthru,
    VectorOf<[AnySignlessInteger]>:$lhs,
    AnySignlessInteger:$scalar,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnySignlessInteger]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $scalar `,` $vl attr-dict `:` type($passthru) `,` type($scalar) `,` type($vl)";
  let hasVerifier = 1;
}

def VFMulVFOp : Spatz_Op<"vfmul_vf", [AllTypesMatch<["passthru", "lhs", "res"]>]> {
  let summary = "Spatz floating-point vector multiply by scalar (vf)";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    VectorOf<[AnyFloat]>:$lhs,
    AnyFloat:$scalar,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $scalar `,` $vl attr-dict `:` type($passthru) `,` type($scalar) `,` type($vl)";
  let hasVerifier = 1;
}

// Maps to llvm.riscv.vfmax.* (unmasked form).
def VFMaxVVOp : Spatz_Op<"vfmax_vv", [AllTypesMatch<["passthru", "lhs", "rhs", "res"]>]> {
  let summary = "Spatz floating-point vector maximum (vv)";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    VectorOf<[AnyFloat]>:$lhs,
    VectorOf<[AnyFloat]>:$rhs,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $rhs `,` $vl attr-dict `:` type($passthru) `,` type($vl)";
}

// Maps to llvm.riscv.vfsub.* (unmasked form).
def VFSubVVOp : Spatz_Op<"vfsub_vv", [AllTypesMatch<["passthru", "lhs", "rhs", "res"]>]> {
  let summary = "Spatz floating-point vector subtract (vv)";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    VectorOf<[AnyFloat]>:$lhs,
    VectorOf<[AnyFloat]>:$rhs,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $rhs `,` $vl attr-dict `:` type($passthru) `,` type($vl)";
}

// Maps to llvm.riscv.vfmul.* (unmasked form).
def VFMulVVOp : Spatz_Op<"vfmul_vv", [AllTypesMatch<["passthru", "lhs", "rhs", "res"]>]> {
  let summary = "Spatz floating-point vector multiply (vv)";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    VectorOf<[AnyFloat]>:$lhs,
    VectorOf<[AnyFloat]>:$rhs,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $rhs `,` $vl attr-dict `:` type($passthru) `,` type($vl)";
}

// Maps to llvm.riscv.vfdiv.* (unmasked form).
def VFDivVVOp : Spatz_Op<"vfdiv_vv", [AllTypesMatch<["passthru", "lhs", "rhs", "res"]>]> {
  let summary = "Spatz floating-point vector divide (vv)";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    VectorOf<[AnyFloat]>:$lhs,
    VectorOf<[AnyFloat]>:$rhs,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $rhs `,` $vl attr-dict `:` type($passthru) `,` type($vl)";
}

// Spatz splat scalar into vector
def VFMvVFOp : Spatz_Op<"vfmv_v_f", [AllTypesMatch<["passthru", "res"]>]> {
  let summary = "Spatz splat scalar float into vector";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    AnyFloat:$scalar,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $scalar `,` $vl attr-dict `:` type($passthru) `,` type($scalar) `,` type($vl)";
}


// Maps to llvm.riscv.vfmacc.vf (unmasked form)
def VFMaccVFOp : Spatz_Op<"vfmacc_vf", [AllTypesMatch<["passthru", "src", "res"]>]> {
  let summary = "Spatz floating-point vector fused multiply-add with scalar";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,   // accumulatore (vettore)
    VectorOf<[AnyFloat]>:$src,        // vettore da moltiplicare per scalare
    AnyFloat:$scalar,                 // scalare (moltiplicatore)
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $src `,` $scalar `,` $vl attr-dict `:` type($passthru) `,` type($scalar) `,` type($vl)";
}

// Maps to llvm.riscv.vfcvt.rtz.xu.f.v (unmasked form)
def VFCvtRtzXUFVOp : Spatz_Op<"vfcvt_rtz_xu_f_v"> {
  let summary = "Spatz convert float vector to unsigned int vector with truncation toward zero";
  let arguments = (ins
    VectorOf<[AnySignlessInteger]>:$passthru,
    VectorOf<[AnyFloat]>:$src,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnySignlessInteger]>:$res);
  let assemblyFormat = "$passthru `,` $src `,` $vl attr-dict `:` type($passthru) `,` type($src) `,` type($vl) `,` type($res)";
}

// Maps to llvm.riscv.vfsub.* (vector-scalar form)
def VFSubVFOp : Spatz_Op<"vfsub_vf", [AllTypesMatch<["passthru", "lhs", "res"]>]> {
  let summary = "Spatz floating-point vector subtract by scalar (vf)";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    VectorOf<[AnyFloat]>:$lhs,
    AnyFloat:$scalar,
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $lhs `,` $scalar `,` $vl attr-dict `:` type($passthru) `,` type($scalar) `,` type($vl)";
}

// Maps to llvm.riscv.vfredusum.* (unordered reduction sum)
def VFredUSumOp : Spatz_Op<"vfredusum_vs", [AllTypesMatch<["passthru", "vector", "scalar", "res"]>]> {
  let summary = "Spatz floating-point vector unordered reduction sum";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$passthru,
    VectorOf<[AnyFloat]>:$vector,
    VectorOf<[AnyFloat]>:$scalar, 
    AnySignlessIntegerOrIndex:$vl
  );
  let results = (outs VectorOf<[AnyFloat]>:$res);
  let assemblyFormat = "$passthru `,` $vector `,` $scalar `,` $vl attr-dict `:` type($passthru) `,` type($vl)";
}

def VFMvFSOp : Spatz_Op<"vfmv_f_s"> {
  let summary = "Spatz extract first element of vector to scalar";
  let arguments = (ins
    VectorOf<[AnyFloat]>:$vector,       // vettore da cui leggere
    AnySignlessIntegerOrIndex:$vl       // lunghezza vettore (passata ma ignorata)
  );
  let results = (outs AnyFloat:$res);   // scalare estratto
  let assemblyFormat = "$vector `,` $vl attr-dict `:` type($vector) `,` type($vl) `->` type($res)";
}

// Generic bitcast between vectors of same total bit width
def VBitcastOp : Spatz_Op<"vbitcast"> {
  let summary = "Bitcast between vectors of same total bit width";
  let arguments = (ins AnyVector:$input);
  let results = (outs AnyVector:$res);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($res)";
}

def MatrixAddOp : Spatz_Op<"matrix_add"> {
  let summary = "Add two matrices in TCDM using Spatz";
  let arguments = (ins AnyMemRef:$accMatrix, AnyMemRef:$tmpMatrix,
                       Index:$rows, Index:$cols, I64Attr:$vecLen, 
                       I32Attr:$dataType); 
  let assemblyFormat = "$accMatrix `,` $tmpMatrix `,` $rows `,` $cols attr-dict `:` type($accMatrix) `,` type($tmpMatrix) `,` type($rows) `,` type($cols)";
  let hasVerifier = 1;
}

def MatrixVectorAddOp : Spatz_Op<"matrix_vector_add"> {
  let summary = "Add a 1D vector to all rows of a 2D matrix in TCDM using Spatz";
  let arguments = (ins AnyMemRef:$accMatrix, AnyMemRef:$addVector,
                       Index:$rows, Index:$cols, I64Attr:$vecLen, 
                       I32Attr:$dataType);
  let assemblyFormat = "$accMatrix `,` $addVector `,` $rows `,` $cols attr-dict `:` type($accMatrix) `,` type($addVector) `,` type($rows) `,` type($cols)";
  let hasVerifier = 1;
}

def MatrixScalarMulOp : Spatz_Op<"matrix_scalar_mul"> {
  let summary = "Multiply a 2D matrix by a scalar in TCDM using Spatz";
  let arguments = (ins AnyMemRef:$accMatrix, AnyType:$scalar,
                       Index:$rows, Index:$cols, I64Attr:$vecLen, 
                       I32Attr:$dataType);
  let assemblyFormat = "$accMatrix `,` $scalar `,` $rows `,` $cols attr-dict `:` type($accMatrix) `,` type($scalar) `,` type($rows) `,` type($cols)";
  let hasVerifier = 1;
}

def MatrixSubOp : Spatz_Op<"matrix_sub"> {
  let summary = "Subtract two matrices in TCDM using Spatz";
  let arguments = (ins AnyMemRef:$accMatrix, AnyMemRef:$tmpMatrix,
                       Index:$rows, Index:$cols, I64Attr:$vecLen, 
                       I32Attr:$dataType); 
  let assemblyFormat = "$accMatrix `,` $tmpMatrix `,` $rows `,` $cols attr-dict `:` type($accMatrix) `,` type($tmpMatrix) `,` type($rows) `,` type($cols)";
}

def MatrixExpOp : Spatz_Op<"matrix_exp"> {
  let summary = "Apply exponential to a matrix in TCDM using Spatz";
  let arguments = (ins AnyMemRef:$matrix,
                       Index:$rows, Index:$cols, I64Attr:$vecLen, 
                       I32Attr:$dataType);
  let assemblyFormat = "$matrix `,` $rows `,` $cols attr-dict `:` type($matrix) `,` type($rows) `,` type($cols)";
}

def MatrixExpReduceOp : Spatz_Op<"matrix_exp_reduce"> {
  let summary = "Apply exp to an entire matrix tile with optional bias, and reduce vertically into an array";
  let arguments = (ins
    AnyMemRef:$s_buf,
    AnyMemRef:$p_T,
    AnyMemRef:$m_val,
    AnyMemRef:$sum_arr,
    Optional<AnyMemRef>:$bias,
    Index:$nE,
    Index:$mE,
    I64Attr:$vecLen,
    I32Attr:$dataType
  );
  
  let assemblyFormat = "$s_buf `,` $p_T `,` $m_val `,` $sum_arr (`,` $bias^)? `,` $nE `,` $mE attr-dict `:` type($s_buf) `,` type($p_T) `,` type($m_val) `,` type($sum_arr) (`,` type($bias)^)? `,` type($nE) `,` type($mE)";
}

def MatrixColumnMaxOp : Spatz_Op<"matrix_column_max"> {
  let summary = "Find the maximum of each column in a 2D matrix using Spatz";
  let arguments = (ins AnyMemRef:$matrix, AnyMemRef:$maxVector,
                       Index:$rows, Index:$cols, I64Attr:$vecLen, 
                       I32Attr:$dataType);
  let assemblyFormat = "$matrix `,` $maxVector `,` $rows `,` $cols attr-dict `:` type($matrix) `,` type($maxVector) `,` type($rows) `,` type($cols)";
}

def VectorMaxUpdateOp : Spatz_Op<"vector_max_update"> {
  let summary = "Update a vector with element-wise maximum and output the difference from the old values";
  let arguments = (ins
    AnyMemRef:$vec_prev,   
    AnyMemRef:$vec_new,    
    AnyMemRef:$vec_diff,   
    Index:$len,            
    I64Attr:$vecLen,
    I32Attr:$dataType
  );
  let assemblyFormat = "$vec_prev `,` $vec_new `,` $vec_diff `,` $len attr-dict `:` type($vec_prev) `,` type($vec_new) `,` type($vec_diff) `,` type($len)";
}

def VectorExpOp : Spatz_Op<"vector_exp"> {
  let summary = "Apply fast Schraudolph exp to a 1D vector";
  let arguments = (ins
    AnyMemRef:$vec_in,    
    AnyMemRef:$vec_out,   
    Index:$len,           
    I64Attr:$vecLen,
    I32Attr:$dataType
  );
  let assemblyFormat = "$vec_in `,` $vec_out `,` $len attr-dict `:` type($vec_in) `,` type($vec_out) `,` type($len)";
}

def VectorAddOp : Spatz_Op<"vector_add"> {
  let summary = "Element-wise addition of two 1D vectors (in-place on vec_a)";
  let arguments = (ins
    AnyMemRef:$vec_a,      
    AnyMemRef:$vec_b,      
    Index:$len,
    I64Attr:$vecLen,
    I32Attr:$dataType
  );
  let assemblyFormat = "$vec_a `,` $vec_b `,` $len attr-dict `:` type($vec_a) `,` type($vec_b) `,` type($len)";
}

def VectorMulOp : Spatz_Op<"vector_mul"> {
  let summary = "Element-wise multiplication of two 1D vectors (in-place on vec_a)";
  let arguments = (ins
    AnyMemRef:$vec_a,      
    AnyMemRef:$vec_b,      
    Index:$len,
    I64Attr:$vecLen,
    I32Attr:$dataType
  );
  let assemblyFormat = "$vec_a `,` $vec_b `,` $len attr-dict `:` type($vec_a) `,` type($vec_b) `,` type($len)";
}

def MatrixRowScaleOp : Spatz_Op<"matrix_row_scale"> {
  let summary = "Multiply each row of a matrix by the corresponding element in a 1D scale vector";
  let arguments = (ins
    AnyMemRef:$matrix,     
    AnyMemRef:$scaleVec,   
    Index:$rows,           
    Index:$cols,           
    I64Attr:$vecLen,
    I32Attr:$dataType
  );
  let assemblyFormat = "$matrix `,` $scaleVec `,` $rows `,` $cols attr-dict `:` type($matrix) `,` type($scaleVec) `,` type($rows) `,` type($cols)";
}

def MatrixRowDivOp : Spatz_Op<"matrix_row_div"> {
  let summary = "Divide each row of a matrix by the corresponding element in a 1D divisor vector";
  let arguments = (ins
    AnyMemRef:$matrix,     
    AnyMemRef:$divVec,     
    Index:$rows,           
    Index:$cols,           
    I64Attr:$vecLen,
    I32Attr:$dataType
  );
  let assemblyFormat = "$matrix `,` $divVec `,` $rows `,` $cols attr-dict `:` type($matrix) `,` type($divVec) `,` type($rows) `,` type($cols)";
}

#endif // SPATZ
