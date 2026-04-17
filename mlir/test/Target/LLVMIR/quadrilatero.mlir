// RUN: mlir-opt %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s --check-prefix=LLVMIR

// ROUNDTRIP-LABEL: llvm.func @quadrilatero_kernel
// ROUNDTRIP: quadrilatero.tcdm_matmul
// ROUNDTRIP: llvm.return

// LLVMIR-LABEL: define void @quadrilatero_kernel
// LLVMIR: call void asm sideeffect "mmac.dt $7, $8, $9

llvm.func @quadrilatero_kernel(%a: !llvm.ptr<i8>, %b: !llvm.ptr<i8>,
                               %c: !llvm.ptr<i8>, %m: i32, %n: i32, %k: i32,
                               %shift: i32) {
  quadrilatero.tcdm_matmul %a, %b, %c, %m, %n, %k, %shift {dtC = 3 : i32, dtA = 1 : i32, dtB = 0 : i32}
      : !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i32, i32, i32, i32

  llvm.return
}
