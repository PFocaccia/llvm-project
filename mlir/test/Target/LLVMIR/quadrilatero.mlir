// RUN: mlir-opt %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s --check-prefix=LLVMIR

// ROUNDTRIP-LABEL: llvm.func @quadrilatero_kernel
// ROUNDTRIP: %[[M0:.*]] = quadrilatero.mld_lhs
// ROUNDTRIP: %[[M1:.*]] = quadrilatero.mld_rhs
// ROUNDTRIP: %[[ACC:.*]] = quadrilatero.mmacc %[[M0]], %[[M1]]
// ROUNDTRIP: quadrilatero.mst
// ROUNDTRIP: %[[CFG:.*]] = quadrilatero.mcfgm
// ROUNDTRIP: quadrilatero.mmac_dt 3, 1, 0
// ROUNDTRIP: llvm.return %[[CFG]] : i32

// LLVMIR-LABEL: define i32 @quadrilatero_kernel
// LLVMIR: call i32 @llvm.riscv.quadrilatero.mld.lhs
// LLVMIR: call i32 @llvm.riscv.quadrilatero.mld.rhs
// LLVMIR: call i32 @llvm.riscv.quadrilatero.mmacc
// LLVMIR: call i32 @llvm.riscv.quadrilatero.mmov.am
// LLVMIR: call void @llvm.riscv.quadrilatero.mst
// LLVMIR: call i32 @llvm.riscv.quadrilatero.mcfgm.i32(i32 %{{.*}}, i32 1)
// LLVMIR: call void @llvm.riscv.quadrilatero.mmac.dt(i32 3, i32 1, i32 0)

llvm.func @quadrilatero_kernel(%base: !llvm.ptr<i8>, %stride: i32, %x: i32) -> i32 {
  %m0 = quadrilatero.mld_lhs %base, %stride : !llvm.ptr<i8>, i32 -> !quadrilatero.matrix
  %m1 = quadrilatero.mld_rhs %base, %stride : !llvm.ptr<i8>, i32 -> !quadrilatero.matrix
  %acc = quadrilatero.mmacc %m0, %m1 : !quadrilatero.matrix, !quadrilatero.matrix -> !quadrilatero.acc
  %m2 = quadrilatero.mmov_am %acc : !quadrilatero.acc -> !quadrilatero.matrix
  quadrilatero.mst %m2, %base, %stride : !quadrilatero.matrix, !llvm.ptr<i8>, i32

  %cfg = quadrilatero.mcfgm %x, 1 : i32 -> i32
  quadrilatero.mmac_dt 3, 1, 0

  llvm.return %cfg : i32
}
