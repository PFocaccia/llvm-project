// RUN: mlir-opt -convert-snitch-memory-to-llvm="l1-memory-space=1" -split-input-file %s | FileCheck %s --check-prefix=SNITCH
// RUN: mlir-opt -convert-snitch-memory-to-llvm="l1-memory-space=1" -convert-memref-to-llvm -split-input-file %s | FileCheck %s --check-prefix=PIPE

// SNITCH-LABEL: func @xdma_copy_l1_to_global
// PIPE-LABEL: func @xdma_copy_l1_to_global
func @xdma_copy_l1_to_global(%src: memref<4x8xi32, 1>, %dst: memref<4x8xi32>) {
  memref.copy %src, %dst : memref<4x8xi32, 1> to memref<4x8xi32>
  // SNITCH: llvm.call @llvm.riscv.sdma.start.twod
  // SNITCH: llvm.call @llvm.riscv.sdma.wait.for.idle() : () -> ()
  // PIPE: llvm.call @llvm.riscv.sdma.start.twod
  // PIPE: llvm.call @llvm.riscv.sdma.wait.for.idle() : () -> ()
  return
}

// -----

// SNITCH-LABEL: func @l1_alloc
// PIPE-LABEL: func @l1_alloc
func @l1_alloc() -> memref<64xi32, 1> {
  %0 = memref.alloc() : memref<64xi32, 1>
  // SNITCH: llvm.call @snrt_l1alloc
  // PIPE: llvm.call @snrt_l1alloc
  return %0 : memref<64xi32, 1>
}

// -----

// SNITCH-LABEL: func @l1_dealloc
// PIPE-LABEL: func @l1_dealloc
func @l1_dealloc(%arg0: memref<64xi32, 1>) {
  memref.dealloc %arg0 : memref<64xi32, 1>
  // SNITCH-NOT: llvm.call @free
  // PIPE-NOT: llvm.call @free
  return
}

// -----

// SNITCH-LABEL: func @copy_untouched
func @copy_untouched(%src: memref<4x8xi32>, %dst: memref<4x8xi32>) {
  memref.copy %src, %dst : memref<4x8xi32> to memref<4x8xi32>
  // SNITCH: memref.copy
  return
}
