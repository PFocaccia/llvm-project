# Matrix extension for RISC-V integrated into LLVM of PULP platform

## Compiling LLVM

```
mkdir build

cd build

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_TARGETS_TO_BUILD="RISCV" -DLLVM_OPTIMIZED_TABLEGEN=ON ../llvm

ninja
```

## Environment Setup (In-tree Usage)

To use the locally compiled tools (like `clang`, `llc` or `llvm-objdump`) without installing them system-wide (skipping `ninja install`), you must add the build binary directory to your `PATH` environment variable.

Replace `<YOUR_BUILD_DIR>` with the absolute path to your build directory (e.g., `/scratch/user/llvm-project/build`).

### For Bash / Zsh users

Run this in your terminal: `export PATH="<YOUR_BUILD_DIR>/bin:$PATH"`

### For Tcsh / Csh users

Run this in your terminal: `setenv PATH "<YOUR_BUILD_DIR>/bin:$PATH"`

### Verification 

To verify the environment setup, run: `clang --version`, `llc --version`.

## Compiling with LLVM 

To generate the assembly file: `clang --target=riscv32 -Xclang -target-feature -Xclang +experimental-xtheadmatrix -S helloworld.c -o helloworld.s`

To generate the object file: `clang --target=riscv32 -Xclang -target-feature -Xclang +experimental-xtheadmatrix -c helloworld.c -o helloworld.o`

## Explore obj file

`llvm-objdump -d --mattr=+experimental-xtheadmatrix helloworld.o`

## Quadrilatero-Spatz MLIR pipeline

```
mlir-opt mat_mul.mlir -canonicalize -lower-linalg-matmul-to-quadrilatero -lower-spatz-matrix-add -convert-linalg-to-loops -convert-scf-to-cf -convert-spatz-to-llvm -convert-func-to-llvm -convert-snitch-memory-to-llvm -convert-quadrilatero-to-llvm -convert-math-to-llvm -convert-arith-to-llvm -convert-memref-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts > MatMul.mlir

mlir-translate --mlir-to-llvmir MatMul.mlir -o MatMul.ll

llc -march=riscv32 -mattr=+m,+f,+v,+xdma,+experimental-xtheadmatrix -riscv-v-vector-bits-min=2048 MatMul.ll -o MatMul.s
```

