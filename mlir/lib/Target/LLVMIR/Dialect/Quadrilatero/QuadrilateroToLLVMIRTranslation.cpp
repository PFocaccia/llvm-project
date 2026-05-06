//===- QuadrilateroToLLVMIRTranslation.cpp - Translate Quadrilatero to LLVM IR -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the Quadrilatero dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/Quadrilatero/QuadrilateroToLLVMIRTranslation.h"
#include "mlir/Dialect/Quadrilatero/QuadrilateroDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsRISCV.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the Quadrilatero dialect to LLVM IR.
class QuadrilateroDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  static llvm::Function *getDecl(llvm::IRBuilderBase &builder,
                                 llvm::Intrinsic::ID id,
                                 llvm::ArrayRef<llvm::Type *> overloads = {}) {
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    return llvm::Intrinsic::getDeclaration(module, id, overloads);
  }

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    if (auto mld = dyn_cast<quadrilatero::MldLhsOp>(op)) {
      auto operands = moduleTranslation.lookupValues(mld->getOperands());
      llvm::Type *strideTy = operands[1]->getType();
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mld_lhs,
                  {strideTy});
      llvm::Value *res = builder.CreateCall(fn, operands);
      moduleTranslation.mapValue(mld->getResult(0), res);
      return success();
    }

    if (auto mld = dyn_cast<quadrilatero::MldRhsOp>(op)) {
      auto operands = moduleTranslation.lookupValues(mld->getOperands());
      llvm::Type *strideTy = operands[1]->getType();
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mld_rhs,
                  {strideTy});
      llvm::Value *res = builder.CreateCall(fn, operands);
      moduleTranslation.mapValue(mld->getResult(0), res);
      return success();
    }

    if (auto mst = dyn_cast<quadrilatero::MstOp>(op)) {
      auto operands = moduleTranslation.lookupValues(mst->getOperands());
      llvm::Type *strideTy = operands[2]->getType();
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mst,
                  {strideTy});
      builder.CreateCall(fn, operands);
      return success();
    }

    if (auto mmacc = dyn_cast<quadrilatero::MmaccOp>(op)) {
      auto operands = moduleTranslation.lookupValues(mmacc->getOperands());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mmacc);
      llvm::Value *res = builder.CreateCall(fn, operands);
      moduleTranslation.mapValue(mmacc->getResult(0), res);
      return success();
    }

    if (auto mzero = dyn_cast<quadrilatero::MzeroMOp>(op)) {
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mzero_m);
      llvm::Value *res = builder.CreateCall(fn, {});
      moduleTranslation.mapValue(mzero->getResult(0), res);
      return success();
    }

    if (auto mzero = dyn_cast<quadrilatero::MzeroAOp>(op)) {
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mzero_a);
      llvm::Value *res = builder.CreateCall(fn, {});
      moduleTranslation.mapValue(mzero->getResult(0), res);
      return success();
    }

    if (auto mmov = dyn_cast<quadrilatero::MmovMmOp>(op)) {
      auto operands = moduleTranslation.lookupValues(mmov->getOperands());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mmov_mm);
      llvm::Value *res = builder.CreateCall(fn, operands);
      moduleTranslation.mapValue(mmov->getResult(0), res);
      return success();
    }

    if (auto mmov = dyn_cast<quadrilatero::MmovMaOp>(op)) {
      auto operands = moduleTranslation.lookupValues(mmov->getOperands());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mmov_ma);
      llvm::Value *res = builder.CreateCall(fn, operands);
      moduleTranslation.mapValue(mmov->getResult(0), res);
      return success();
    }

    if (auto mmov = dyn_cast<quadrilatero::MmovAmOp>(op)) {
      auto operands = moduleTranslation.lookupValues(mmov->getOperands());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mmov_am);
      llvm::Value *res = builder.CreateCall(fn, operands);
      moduleTranslation.mapValue(mmov->getResult(0), res);
      return success();
    }

    if (auto mmov = dyn_cast<quadrilatero::MmovAaOp>(op)) {
      auto operands = moduleTranslation.lookupValues(mmov->getOperands());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mmov_aa);
      llvm::Value *res = builder.CreateCall(fn, operands);
      moduleTranslation.mapValue(mmov->getResult(0), res);
      return success();
    }

    if (auto cfg = dyn_cast<quadrilatero::McfgkOp>(op)) {
      auto operands = moduleTranslation.lookupValues(cfg->getOperands());
      llvm::Type *xlenTy = operands[0]->getType();
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mcfgk,
                  {xlenTy});
      llvm::Value *res = builder.CreateCall(fn, operands);
      moduleTranslation.mapValue(cfg->getResult(0), res);
      return success();
    }

    if (auto cfg = dyn_cast<quadrilatero::McfgmOp>(op)) {
      auto operands = moduleTranslation.lookupValues(cfg->getOperands());
      llvm::Type *xlenTy = operands[0]->getType();
      auto imm2 = cfg->getAttrOfType<IntegerAttr>("imm2");
      if (!imm2)
        return failure();
      llvm::Value *imm = builder.getInt32(imm2.getInt());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mcfgm,
                  {xlenTy});
      llvm::Value *res = builder.CreateCall(fn, {operands[0], imm});
      moduleTranslation.mapValue(cfg->getResult(0), res);
      return success();
    }

    if (auto cfg = dyn_cast<quadrilatero::McfgnOp>(op)) {
      auto operands = moduleTranslation.lookupValues(cfg->getOperands());
      llvm::Type *xlenTy = operands[0]->getType();
      auto imm2 = cfg->getAttrOfType<IntegerAttr>("imm2");
      if (!imm2)
        return failure();
      llvm::Value *imm = builder.getInt32(imm2.getInt());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mcfgn,
                  {xlenTy});
      llvm::Value *res = builder.CreateCall(fn, {operands[0], imm});
      moduleTranslation.mapValue(cfg->getResult(0), res);
      return success();
    }

    if (auto dt = dyn_cast<quadrilatero::MmacDtOp>(op)) {
      auto dtC = dt->getAttrOfType<IntegerAttr>("dtC");
      auto dtA = dt->getAttrOfType<IntegerAttr>("dtA");
      auto dtB = dt->getAttrOfType<IntegerAttr>("dtB");
      if (!dtC || !dtA || !dtB)
        return failure();
      llvm::Value *c = builder.getInt32(dtC.getInt());
      llvm::Value *a = builder.getInt32(dtA.getInt());
      llvm::Value *b = builder.getInt32(dtB.getInt());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mmac_dt);
      builder.CreateCall(fn, {c, a, b});
      return success();
    }

    if (auto tcdm = dyn_cast<quadrilatero::TcdmMatmulOp>(op)) {
      auto operands = moduleTranslation.lookupValues(tcdm->getOperands());
      auto dtC = tcdm->getAttrOfType<IntegerAttr>("dtC");
      auto dtA = tcdm->getAttrOfType<IntegerAttr>("dtA");
      auto dtB = tcdm->getAttrOfType<IntegerAttr>("dtB");
      if (!dtC || !dtA || !dtB)
        return failure();

      llvm::Value *c = builder.getInt32(dtC.getInt());
      llvm::Value *a = builder.getInt32(dtA.getInt());
      llvm::Value *b = builder.getInt32(dtB.getInt());

      SmallVector<llvm::Value *, 10> asmOperands;
      asmOperands.append(operands.begin(), operands.end());
      asmOperands.push_back(c);
      asmOperands.push_back(a);
      asmOperands.push_back(b);

      auto toI8Ptr = [&](llvm::Value *value) -> llvm::Value * {
        auto *ptrTy = llvm::cast<llvm::PointerType>(value->getType());
        auto *i8Ty = llvm::Type::getInt8Ty(builder.getContext());
        auto *i8PtrTy = llvm::PointerType::get(i8Ty, ptrTy->getAddressSpace());
        return builder.CreateBitCast(value, i8PtrTy);
      };

      asmOperands[0] = toI8Ptr(asmOperands[0]);
      asmOperands[1] = toI8Ptr(asmOperands[1]);
      asmOperands[2] = toI8Ptr(asmOperands[2]);

      SmallVector<llvm::Type *, 10> asmTypes;
      asmTypes.reserve(asmOperands.size());
      for (llvm::Value *value : asmOperands)
        asmTypes.push_back(value->getType());

      auto *fnTy = llvm::FunctionType::get(builder.getVoidTy(), asmTypes, false);
      
      auto asmString =
          "mmac.dt $7, $8, $9\n\t"
          "add t0, x0, $3\n\t"       // t0 = M_rem
          "add t1, x0, $2\n\t"       // t1 = Base C
          "add t2, x0, $0\n\t"       // t2 = Base A
          "li t3, 256\n\t"           // t3 = Stride unico (256) per A, B, C
          
          "1:\n\t" // M-loop
          "mcfgm t4, t0, 1\n\t"      // t4 = M_chunk
          "add t5, x0, $4\n\t"       // t5 = N_rem (pescato da input $4)
          "add t6, x0, t1\n\t"       // t6 = Ptr C (N-loop)
          "add a0, x0, $1\n\t"       // a0 = Base B (pescato da input $1)
          
          "2:\n\t" // N-loop
          "mcfgn a1, t5, 1\n\t"      // a1 = N_chunk
          "mzero.a acc0\n\t"
          
          "add a2, x0, $5\n\t"       // a2 = K_rem (pescato da input $5)
          "add a3, x0, t2\n\t"       // a3 = Ptr A (K-loop)
          "add a4, x0, a0\n\t"       // a4 = Ptr B (K-loop)
          
          "3:\n\t" // K-loop
          "mcfgk a5, a2\n\t"         // a5 = K_chunk
          "mld.lhs m0, (a3), t3\n\t"
          "mld.rhs m4, (a4), t3\n\t"
          "mmacc acc0, m4, m0\n\t"
          "sub a2, a2, a5\n\t"
          "mcfgk a6, a2\n\t"         // a6 = next_k_chunk
          
          // --- SHIFT ($6) MANTENUTO SOLO NEL K-LOOP ---
          "srl s0, a5, $6\n\t"       // s0 = a5 / 2^shift
          "mul s0, s0, t3\n\t"       // s0 = offset temporaneo
          "add s1, a3, s0\n\t"       // s1 = ptr A temporaneo
          "mld.lhs m2, (s1), t3\n\t"
          "add s2, a4, s0\n\t"       // s2 = ptr B temporaneo
          "mld.rhs m6, (s2), t3\n\t"
          "mmacc acc0, m6, m2\n\t"
          "add a5, a5, a6\n\t"       // a5 = K_chunk + next_k_chunk
          
          "srl s0, a5, $6\n\t"
          "mul s0, s0, t3\n\t"
          "add a3, a3, s0\n\t"       // ptr A definitivo += offset
          "add a4, a4, s0\n\t"       // ptr B definitivo += offset
          // ---------------------------------------------
          
          "sub a2, a2, a6\n\t"
          "bgtz a2, 3b\n\t"
          
          "mmov.am m8, acc0\n\t"
          
          // --- N-loop: NESSUN SHIFT ($6 rimosso) ---
          "slli s1, a1, 2\n\t"       // s1 = N_chunk * 4 (calcolo offset in byte)
          "add a0, a0, s1\n\t"       // Base B += offset
          "sub t5, t5, a1\n\t"
          "mst m8, (t6), t3\n\t"
          "add t6, t6, s1\n\t"       // Ptr C += offset
          // -----------------------------------------
          
          "bgtz t5, 2b\n\t"
          
          // --- M-loop: NESSUN SHIFT ($6 rimosso) ---
          "mul s1, t4, t3\n\t"       // s1 = M_chunk * stride(256)
          "add t1, t1, s1\n\t"       // Base C += offset_stride
          "slli s1, t4, 2\n\t"       // s1 = M_chunk * 4 (calcolo offset in byte)
          "add t2, t2, s1\n\t"       // Base A += offset_bytes
          // -----------------------------------------
          
          "sub t0, t0, t4\n\t"
          "bgtz t0, 1b";

      // CLOBBER LIST: Il compilatore gestirà lo stack per s0, s1 e s2 in automatico!
      auto constraints =
          "r,r,r,r,r,r,r,i,i,i,"
          "~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},"
          "~{a0},~{a1},~{a2},~{a3},~{a4},~{a5},~{a6},"
          "~{s0},~{s1},~{s2},~{memory}";

      auto *inlineAsm = llvm::InlineAsm::get(fnTy, asmString, constraints, /*hasSideEffects=*/true, /*isAlignStack=*/false);
      builder.CreateCall(inlineAsm, asmOperands);
      return success();
    }

    return failure();
  }
};
}

void mlir::registerQuadrilateroDialectTranslation(DialectRegistry &registry) {
  registry.insert<quadrilatero::QuadrilateroDialect>();
  registry.addExtension(
      +[](MLIRContext *ctx, quadrilatero::QuadrilateroDialect *dialect) {
        dialect->addInterfaces<QuadrilateroDialectLLVMIRTranslationInterface>();
      });
}

void mlir::registerQuadrilateroDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerQuadrilateroDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
