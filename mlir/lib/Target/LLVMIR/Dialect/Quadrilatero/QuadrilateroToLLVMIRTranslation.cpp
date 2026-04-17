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

      SmallVector<llvm::Type *, 10> asmTypes;
      asmTypes.reserve(asmOperands.size());
      for (llvm::Value *value : asmOperands)
        asmTypes.push_back(value->getType());

      auto *fnTy = llvm::FunctionType::get(builder.getVoidTy(), asmTypes, false);
      auto asmString =
          "mmac.dt $7, $8, $9\n\t"
          "add t0, x0, $3\n\t"
          "add s0, x0, $2\n\t"
          "add s1, x0, $0\n\t"
          "sll s10, $3, $6\n\t"
          "sll s11, $4, $6\n\t"
          "slli a6, $4, 2\n\t"
          "1:\n\t"
          "mcfgm t3, t0, 1\n\t"
          "add t1, x0, $4\n\t"
          "add s2, x0, s0\n\t"
          "add s3, x0, $1\n\t"
          "2:\n\t"
          "mcfgn t4, t1, 1\n\t"
          "mzero.a acc0\n\t"
          "add t2, x0, $5\n\t"
          "add s4, x0, s1\n\t"
          "add s5, x0, s3\n\t"
          "3:\n\t"
          "mcfgk t5, t2\n\t"
          "mld.lhs m0, (s4), s10\n\t"
          "mld.rhs m4, (s5), s11\n\t"
          "mmacc acc0, m4, m0\n\t"
          "sub t2, t2, t5\n\t"
          "mcfgk t6, t2\n\t"
          "mul s8, t5, s10\n\t"
          "add s6, s4, s8\n\t"
          "mld.lhs m2, (s6), s10\n\t"
          "mul s9, t5, s11\n\t"
          "add s7, s5, s9\n\t"
          "mld.rhs m6, (s7), s11\n\t"
          "mmacc acc0, m6, m2\n\t"
          "add t5, t5, t6\n\t"
          "mul s8, t5, s10\n\t"
          "add s4, s4, s8\n\t"
          "mul s9, t5, s11\n\t"
          "add s5, s5, s9\n\t"
          "sub t2, t2, t6\n\t"
          "bgtz t2, 3b\n\t"
          "mmov.am m8, acc0\n\t"
          "slli t6, t4, 2\n\t"
          "add s3, s3, t6\n\t"
          "sub t1, t1, t4\n\t"
          "mst m8, (s2), a6\n\t"
          "add s2, s2, t6\n\t"
          "bgtz t1, 2b\n\t"
          "mul t6, t3, a6\n\t"
          "add s0, s0, t6\n\t"
          "slli t6, t3, 2\n\t"
          "add s1, s1, t6\n\t"
          "sub t0, t0, t3\n\t"
          "bgtz t0, 1b";

      auto constraints =
          "r,r,r,r,r,r,r,i,i,i,"
          "~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},"
          "~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},"
          "~{s8},~{s9},~{s10},~{s11},~{a6},~{memory}";

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
