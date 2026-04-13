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
      auto immC = dt->getAttrOfType<IntegerAttr>("immC");
      auto immA = dt->getAttrOfType<IntegerAttr>("immA");
      auto immB = dt->getAttrOfType<IntegerAttr>("immB");
      if (!immC || !immA || !immB)
        return failure();
      llvm::Value *c = builder.getInt32(immC.getInt());
      llvm::Value *a = builder.getInt32(immA.getInt());
      llvm::Value *b = builder.getInt32(immB.getInt());
      llvm::Function *fn =
          getDecl(builder, llvm::Intrinsic::riscv_quadrilatero_mmac_dt);
      builder.CreateCall(fn, {c, a, b});
      return success();
    }

    return failure();
  }
};
} // namespace

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
