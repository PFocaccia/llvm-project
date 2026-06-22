#include "mlir/Conversion/ArithToSnitch/ArithToSnitch.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct DivfToCallRewrite : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op, PatternRewriter &rewriter) const override {
    
    auto type = op.getType().dyn_cast<FloatType>();
    if (!type) return failure();

    StringRef funcName;
    switch (type.getWidth()) {
      case 64: funcName = "baremetal_fdiv_f64"; break;
      case 32: funcName = "baremetal_fdiv_f32"; break;
      case 16: funcName = "baremetal_fdiv_f16"; break;
      case 8:  funcName = "baremetal_fdiv_f8";  break;
      default: return failure();
    }

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    if (!module.lookupSymbol<func::FuncOp>(funcName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto funcType = rewriter.getFunctionType({type, type}, {type});
      auto funcDecl = rewriter.create<func::FuncOp>(op.getLoc(), funcName, funcType);
      funcDecl.setVisibility(SymbolTable::Visibility::Private);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, funcName, TypeRange{type}, ValueRange{op.getLhs(), op.getRhs()});

    return success();
  }
};

struct ConvertArithToSnitchPass : public ConvertArithToSnitchBase<ConvertArithToSnitchPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    
    patterns.add<DivfToCallRewrite>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertArithToSnitchPass() {
  return std::make_unique<ConvertArithToSnitchPass>();
}
} // namespace mlir