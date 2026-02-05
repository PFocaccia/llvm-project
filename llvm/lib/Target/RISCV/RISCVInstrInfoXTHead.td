//===-- RISCVInstrInfoXTHead.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by T-Head of Alibaba.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Experimental RV64 i32 legalization patterns.
//===----------------------------------------------------------------------===//

class RVMArithBase<bits<4> funct4_h, bits<5> funct5_l, dag outs, dag ins,
                  string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {

  let Inst{31-28} = funct4_h;
  let Inst{27-25} = 0b000;
  let Inst{14-12} = 0b000;
  let Inst{11-7} = funct5_l;
  let Inst{6-0} = OPC_CUSTOM_0.Value;
}

class RVMArith<bits<4> funct4_h, bits<5> funct5_l, dag outs, dag ins,
                  string opcodestr, string argstr>
    : RVMArithBase<funct4_h, funct5_l, outs, ins, opcodestr, argstr> {
  bits<3> ms2;
  bits<3> ms1;
  bits<2> md ;

  let Inst{24} = 0b0;
  let Inst{23-21} = ms2;
  let Inst{20-18} = ms1;
  let Inst{17} = 0b0;
  let Inst{16-15} = md ;
}

class RVMMove<bits<4> funct4_h, bits<5> funct5_l, dag outs, dag ins,
                  string opcodestr, string argstr>
    : RVMArithBase<funct4_h, funct5_l, outs, ins, opcodestr, argstr> {
  bits<3> ms1;
  bits<3> md ;

  let Inst{24-21} = 0b0000;
  let Inst{20-18} = ms1;
  let Inst{17-15} = md ;
}

class RVMZero<bits<4> funct4_h, bits<5> funct5_l, bits<1> funct, dag outs, dag ins,
                  string opcodestr, string argstr>
    : RVMArithBase<funct4_h, funct5_l, outs, ins, opcodestr, argstr> {
  bits<3> ms1;
  bits<3> md ;

  let Inst{24} = funct;
  let Inst{23-18} = 0b000000;
  let Inst{17-15} = md ;
}

class RVMLoadStore<bits<1> matrix, bits<3> ldst, bits<2> size, dag outs, dag ins,
                  string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<3> mds3;

  let Inst{31-29} = 0b000;
  let Inst{28}    = matrix;
  let Inst{27-25} = ldst;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{11-10} = size;
  let Inst{9-7} = mds3;

  let Inst{6-0} = OPC_CUSTOM_0.Value;
}

class RVMConfig<bits<3> funct3_h, dag outs, dag ins, string opcodestr, string argstr> 
  : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
    bits<5> rs1;
    bits<5> rd;

    let Inst{31} = 0b0;
    let Inst{30-28} = funct3_h;
    let Inst{27-23} = 0b11100;
    let Inst{19-15} = rs1;
    let Inst{14-12} = 0b000;
    let Inst{11-7}  = rd;
    let Inst{6-0} = OPC_CUSTOM_0.Value;
}

class RVMConfigK<bits<3> funct3_h, dag outs, dag ins, string opcodestr, string argstr> 
  : RVMConfig<funct3_h, outs, ins, opcodestr, argstr> {
    bits<3> imm3;
    let Inst{22-20} = imm3;
}

let Predicates = [HasVendorXTHeadMatrix], DecoderNamespace = "XTHeadMatrix",hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  let hasSideEffects = 1 in {
    def MCFGK : RVMConfigK<0b000, (outs GPR:$rd), (ins GPR:$rs1, uimm3:$imm3), "mcfgk", "$rd, $rs1, $imm3">;
    def MCFGM : RVMConfig<0b001, (outs GPR:$rd), (ins GPR:$rs1), "mcfgm", "$rd, $rs1">;
    def MCFGN : RVMConfig<0b010, (outs GPR:$rd), (ins GPR:$rs1), "mcfgn", "$rd, $rs1">;
  }
  def FMMACC_B : RVMArith<0b0001, 0b00000, (outs ACC:$md), (ins MR:$ms1, MR:$ms2), "fmmacc.b", "$md, $ms2, $ms1">;
  def FMMACC_H : RVMArith<0b0001, 0b01000, (outs ACC:$md), (ins MR:$ms1, MR:$ms2), "fmmacc.h", "$md, $ms2, $ms1">;
  def FMMACC_S : RVMArith<0b0001, 0b10000, (outs ACC:$md), (ins MR:$ms1, MR:$ms2), "fmmacc.s", "$md, $ms2, $ms1">;
  def MMAQA_B  : RVMArith<0b0010, 0b00000, (outs ACC:$md), (ins MR:$ms1, MR:$ms2), "mmaqa.b" , "$md, $ms2, $ms1">;
  def MMADA_H  : RVMArith<0b0010, 0b01000, (outs ACC:$md), (ins MR:$ms1, MR:$ms2), "mmada.h" , "$md, $ms2, $ms1">;
  def MMASA_W  : RVMArith<0b0010, 0b10000, (outs ACC:$md), (ins MR:$ms1, MR:$ms2), "mmasa.w" , "$md, $ms2, $ms1">;

  def MMOV_MM  : RVMMove<0b0000, 0b00001, (outs MR:$md), (ins MR:$ms1), "mmov.mm" , "$md, $ms1">;
  def MMOV_MA  : RVMMove<0b0000, 0b01001, (outs ACC:$md), (ins MR:$ms1), "mmov.ma" , "$md, $ms1">;
  def MMOV_AM  : RVMMove<0b0000, 0b10001, (outs MR:$md), (ins ACC:$ms1), "mmov.am" , "$md, $ms1">;
  def MMOV_AA  : RVMMove<0b0000, 0b11001, (outs ACC:$md), (ins ACC:$ms1), "mmov.aa" , "$md, $ms1">;

  def MZERO_M : RVMZero<0b1010, 0b00000, 0b0, (outs MR:$md), (ins), "mzero.m", "$md">;
  def MZERO_A : RVMZero<0b1010, 0b00000, 0b1, (outs ACC:$md), (ins), "mzero.a", "$md">;

  let mayLoad = 1 in {
    def MLD_LHS_B : RVMLoadStore<0b0, 0b100, 0b00, (outs MR:$mds3), (ins GPR:$rs1, GPR:$rs2), "mld.lhs.b", "$mds3, (${rs1}), $rs2">;
    def MLD_LHS_H : RVMLoadStore<0b0, 0b100, 0b01, (outs MR:$mds3), (ins GPR:$rs1, GPR:$rs2), "mld.lhs.h", "$mds3, (${rs1}), $rs2">;
    def MLD_LHS_W : RVMLoadStore<0b0, 0b100, 0b10, (outs MR:$mds3), (ins GPR:$rs1, GPR:$rs2), "mld.lhs.w", "$mds3, (${rs1}), $rs2">;
    def MLD_RHS_B : RVMLoadStore<0b1, 0b100, 0b00, (outs MR:$mds3), (ins GPR:$rs1, GPR:$rs2), "mld.rhs.b", "$mds3, (${rs1}), $rs2">;
    def MLD_RHS_H : RVMLoadStore<0b1, 0b100, 0b01, (outs MR:$mds3), (ins GPR:$rs1, GPR:$rs2), "mld.rhs.h", "$mds3, (${rs1}), $rs2">;
    def MLD_RHS_W : RVMLoadStore<0b1, 0b100, 0b10, (outs MR:$mds3), (ins GPR:$rs1, GPR:$rs2), "mld.rhs.w", "$mds3, (${rs1}), $rs2">;
  }

  let mayStore = 1 in {
    def MST_B : RVMLoadStore<0b0, 0b101, 0b00, (outs), (ins MR:$mds3, GPR:$rs1, GPR:$rs2), "mst.b", "$mds3, (${rs1}), $rs2">;
    def MST_H : RVMLoadStore<0b0, 0b101, 0b01, (outs), (ins MR:$mds3, GPR:$rs1, GPR:$rs2), "mst.h", "$mds3, (${rs1}), $rs2">;
    def MST_W : RVMLoadStore<0b0, 0b101, 0b10, (outs), (ins MR:$mds3, GPR:$rs1, GPR:$rs2), "mst.w", "$mds3, (${rs1}), $rs2">;
  }

}
