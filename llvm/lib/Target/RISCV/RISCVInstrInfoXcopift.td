//===-- RISCVInstrInfoXcopift.td - RISC-V 'Xcopift' instructions ---*- tablegen -*-===//
//
// Copyright 2024 ETH Zurich, University of Bologna.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the custom 'Xcopift'
// instruction set extension.
//
//===----------------------------------------------------------------------===//

def OPC_XCOPIFT : RISCVOpcode<"XCOPIFT", 0b0101011>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasExtXcopift] in {
    let isCompare = 1, hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
        def FCVT_D_WU_COPIFT : RVInst<(outs FPR64:$rd), (ins FPR64:$rs1), "fcvt.d.wu.copift", "$rd, $rs1", [], InstFormatOther> {
            bits<5> rs1;
            bits<5> rd;

            let Inst{31-27} = 0b11010;
            let Inst{26-25} = 0b01;
            let Inst{24-20} = 0b00001;
            let Inst{19-15} = rs1;
            let Inst{14-12} = 0b000;
            let Inst{11-7} = rd;
            let Inst{6-0} = OPC_XCOPIFT.Value;
        }
        def FCVT_D_W_COPIFT : RVInst<(outs FPR64:$rd), (ins FPR64:$rs1), "fcvt.d.w.copift", "$rd, $rs1", [], InstFormatOther> {
            bits<5> rs1;
            bits<5> rd;

            let Inst{31-27} = 0b11010;
            let Inst{26-25} = 0b01;
            let Inst{24-20} = 0b00000;
            let Inst{19-15} = rs1;
            let Inst{14-12} = 0b000;
            let Inst{11-7} = rd;
            let Inst{6-0} = OPC_XCOPIFT.Value;
        }
        def FLT_D_COPIFT : RVInst<(outs FPR64:$rd), (ins FPR64:$rs1, FPR64:$rs2), "flt.d.copift", "$rd, $rs1, $rs2", [], InstFormatOther> {
            bits<5> rs2;
            bits<5> rs1;
            bits<5> rd;

            let Inst{31-27} = 0b10100;
            let Inst{26-25} = 0b01;
            let Inst{24-20} = rs2;
            let Inst{19-15} = rs1;
            let Inst{14-12} = 0b001;
            let Inst{11-7}  = rd;
            let Inst{6-0} = OPC_XCOPIFT.Value;
        }
    }
}
