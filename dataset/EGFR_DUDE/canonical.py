#!/usr/bin/env python
import sys, os
import numpy as np
from rdkit import Chem


def canonicalize(input_file, output_file):

    data = [x.strip().split() for x in open(input_file)]
    fp_out = open(output_file, 'w')
    for arr in data:
        smi = arr[0]
        Nsmi=len(smi)
        mol=Chem.MolFromSmiles(smi)
        if mol == None :
            continue
        if Chem.SanitizeMol(mol,catchErrors=True):
            continue
        smi2=Chem.MolToSmiles(mol)
        line_out = '%s' %(smi2)
        for a in arr[1:]:
            line_out += ' %s' %(a)
        line_out += '\n'
        fp_out.write(line_out)
    fp_out.close()


def main():

    data_dir = '.'
    active_filename = data_dir+"/actives_final.ism"
    decoy_filename = data_dir+"/decoys_final.ism"
    active_output = data_dir+"/actives_canonical.txt"
    decoy_output = data_dir+"/decoys_canonical.txt"

    canonicalize(active_filename, active_output)
    canonicalize(decoy_filename, decoy_output)

if __name__=="__main__":
    main()

