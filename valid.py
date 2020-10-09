#!/usr/bin/env python
import sys, os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem as AllChem

from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue



USAGE="""
valid.py data_dir
"""

def creator(q, data, Nproc):
    Ndata = len(data)
    for d in data:
        idx=d[0]
        smiles=d[1]
        q.put((idx, smiles))

    for i in range(0, Nproc):
        q.put('DONE')


def check_validity(q,return_dict_valid):

    while True:
        qqq = q.get()
        if qqq == 'DONE':
#            print('proc =', os.getpid())
            break
        idx, smi0 = qqq
        if idx%100==0:
            print(idx)

        lis=smi0.split(".")
        scount=0
        len_frag_max=0
        for i in range(len(lis)):
            smi_frag=lis[i]
            Nfrag=len(smi_frag)
            if Nfrag>len_frag_max:
                smi=smi_frag
                len_frag_max=Nfrag
#            if len(smi_frag)>5:
#                smi=smi_frag
#                scount+=1
#        if scount>1 or scount==0:
#            print(smi0)
#            continue

        Nsmi=len(smi)
        mol=Chem.MolFromSmiles(smi)
        if mol == None :
            continue
        if Chem.SanitizeMol(mol,catchErrors=True):
            continue

        smi2=Chem.MolToSmiles(mol)

#        if smi!=smi2:
#            print(i,smi,smi2)
        return_dict_valid[idx]=[smi2]


def main():
    if len(sys.argv)<1:
        print(USAGE)
        sys.exit()

    data_dir=sys.argv[1]

    Nproc=10
    gen_file=data_dir+"/smiles_gen.txt"
    fp=open(gen_file)
    lines=fp.readlines()
    fp.close()
    k=-1
    gen_data=[]
    for line in lines:
        if line.startswith("#"):
            continue

        smi=line.strip()
        if len(smi)>200:
            continue
        k+=1
        gen_data+=[[k,smi]]

    Ndata=len(gen_data)

    q = Queue()
    manager = Manager()
    return_dict_valid = manager.dict()
    proc_master = Process(target=creator, args=(q, gen_data, Nproc))
    proc_master.start()

    procs = []
    for k in range(0, Nproc):
        proc = Process(target=check_validity, args=(q,return_dict_valid))
        procs.append(proc)
        proc.start()

    q.close()
    q.join_thread()
    proc_master.join()
    for proc in procs:
        proc.join()

    keys = sorted(return_dict_valid.keys())
    num_valid=keys

    valid_smi_list=[]
    for idx in keys:
        valid_smi=return_dict_valid[idx][0]
        valid_smi_list+=[valid_smi]

    num_valid=len(valid_smi_list)

    line_out="valid:  %6d %6d %6.4f" %(num_valid,Ndata,float(num_valid)/Ndata)
    print(line_out)

    unique_set=set(valid_smi_list)
    num_set=len(unique_set)
    unique_list=sorted(unique_set)

    line_out="Unique:  %6d %6d %6.4f" %(num_set,num_valid,float(num_set)/float(num_valid))
    print(line_out)

    file_output2=data_dir+"/smiles_unique.txt"
    fp_out2=open(file_output2,"w")
    line_out="#smi\n"
    fp_out2.write(line_out)

    for smi in unique_list:
        line_out="%s\n" %(smi)
        fp_out2.write(line_out)
    fp_out2.close()


if __name__=="__main__":
    main()






