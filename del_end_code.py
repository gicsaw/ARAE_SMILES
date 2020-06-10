#!/usr/bin/env python
import sys

def main():

    if len(sys.argv)<1:
        print('del_end_code.py data_dir')

    data_dir=sys.argv[1]
    file_name=data_dir+'/smiles_fake.txt'
    end_code='Y'
    fp=open(file_name)
    lines=fp.readlines()
    fp.close()

    file_out=data_dir+'smiles_gen.txt'
    fp_out=open(file_out,'w')
    for line in lines:
        j=line.find(end_code)
        line_out=line[:j]+'\n'
        fp_out.write(line_out)
    fp_out.close()


if __name__=="__main__":
    main()
