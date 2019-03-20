#from utils.utils import *
import numpy as np
import os,sys
import time

char_list= ["H","C","N","O","F",
"n","c","o",
"1","2","3","4","5",
"(",")","[","]",
"-","=","#","+","X","Y"]
#char_dict=dict()
#i=-1
#for c in char_list:
#    i+=1
#    char_dict[c]=i
#print(char_dict)
#sys.exit()

char_dict={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 
'n': 5, 'c': 6, 'o': 7, 
'1': 8, '2': 9, '3': 10, '4': 11, '5': 12, 
'(': 13, ')': 14, '[': 15, ']': 16, 
'-': 17, '=': 18, '#': 19, '+': 20, 'X': 21, 'Y': 22}

char_list1=list()
char_list2=list()
char_dict1=dict()
char_dict2=dict()
for key in char_list:
    if len(key)==1:
        char_list1+=[key]
        char_dict1[key]=char_dict[key]
    elif len(key)==2:
        char_list2+=[key]
        char_dict2[key]=char_dict[key]
    else:
        print("strange ",key)

#print(char_list1)
#print(char_list2)
#print(char_dict1)
#print(char_dict2)


Nchar=len(char_list)

batch_size = 100

sample_size = 100
seq_length = 34

data_dir ='./dataset/QM9'
data_name="train"
if len(sys.argv)>1:
    data_name=sys.argv[1]

print(data_name)

smiles_filename=data_dir+"/"+data_name+".smi"

fp = open(smiles_filename)
data_lines=fp.readlines()
fp.close()
smiles_list=[]


Maxsmiles=0
Xdata=[]
Ydata=[]
Ldata=[]
#Pdata=[]
title=""
for line in data_lines:
    if line[0]=="#":
        title=line[1:-1]
        title_list=title.split()
        continue
    arr=line.split()
    if len(arr)<2:
        continue
    smiles=arr[1]
    if len(smiles)>seq_length:
        continue
    smiles0=smiles.ljust(seq_length,'Y')
    smiles_list+=[smiles]
    Narr=len(arr)

    X_smiles='X'+smiles
    Y_smiles=smiles+'Y'
    X_d=np.zeros([seq_length+1],dtype=int)
    Y_d=np.zeros([seq_length+1],dtype=int)
    X_d[0]=char_dict['X']
    Y_d[-1]=char_dict['Y']

    Nsmiles=len(smiles)
    if Maxsmiles<Nsmiles:
        Maxsmiles=Nsmiles
    i=0
    istring=0
    check=True
    while check:
        char2=smiles[i:i+2]
        char1=smiles[i]
        if char2 in char_list2 :
            j=char_dict2[char2]
            i+=2
            if i>=Nsmiles:
                check=False
        elif char1 in char_list1 :
            j=char_dict1[char1]
            i+=1
            if i>=Nsmiles:
                check=False
        else:
            print(char1,char2,"error")
            sys.exit()
        X_d[istring+1]=j
        Y_d[istring]=j
        istring+=1
    for i in range(istring,seq_length):
        X_d[i+1]=char_dict['Y']
        Y_d[i]=char_dict['Y']

    Xdata+=[X_d]
    Ydata+=[Y_d]
    Ldata+=[istring+1]

Xdata = np.asarray(Xdata,dtype="int32")
Ydata = np.asarray(Ydata,dtype="int32")
Ldata = np.asarray(Ldata,dtype="int32")
#Pdata = np.asarray(Pdata,dtype="float32")
print(Xdata.shape,Ydata.shape,Ldata.shape)

data_dir2="./data/QM9/"
if not os.path.exists(data_dir2):
    os.makedirs(data_dir2)

Xfile=data_dir2+"X"+data_name+".npy"
Yfile=data_dir2+"Y"+data_name+".npy"
Lfile=data_dir2+"L"+data_name+".npy"
#Pfile=data_dir2+"P"+data_name+".npy"
np.save(Xfile,Xdata)
np.save(Yfile,Ydata)
np.save(Lfile,Ldata)
#np.save(Pfile,Pdata)



