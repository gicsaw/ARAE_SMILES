#from utils.utils import *
import numpy as np
import os,sys
import time

char_list= ["H","C","N","O","F","P","S","Cl","Br","I",
"n","c","o","s",
"1","2","3","4","5","6","7","8",
"(",")","[","]",
"-","=","#","/","\\","+","@","X","Y"]
#char_dict=dict()
#i=-1
#for c in char_list:
#    i+=1
#    char_dict[c]=i
#print(char_dict)
#sys.exit()

char_dict={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 
'S': 6, 'Cl': 7, 'Br': 8, 'I': 9, 
'n': 10, 'c': 11, 'o': 12, 's': 13, 
'1': 14, '2': 15, '3': 16, '4': 17, '5': 18, '6': 19, '7': 20, '8': 21, 
'(': 22, ')': 23, '[': 24, ']': 25, '-': 26, '=': 27, '#': 28, 
'/': 29, '\\': 30, '+': 31, '@': 32, 'X': 33, 'Y': 34}



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
seq_length = 109
dev = 0.2

data_dir ='./dataset/ZINC'
data_name="train"
if len(sys.argv)>1:
    data_name=sys.argv[1]

print(data_name)

smiles_filename=data_dir+"/"+data_name+".txt"

fp = open(smiles_filename)
data_lines=fp.readlines()
fp.close()
smiles_list=[]


Maxsmiles=0
Xdata=[]
Ydata=[]
Ldata=[]
Pdata=[]
title=""
for line in data_lines:
    if line[0]=="#":
        title=line[1:-1]
        title_list=title.split()
        print(title_list)
        continue
    arr=line.split()
    if len(arr)<2:
        continue
    smiles=arr[0]
    if len(smiles)>seq_length:
        continue
    smiles0=smiles.ljust(seq_length,'Y')
    smiles_list+=[smiles]
    Narr=len(arr)
    cdd=[]
    for i in range(1,Narr):
        if title_list[i]=="logP":
            cdd+=[float(arr[i])/10.0]
        elif title_list[i]=="SAS":
            cdd+=[float(arr[i])/10.0]
#        elif title_list[i]=="QED":
#            cdd+=[float(arr[i])/1.0]
#        elif title_list[i]=="MW":
#            cdd+=[float(arr[i])/500.0]
        elif title_list[i]=="TPSA":
            cdd+=[float(arr[i])/150.0]

    Pdata+=[cdd] #affinity classification

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
Pdata = np.asarray(Pdata,dtype="float32")
print(Xdata.shape,Ydata.shape,Ldata.shape,Pdata.shape)

data_dir2="./data/ZINC/"
if not os.path.exists(data_dir2):
    os.makedirs(data_dir2)

Xfile=data_dir2+"X"+data_name+".npy"
Yfile=data_dir2+"Y"+data_name+".npy"
Lfile=data_dir2+"L"+data_name+".npy"
Pfile=data_dir2+"P"+data_name+".npy"
np.save(Xfile,Xdata)
np.save(Yfile,Ydata)
np.save(Lfile,Ldata)
np.save(Pfile,Pdata)



