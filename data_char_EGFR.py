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

seq_length = 123

data_dir ='./dataset/EGFR_DUDE'
data_name = "data"

active_filename = data_dir+"/actives_final.ism"
data_active = [[x.strip().split()[0], 1] for x in open(active_filename)]
decoy_filename = data_dir+"/decoys_final.ism"
data_decoy = [[x.strip().split()[0], 0] for x in open(decoy_filename)]
data_list = data_active + data_decoy
num_data = len(data_list)
print(num_data)

Xdata=[]
Ydata=[]
Ldata=[]
Pdata=[]
title=""
for arr in data_list:
    smiles=arr[0]
    if len(smiles)>seq_length:
        continue
    smiles0=smiles.ljust(seq_length,'Y')
    Narr=len(arr)

    X_smiles='X'+smiles
    Y_smiles=smiles+'Y'
    X_d=np.zeros([seq_length+1],dtype=int)
    Y_d=np.zeros([seq_length+1],dtype=int)
    X_d[0]=char_dict['X']
    Y_d[-1]=char_dict['Y']

    Nsmiles=len(smiles)
    i=0
    istring=0
    check=True
    error = False
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
            print(char1, char2, "error")
            error = True
            break
        X_d[istring+1]=j
        Y_d[istring]=j
        istring+=1
    if error:
        continue
    for i in range(istring,seq_length):
        X_d[i+1]=char_dict['Y']
        Y_d[i]=char_dict['Y']

    Xdata+=[X_d]
    Ydata+=[Y_d]
    Ldata+=[istring+1]

    cdd=[arr[1]]
    Pdata+=[cdd] #affinity classification

Xdata = np.asarray(Xdata,dtype="int32")
Ydata = np.asarray(Ydata,dtype="int32")
Ldata = np.asarray(Ldata,dtype="int32")
Pdata = np.asarray(Pdata,dtype="float32")
print(Xdata.shape,Ydata.shape,Ldata.shape,Pdata.shape)

data_dir2="./data/EGFR/"
if not os.path.exists(data_dir2):
    os.makedirs(data_dir2)
num_data = Xdata.shape[0]
index = np.arange(num_data)
np.random.shuffle(index)
num_fold = 10
num_test = int(num_data/10)
index_test = index[:num_test]
index_train = index[num_test:]
index_test.sort()
index_train.sort()
num_train = index_train.shape[0]
print(num_train, num_test)
Xtest = Xdata[index_test]
Ytest = Ydata[index_test]
Ltest = Ldata[index_test]
Ptest = Pdata[index_test]

Xfile=data_dir2+"Xtest.npy"
Yfile=data_dir2+"Ytest.npy"
Lfile=data_dir2+"Ltest.npy"
Pfile=data_dir2+"Ptest.npy"
np.save(Xfile,Xtest)
np.save(Yfile,Ytest)
np.save(Lfile,Ltest)
np.save(Pfile,Ptest)

Xtrain = Xdata[index_train]
Ytrain = Ydata[index_train]
Ltrain = Ldata[index_train]
Ptrain = Pdata[index_train]

Xfile=data_dir2+"Xtrain.npy"
Yfile=data_dir2+"Ytrain.npy"
Lfile=data_dir2+"Ltrain.npy"
Pfile=data_dir2+"Ptrain.npy"
np.save(Xfile,Xtrain)
np.save(Yfile,Ytrain)
np.save(Lfile,Ltrain)
np.save(Pfile,Ptrain)

