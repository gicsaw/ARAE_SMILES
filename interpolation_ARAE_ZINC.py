from model.ARAE import ARAE
#from utils.utils import *
import numpy as np
import os, sys
import time
import tensorflow as tf
import collections
from six.moves import cPickle
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

def convert_to_smiles(vector, char):
    smiles=""
    for i in vector:
        smiles+=char[i]
    return smiles

def cal_accuracy(S1, S2, length):
    count = 0
    for i in range(len(S1)):
        if np.array_equal(S1[i][1:length[i]+1],S2[i][:length[i]]):
            count+=1
    return count

char_list= ["H","C","N","O","F","P","S","Cl","Br","I",
"n","c","o","s",
"1","2","3","4","5","6","7","8",
"(",")","[","]",
"-","=","#","/","\\","+","@","X","Y"]

char_dict={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 
'S': 6, 'Cl': 7, 'Br': 8, 'I': 9, 
'n': 10, 'c': 11, 'o': 12, 's': 13, 
'1': 14, '2': 15, '3': 16, '4': 17, '5': 18, '6': 19, '7': 20, '8': 21, 
'(': 22, ')': 23, '[': 24, ']': 25, '-': 26, '=': 27, '#': 28, 
'/': 29, '\\': 30, '+': 31, '@': 32, 'X': 33, 'Y': 34}

vocab_size = len(char_list)
latent_size = 300
batch_size = 100
sample_size = 100
seq_length = 110
dev = 0.0


model_name="ARAE_ZINC"
save_dir="./save/"+model_name

model = ARAE(vocab_size = vocab_size,
             batch_size = batch_size,
             latent_size = latent_size,
             sample_size = sample_size,
             )

out_dir0="interpolation_"+model_name
if not os.path.exists(out_dir0):
    os.makedirs(out_dir0)

total_st=time.time()

epoch = 39
out_dir=out_dir0+"/%d" %epoch
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
output_file=out_dir+"/result_"+model_name+"_%d.txt" %epoch
fp0=open(output_file,"w")
model.restore(save_dir+"/model.ckpt-%d" %epoch)

reuse=False
model.recover_sample_vector_df(reuse)

latent_vector_fake=[]
Y_fake=[]
smiles_fake=[]

xtest = np.load('data/ZINC/Xtest.npy')
ytest = np.load('data/ZINC/Ytest.npy')
ltest = np.load('data/ZINC/Ltest.npy')
x = np.full([batch_size,seq_length], 34, dtype=np.long)
y = np.full([batch_size,seq_length], 34, dtype=np.long)
l = np.full([batch_size],0,dtype=np.long)
x[0:2] = xtest[0:2]
y[0:2] = ytest[0:2]
l[0:2] = ltest[0:2]

s = np.random.normal(0.0, 0.25, [batch_size, sample_size]).clip(-1.0,1.0)
n = np.random.normal(0.0, 0.00, [batch_size, latent_size])

Y, cost1, cost2, cost3, cost4, latent_vector, mol_encoded0 = model.test(x, y, l, s, n)
s0=model.recover_sample_vector(mol_encoded0)

Ninterpolation = 1000
s_interpolation = np.zeros([Ninterpolation,sample_size],dtype=np.float32)
for i in range(Ninterpolation):
    s_interpolation[i] = i/1000.0 * s0[0] + (1000.0-i)/1000.0 * s0[1]
num_batches = Ninterpolation//batch_size

for itest in range(num_batches):

    decoder_state = model.get_decoder_state()
    s = s_interpolation[itest*batch_size:(itest+1)*(batch_size)]
    latent_vector = model.generate_latent_vector(s)
    latent_vector_fake.append(latent_vector)

    start_token = np.array([char_list.index('X') for i in range(batch_size)])
    start_token = np.reshape(start_token, [batch_size, 1])
    length = np.array([1 for i in range(batch_size)])
    smiles = ['' for i in range(batch_size)]
    Y=[]
    for i in range(seq_length):
        m, state = model.generate_molecule(start_token, latent_vector, length, decoder_state)
        decoder_state = state
        start_token = np.argmax(m,2)
        Y.append(start_token[:,0])
        smiles = [s + str(char_list[start_token[j][0]]) for j,s in enumerate(smiles)]
    Y=list(map(list,zip(*Y)))
    Y_fake.append(Y)
    smiles_fake+=smiles


latent_vector_fake=np.array(latent_vector_fake,dtype="float32").reshape(-1,latent_size)
Y_fake=np.array(Y_fake,dtype="int32").reshape(-1,seq_length)
outfile=out_dir+"/Zfake.npy"
np.save(outfile,latent_vector_fake)
outfile=out_dir+"/Yfake.npy"
np.save(outfile,Y_fake)

outfile=out_dir+"/smiles_fake.txt"
fp_out=open(outfile,'w')
for line in smiles_fake:
    line_out=line+"\n"
    fp_out.write(line_out)
fp_out.close()

total_et=time.time()
print ("total_time : ", total_et-total_st)

