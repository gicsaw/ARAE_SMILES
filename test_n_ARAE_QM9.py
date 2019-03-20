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

def read_data(data_dir,data):

    Xfile=data_dir+"X"+data+".npy"
    Yfile=data_dir+"Y"+data+".npy"
    Lfile=data_dir+"L"+data+".npy"

    Xdata=np.load(Xfile)
    Ydata=np.load(Yfile)
    Ldata=np.load(Lfile)

    return Xdata,Ydata,Ldata

char_list= ["H","C","N","O","F",
"n","c","o",
"1","2","3","4","5",
"(",")","[","]",
"-","=","#","+","X","Y"]

char_dict={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 
'n': 5, 'c': 6, 'o': 7, 
'1': 8, '2': 9, '3': 10, '4': 11, '5': 12, 
'(': 13, ')': 14, '[': 15, ']': 16, 
'-': 17, '=': 18, '#': 19, '+': 20, 'X': 21, 'Y': 22}

vocab_size = len(char_list)
latent_size = 200
batch_size = 100
sample_size = 100
seq_length = 34
dev = 0.0

coord = tf.train.Coordinator()

data_dir="./data/QM9/"
model_name="ARAE_QM9"
save_dir="./save/"+model_name

Xtest,Ytest,Ltest=read_data(data_dir,"test")
num_test_batches = int(len(Xtest)/batch_size)

Xtest = Xtest[0:num_test_batches*batch_size]
Ytest = Ytest[0:num_test_batches*batch_size]
Ltest = Ltest[0:num_test_batches*batch_size]
xtest_batches = np.split(Xtest, num_test_batches, 0)
ytest_batches = np.split(Ytest, num_test_batches, 0)
ltest_batches = np.split(Ltest, num_test_batches, 0)
Ntest=Xtest.shape[0]


model = ARAE(vocab_size = vocab_size,
             batch_size = batch_size,
             latent_size = latent_size,
             sample_size = sample_size,
             )


out_dir0="out_"+model_name
if not os.path.exists(out_dir0):
    os.makedirs(out_dir0)

total_st=time.time()

epochs=[79]

for epoch in epochs:
    out_dir=out_dir0+"/%d" %epoch
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file=out_dir+"/result_"+model_name+"_%d.txt" %epoch
    fp0=open(output_file,"w")
    model.restore(save_dir+"/model.ckpt-%d" %epoch)

    latent_vector_real=[]
    Y_real=[]
    smiles_real=[]
    latent_vector_fake=[]
    Y_fake=[]
    smiles_fake=[]

    total_accuracy=0
    total_cost1=0
    total_cost2=0
    total_cost3=0
    total_cost4=0

    for itest in range(num_test_batches):
        x = xtest_batches[itest]
        y = ytest_batches[itest]
        l = ltest_batches[itest]

        s = np.random.normal(0.0, 0.25, [batch_size, sample_size]).clip(-1.0,1.0)
        n = np.random.normal(0.0, dev, [batch_size, latent_size])

        Y, cost1, cost2, cost3, cost4, latent_vector, mol_encoded0 = model.test(x, y, l, s, n)
        Y_real.append(Y)
        latent_vector_real.append(mol_encoded0)

        norm = np.linalg.norm(latent_vector, 2, axis=-1)
        line_out="norm check : %f %f %f %f %f\n" %tuple(norm[0:5])
        fp0.write(line_out)
        line_out="test loss : %f %f %f %f\n" %(cost1,cost2,cost3,cost4)
        fp0.write(line_out)

        total_cost1+=cost1
        total_cost2+=cost2
        total_cost3+=cost3
        total_cost4+=cost4

        accuracy = cal_accuracy(x, Y, l)
        total_accuracy+=accuracy
        line_out="Accuracy : %f\n" %accuracy
        fp0.write(line_out)

        for i in range (2):
            s1_1 = convert_to_smiles(x[i,:], np.array(char_list))
            s1_2 = convert_to_smiles(Y[i,:], np.array(char_list))
            line_out=s1_1[1:]+"\n"+s1_2[:-1]+"\n"
            fp0.write(line_out)
        smiles=[]
        for i in range(0,len(Y)):
            s2= convert_to_smiles(Y[i,:],np.array(char_list))
            smiles+=[s2]
        smiles_real+=smiles

        decoder_state = model.get_decoder_state()
        fp0.write('**********************************************\n')
        s = np.random.normal(0.0, 0.25, [batch_size, sample_size]).clip(-1.0,1.0)
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

        for s in smiles[:10]:
            fp0.write(s)

    avg_cost1=total_cost1/num_test_batches
    avg_cost2=total_cost2/num_test_batches
    avg_cost3=total_cost3/num_test_batches
    avg_cost4=total_cost4/num_test_batches

    line_out="Total loss : %f %f %f %f" %(avg_cost1,avg_cost2,avg_cost3,avg_cost4)
    fp0.write(line_out)

    avg_accuracy=float(total_accuracy)/Ntest
    line_out="Reconstruction Accuracy : %f" %avg_accuracy
    fp0.write(line_out)

    print ("epoch:", epoch, "Reconstruction Accuracy:", avg_accuracy)


    latent_vector_real=np.array(latent_vector_real,dtype="float32").reshape(-1,latent_size)
    latent_vector_fake=np.array(latent_vector_fake,dtype="float32").reshape(-1,latent_size)
    Y_real=np.array(Y_real,dtype="int32").reshape(-1,seq_length+1)
    Y_fake=np.array(Y_fake,dtype="int32").reshape(-1,seq_length)
    outfile=out_dir+"/Zreal.npy"
    np.save(outfile,latent_vector_real)
    outfile=out_dir+"/Zfake.npy"
    np.save(outfile,latent_vector_fake)
    outfile=out_dir+"/Yreal.npy"
    np.save(outfile,Y_real)
    outfile=out_dir+"/Yfake.npy"
    np.save(outfile,Y_fake)

    outfile=out_dir+"/smiles_real.txt"
    fp_out=open(outfile,'w')
    for line in smiles_real:
        line_out=line+"\n"
        fp_out.write(line_out)
    fp_out.close()

    outfile=out_dir+"/smiles_fake.txt"
    fp_out=open(outfile,'w')
    for line in smiles_fake:
        line_out=line+"\n"
        fp_out.write(line_out)
    fp_out.close()


total_et=time.time()
print ("total_time : ", total_et-total_st)

