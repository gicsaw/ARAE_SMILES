from model.CARAE import ARAE
#from utils.utils import *
import numpy as np
import os, sys
import time
import tensorflow as tf
import collections
import copy
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
    Pfile=data_dir+"P"+data+".npy"
    Xdata=np.load(Xfile)
    Ydata=np.load(Yfile)
    Ldata=np.load(Lfile)
    Pdata=np.load(Pfile)
    return Xdata,Ydata,Ldata,Pdata

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

#input properties, [logP,SAS,TPSA]
#task_val=np.array([1.5,2,30])
if len(sys.argv)<=3:
    print("python test_n_CARAE_con_logP_SAS_TPSA.py logP SAS TPSA ")
    sys.exit()

logP_set=float(sys.argv[1])
SAS_set=float(sys.argv[2])
TPSA_set=float(sys.argv[3])
task_val=np.array([logP_set,SAS_set,TPSA_set])

print(task_val)

data_dir="./data/ZINC/"
model_name="CARAE_logP_SAS_TPSA"
save_dir="./save/"+model_name

out_dir0="out_"+model_name+"G_%d_%d_%d" %(int(logP_set*10),int(SAS_set),int(TPSA_set))
if not os.path.exists(out_dir0):
    os.makedirs(out_dir0)

Xtest,Ytest,Ltest,Ptest=read_data(data_dir,"test")
property_task=Ptest.shape[1]
task_nor=np.array([10.0,10.0,150.0])
task_low=np.array([-1.0,1.0,0.0])
task_high=np.array([5.0,8.0,150.0])
task_low=task_low/task_nor
task_high=task_high/task_nor



task_val=task_val/task_nor


num_test_batches = int(len(Xtest)/batch_size)

Xtest = Xtest[0:num_test_batches*batch_size]
Ytest = Ytest[0:num_test_batches*batch_size]
Ltest = Ltest[0:num_test_batches*batch_size]
Ptest = Ptest[0:num_test_batches*batch_size]
xtest_batches = np.split(Xtest, num_test_batches, 0)
ytest_batches = np.split(Ytest, num_test_batches, 0)
ltest_batches = np.split(Ltest, num_test_batches, 0)
ptest_batches = np.split(Ptest, num_test_batches, 0)
Ntest=Xtest.shape[0]

Ptest2 = copy.deepcopy(Ptest)
np.random.shuffle(Ptest2[:,0])
np.random.shuffle(Ptest2[:,1])
np.random.shuffle(Ptest2[:,2])
p_batches2 = np.split(Ptest2, num_test_batches, 0)

model = ARAE(vocab_size = vocab_size,
             batch_size = batch_size,
             latent_size = latent_size,
             sample_size = sample_size,
             property_task = property_task
             )


total_st=time.time()

epochs=[39]

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
    P_fake=[]
    smiles_fake=[]

    total_accuracy=0
    total_cost1=0
    total_cost2=0
    total_cost3=0
    total_cost4=0
    total_cost5=0
    total_cost6=0

    for itest in range(num_test_batches):
        x = xtest_batches[itest]
        y = ytest_batches[itest]
        l = ltest_batches[itest]
        p = ptest_batches[itest]

        s = np.random.normal(0.0, 0.25, [batch_size, sample_size]).clip(-1.0,1.0)
        n = np.random.normal(0.0, dev, [batch_size, latent_size])
        Y, cost1, cost2, cost3, cost4,cost5, cost6, latent_vector, mol_encoded0 = model.test(x, y, l, s, n,p)
        Y_real.append(Y)
        latent_vector_real.append(mol_encoded0)

        norm = np.linalg.norm(latent_vector, 2, axis=-1)
        line_out="norm check : %f %f %f %f %f\n" %tuple(norm[0:5])
        fp0.write(line_out)

        line_out="test loss : %f %f %f %f %f %f\n" %(cost1,cost2,cost3,cost4,cost5,cost6)
        fp0.write(line_out)

        total_cost1+=cost1
        total_cost2+=cost2
        total_cost3+=cost3
        total_cost4+=cost4
        total_cost5+=cost5
        total_cost6+=cost6

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

#        p = p_batches2[itest]
#        cp = np.random.uniform(task_low,task_high, [batch_size, property_task])
        p=np.empty([batch_size,property_task])
        p[:,0].fill(task_val[0])
        p[:,1].fill(task_val[1])
        p[:,2].fill(task_val[2])

        P_fake.append(p)

        latent_vector = model.generate_latent_vector(s)
        latent_vector_fake.append(latent_vector)

        start_token = np.array([char_list.index('X') for i in range(batch_size)])
        start_token = np.reshape(start_token, [batch_size, 1])
        length = np.array([1 for i in range(batch_size)])
        smiles = ['' for i in range(batch_size)]
        Y=[]
        for i in range(seq_length):
            m, state = model.generate_molecule(start_token, latent_vector, length, p, decoder_state)
            decoder_state = state
            start_token = np.argmax(m,2)
            Y.append(start_token[:,0])
            smiles = [s + str(char_list[start_token[j][0]]) for j,s in enumerate(smiles)]
        Y=list(map(list,zip(*Y)))
        Y_fake.append(Y)
        smiles_fake+=smiles

#        for s in smiles[:10]:
#            print (s)

    avg_cost1=total_cost1/num_test_batches
    avg_cost2=total_cost2/num_test_batches
    avg_cost3=total_cost3/num_test_batches
    avg_cost4=total_cost4/num_test_batches
    avg_cost5=total_cost5/num_test_batches
    avg_cost6=total_cost6/num_test_batches

    line_out="Total loss : %f %f %f %f %f %f\n" %(avg_cost1,avg_cost2,avg_cost3,avg_cost4,avg_cost5,avg_cost6)
    fp0.write(line_out)

    avg_accuracy=float(total_accuracy)/Ntest
    out_line="Reconstruction Accuracy: %f \n" %avg_accuracy

    fp0.write(line_out)

    print ("epoch:", epoch, "Reconstruction Accuracy:", avg_accuracy)


    latent_vector_real=np.array(latent_vector_real,dtype="float32").reshape(-1,latent_size)
    latent_vector_fake=np.array(latent_vector_fake,dtype="float32").reshape(-1,latent_size)
    P_fake=np.array(P_fake,dtype="float32").reshape(-1,property_task)
#    Y_real=np.array(Y_real,dtype="int32").reshape(-1,seq_length+1)
    Y_real=np.array(Y_real,dtype="int32").reshape(-1,seq_length)
    Y_fake=np.array(Y_fake,dtype="int32").reshape(-1,seq_length)
    outfile=out_dir+"/Zreal.npy"
    np.save(outfile,latent_vector_real)
    outfile=out_dir+"/Zfake.npy"
    np.save(outfile,latent_vector_fake)
    outfile=out_dir+"/Pfake.npy"
    np.save(outfile,P_fake)
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




