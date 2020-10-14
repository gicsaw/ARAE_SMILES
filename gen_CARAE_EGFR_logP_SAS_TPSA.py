from model.CARAE_cla_reg import ARAE
#from utils.utils import *
import numpy as np
import os, sys
import time
import tensorflow as tf
import collections
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

#input properties, [logP,SAS,TPSA]
#task_val=np.array([1.5,2,30])
if len(sys.argv)<=4:
    print("python gen_CARAE_EGFR_logP_SAS_TPSA activity logP SAS TPSA ")
    sys.exit()
activity_set=float(sys.argv[1])
logP_set=float(sys.argv[2])
SAS_set=float(sys.argv[3])
TPSA_set=float(sys.argv[4])
task_val=np.array([activity_set,logP_set,SAS_set,TPSA_set])

print(task_val)

model_name="CARAE_EGFR_property"
save_dir="./save/"+model_name

out_dir0="out_"+model_name+"_%d_%d_%d" %(int(logP_set*10),int(SAS_set),int(TPSA_set))
if not os.path.exists(out_dir0):
    os.makedirs(out_dir0)

property_task=4
classification_task=1
regression_task=3

task_nor=np.array([1.0, 10.0,10.0,150.0])
task_low=np.array([0.0, -1.0,1.0,0.0])
task_high=np.array([1.0, 5.0,8.0,150.0])
task_low=task_low/task_nor
task_high=task_high/task_nor

task_val=task_val/task_nor

Ntest=10000
num_test_batches = int(Ntest/batch_size)


model = ARAE(vocab_size = vocab_size,
             batch_size = batch_size,
             latent_size = latent_size,
             sample_size = sample_size,
             classification_task = classification_task,
             regression_task = regression_task
             )


total_st=time.time()

epochs=[10]

for epoch in epochs:
    out_dir=out_dir0+"/%d" %epoch
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file=out_dir+"/result_"+model_name+"_%d.txt" %epoch
    fp0=open(output_file,"w")
    model.restore(save_dir+"/model.ckpt-%d" %epoch)

    latent_vector_fake=[]
    Y_fake=[]
    P_fake=[]
    smiles_fake=[]

    for itest in range(num_test_batches):

#        fp0.write('**********************************************\n')
        decoder_state = model.get_decoder_state()
        s = np.random.normal(0.0, 0.25, [batch_size, sample_size]).clip(-1.0,1.0)

#        p = p_batches2[itest]
#        cp = np.random.uniform(task_low,task_high, [batch_size, property_task])
        p=np.empty([batch_size,property_task])

        p[:,0].fill(task_val[0])
        p[:,1].fill(task_val[1])
        p[:,2].fill(task_val[2])
        p[:,3].fill(task_val[3])

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


    latent_vector_fake=np.array(latent_vector_fake,dtype="float32").reshape(-1,latent_size)
    P_fake=np.array(P_fake,dtype="float32").reshape(-1,property_task)
    Y_fake=np.array(Y_fake,dtype="int32").reshape(-1,seq_length)
    outfile=out_dir+"/Zfake.npy"
    np.save(outfile,latent_vector_fake)
    outfile=out_dir+"/Pfake.npy"
    np.save(outfile,P_fake)
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




