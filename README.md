# ARAE for molecular generation with SMILES representation
#

# Prerequisite:
python3
numpy
tensorflow
RDKit
SA_Score for RDKit
https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score

# Download:

git clone https://github.com/gicsaw/ARAE_SMILES

#  Training data preparation :
cd ARAE_SMILES

#For ZINC 
python data_char_ZINC.py train
python data_char_ZINC.py test

#For QM9
python data_char_QM9.py train
python data_char_QM9.py test

# training:
# ARAE with QM9 dataset
python train_ARAE_QM9.py

# ARAE with ZINC dataset
python train_ARAE_ZINC.py

# CARAE with ZINC dataset for logP, SAS, and TPSA
python train_CARAE_logP_SAS_TPSA.py

# Test and molecular generation:
#We prepared trained hyperparameters in save dir.

#Test for ARAE with QM9
python test_n_ARAE_QM9.py

#Test for ARAE with ZINC
python test_n_ARAE_ZINC.py

#Test for CARAE with ZINC (conditional)
python test_n_CARAE_con_logP_SAS_TPSA.py $logP $SAS $TPSA 

#Test for CARAE with ZINC (unconditional)
python test_n_CARAE_uncon_logP_SAS_TPSA.py

#Molecular generation for ARAE with QM9 
gen_ARAE_QM9.py

#Molecular generation for ARAE with ZINC
gen_ARAE_ZINC.py

#Molecular generation for CARAE with ZINC (conditional)
python gen_CARAE_con_logP_SAS_TPSA.py  $logP $SAS $TPSA


# References:

