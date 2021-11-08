#!/bin/bash 
#PBS -N akash_j_idx_1 
#PBS -q small 
#PBS -l nodes=1:ppn=20 
#PBS -j oe 

# Load anaconda, which provides Python 3 and the
# necessary libraries (such as NumPy, matplotlib, etc.)
conda init bash
conda activate base

# Run my code

for ((i=0; i<=1275; i++))
do
     python ~/iitk_data/JJAS/akash_idx1_j.py "$i" 
     python ~/iitk_data/JJAS/akash_jjas_idx2.py "$i" 
done
