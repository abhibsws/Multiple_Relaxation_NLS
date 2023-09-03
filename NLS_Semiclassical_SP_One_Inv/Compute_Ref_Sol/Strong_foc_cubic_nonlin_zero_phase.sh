#!/bin/bash 


#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -J MyJob_Strong_foc_zero
#SBATCH -o MyJob_Strong_foc_zero.%J.out
#SBATCH -e MyJob_Strong_foc_zero.%J.err
#SBATCH --time=100:00:00


module load anaconda3

start=`date +%s.%N`

python Ref_Sol_Strong_Foc_Zero_Phase.py 


end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo $runtime 




