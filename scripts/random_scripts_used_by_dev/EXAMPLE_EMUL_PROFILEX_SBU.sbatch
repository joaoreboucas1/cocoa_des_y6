#!/bin/bash

#SBATCH --job-name=PR
#SBATCH --output=PR-%A_%a.out
#SBATCH -e PR-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=75
#SBATCH --cpus-per-task=1
#SBATCH --partition=hbm-long-96core
#SBATCH --time=36:00:00

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
declare -i NY=100 # so from 0-99 is all about PROFILE1, 100-199 all about PROFILE2
if [ -z "${ROOTDIR}" ]; then
  declare -i tmp=$(( $SLURM_ARRAY_TASK_ID % NY ))
  sleep $(( 5 + $tmp*15 )) # help avoid different scripts running start_cocoa.sh simultaneously
                           # clash can still happen when one script runs start_cocoa.sh, the other
                           # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                           # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=1

# the parameters are 
# As_1e9    ns        H0      omegab  omegam  
# LSST_DZ_S1 LSST_DZ_S2 LSST_DZ_S3 LSST_DZ_S4 LSST_DZ_S5 
# LSST_A1_1 LSST_A1_2 
# LSST_M1 LSST_M2 LSST_M3 LSST_M4 LSST_M5
# We advise factor ~ 3 for parameters that are well constrained by the data (...) 
# If a parameter is poorly constrained or cov is not given, we recommend factor ~< 1
declare -a factor=( "3" "1.0" "1.0" "1.0" "3" 
                    "3" "3" "3" "3" "3" 
                    "3" "3" "3" "3" "3" 
                    "3" "3" )           
declare -a factor2=( "3" "3.0" "3.0" "3.0" "3" 
                     "3" "3" "3" "3" "3" 
                     "3" "3" "3" "3" "3" 
                     "3" "3" )
declare -a grid=( factor1 factor2 )
declare -a nstws=( "400" "700" )
declare -i i=$(( $SLURM_ARRAY_TASK_ID / NY ))
declare -i j=$(( $SLURM_ARRAY_TASK_ID % NY ))
(( i < ${#grid[@]} )) || exit 0 # padding: skip unused slots
declare -n row="${grid[$i]}"
declare nstw="${nstws[$i]}"
(( j < ${#row[@]} )) || exit 0  # padding: skip unused slots
declare factor="${row[$j]}"

"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe --mca pml ^ucx \
  --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot \
  --map-by slot:pe=${OMP_NUM_THREADS} python \
  ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE$((i+1)).py \
    --root ./projects/lsst_y1/ --cov "chains/EXAMPLE_EMUL_MCMC$((i+1)).covmat" \
    --outroot "EXAMPLE_EMUL_PROFILE$((i+1))" --factor "$factor" \
    --nstw $nstw --numpts 15 --profile "${j}" \
    --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN${i}.txt"
