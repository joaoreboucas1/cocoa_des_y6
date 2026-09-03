#!/bin/bash

#SBATCH --job-name=NAU
#SBATCH --output=NAU-%A_%a.out
#SBATCH -e NAU-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=85
#SBATCH --cpus-per-task=1
#SBATCH --partition=hbm-long-96core
#SBATCH --time=24:00:00

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
if [ -z "${ROOTDIR}" ]; then
  sleep $(( 5 + SLURM_ARRAY_TASK_ID*25 )) # help avoid different scripts running start_cocoa.sh simultaneously
                                          # clash can still happen when one script runs start_cocoa.sh, the other
                                          # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                                          # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=1

"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe --mca pml ^ucx \
  --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot \
  --map-by slot:pe=${OMP_NUM_THREADS} python -m mpi4py.futures \
  ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS${SLURM_ARRAY_TASK_ID}.py \
    --root ./projects/lsst_y1/ \
    --outroot "EXAMPLE_EMUL_NAUTILUS${SLURM_ARRAY_TASK_ID}" \
    --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5
