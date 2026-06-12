#!/bin/bash
#SBATCH -A desicollab
#SBATCH -C cpu
#SBATCH -q overrun              # Changed from regular to overrun to bypass the balance check
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 64
#SBATCH -J waw0_mcmc
#SBATCH -o %x_%j.out

cd $SCRATCH
module load python
conda activate cosmo311
cd ~/DESIxSO

# 1. Define folder names for BOTH locations
FOLDER_NAME="emulated_lensing_ratios_waw0_source_bin1_all_lens_bins_wa_w0_omega_m_likelihood"
SCRATCH_DIR="${SCRATCH}/chains/${FOLDER_NAME}"
HOME_DIR="${HOME}/DESIxSO/chains/${FOLDER_NAME}"

# 2. CLEANUP: Wipe out any old, stale runs from both directories so we start completely fresh
rm -rf ${SCRATCH_DIR}/*
rm -rf ${HOME_DIR}/*

# Ensure the folder structures exist
mkdir -p $SCRATCH_DIR
mkdir -p $HOME_DIR

export COBAYA_USE_FILE_LOCKING=False
export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# 3. Run Cobaya directly inside the high-speed SCRATCH directory
time srun -n 4 -c 64 --cpu_bind=cores bash -c '
export PYTHONPATH="${HOME}/DESIxSO:${PYTHONPATH}"
TASK_YAML="run_config_${SLURM_PROCID}.yaml"
cp yaml/emulated_lensing_ratios_waw0_source_bin1_all_lens_bins_wa_w0_omega_m_likelihood.yaml $TASK_YAML
MY_SEED=$((1234 + SLURM_PROCID))
sed -i "s/seed:.*/seed: ${MY_SEED}/g" $TASK_YAML

$(which cobaya-run) $TASK_YAML -o '${SCRATCH_DIR}'/chain_task_${SLURM_PROCID}
rm -f $TASK_YAML
'

# 4. BACKUP: Copy everything safely over to your HOME directory folder
echo "MCMC execution finished. Backing up chains from Scratch to Home..."
cp -r ${SCRATCH_DIR}/* ${HOME_DIR}/
echo "Backup complete! Data is safe in both locations."
