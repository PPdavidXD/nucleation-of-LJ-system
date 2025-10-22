#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1
#PBS -q gpu
#PBS -N ydeng_LJ
#PBS -o test.out
#PBS -e test.err
cd $PBS_O_WORKDIR


#source /work/jzhang/softwares/lammps_mace_a100/bin/activate
source /work/jzhang/softwares/mambaforge/envs/MACE/bin/activate
export PYTHONPATH=$PYTHONPATH:/work/jzhang/WorkDir/GNNCV/mlcolvar_git


#source /home/ppdavid/deepmd2.2.7_plugin/sourceme.sh


#sleep 7200

#              lr decayrate weightdecay gamma alpha nn1 nn2 
python getmodel.py