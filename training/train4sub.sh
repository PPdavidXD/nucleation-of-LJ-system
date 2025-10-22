#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1
#PBS -q gpu_a100
#PBS -N LJtrain
#PBS -o test.out
#PBS -e test.err
cd $PBS_O_WORKDIR


#source /work/jzhang/softwares/lammps_mace_a100/bin/activate
source /work/jzhang/softwares/mambaforge/envs/MACE/bin/activate
export PYTHONPATH=$PYTHONPATH:/work/jzhang/WorkDir/GNNCV/mlcolvar_git



#                          lr decayrate weightdecay gamma alpha nn1 nn2 id   rditer initer


python training_logv4_nobw.py 1e-3   0.99993        1e-5   1e4  5e-4  80  40  in1_1s1        3    1 >   resultin1_1s1.dat 2>&1   
python training_logv4_nobw.py 1e-3   0.99993        1e-5   1e4  5e-4  80  40  in1_2s1        3    2 >   resultin1_2s1.dat 2>&1   
python training_logv4_nobw.py 1e-3   0.99993        1e-5   1e4  5e-4  80  40  in1_3s1        3    3 >   resultin1_3s1.dat 2>&1   
python training_logv4_nobw.py 1e-3   0.99993        1e-5   1e4  5e-4  80  40  in1_4s1        3    4 >   resultin1_4s1.dat 2>&1   
python training_logv4_nobw.py 1e-3   0.99993        1e-5   1e4  5e-4  80  40  in1_5s1        3    5 >   resultin1_5s1.dat 2>&1   



