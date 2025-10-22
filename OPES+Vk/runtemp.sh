#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -q cpu
#PBS -N ydeng_LJ
#PBS -o test.out
#PBS -e test.err
cd $PBS_O_WORKDIR


source /home/ydeng/package/ydengsourceme2.sh
export PLUMED_NUM_THREADS=1



function RUN {

dir=run1
mkdir ${dir}
cd ${dir}
rm bck*
cp ../start.lmp .

mpirun -np 64 lmp -in start.lmp -v tmy 67 -v dataname ../start  -v plumedfilename ../plumed.start.init.dat

cd ..
}



RUN


