supporting data for the paper
# The role of fluctuations in the nucleation process
Yuanpeng Deng, Peilin Kang, Xiang Xu, Hui Li, and Michele Parrinello

[![ArXiv](https://img.shields.io/badge/arXiv-2410.17029-lightblue)](https://doi.org/10.48550/arXiv.2503.20649)

The training of the models was based on the [mlcolvar library](https://github.com/luigibonati/mlcolvar), where the updated relevant code and updated example notebooks are available:
- [didactical example: Muller-brown](https://github.com/luigibonati/mlcolvar/blob/main/docs/notebooks/tutorials/cvs_committor.ipynb)
- [practical example: Alanine with distances](https://github.com/luigibonati/mlcolvar/blob/main/docs/notebooks/examples/ex_committor.ipynb)

The code for the application of the Kolmogorov bias in PLUMED is available in the file `pytorch_model_bias.cpp`


### Repo structure
The contents of the repository are organized as follows:

Folder training contains the necessary file for the training of the pt file that could be read by the Kolmogorov bias PLUMED plugin.
Folder OPES+Vk contains the file for using LAMMPS+PLUMED to run an OPES sampling with z(x) as CV in the Kolmogorov ensemble.

