# MRI Reconstruction Using Energy-Based Model
The Code is created based on the method described in the following paper: MRI Reconstruction Using Energy-Based Model.

## Overview of the MRI reconstruction.
 <div align="center"><img src="https://github.com/yqx7150/EBMRec/blob/main/Figs/Fig1.png" width = "815" height = "470"> </div>
 
## Detailed comparison of characteristics and structures in the flow chart of GAN and EBM. 
 <div align="center"><img src="https://github.com/yqx7150/EBMRec/blob/main/Figs/Fig2.png" width = "781" height = "450"> </div>
 
## Complex-valued reconstruction results on brain images at R=3 various 1D Cartesian under-sampling percentages in 15 coils parallel imaging.
 <div align="center"><img src="https://github.com/yqx7150/EBMRec/blob/main/Figs/Fig5.png" width = "844" height = "556"> </div>
 
## Complex-valued reconstruction results on brain image at R=6 pseudo random sampling in 12 coils parallel imaging.
 <div align="center"><img src="https://github.com/yqx7150/EBMRec/blob/main/Figs/Fig6.png" width = "893" height = "558"> </div>

# Pretrained Models
We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1spFtJLw-5GFwg9rHB015yA). key number is "gygy "and unzip into the folder cachedir.

# Train
If you want to train the code，please
```bash
python3 EBM_train.py --exp=fastMRI256 --dataset=fastMRI --num_steps=50 --batch_size=16 --step_lr=100 --lr=3e-4 --zero_kl --replay_batch --ResNet128_model --cclass --swish_act
```
All code supports horovod execution, so model training can be increased substantially by using multiple different workers by running each command.
```bash
mpiexec -n <worker_num>  <command>
For example: "mpiexec --oversubscribe -n 1" or "mpiexec --oversubscribe -n 4"
```

# Test
If you want to test the code，please
```bash
python3 EBM_test.py --exp=siat256 --resume_iter=164250 --step_lr=300 --swish_act

python3 EBM_test_ddp.py --exp=siat256 --resume_iter=164250 --step_lr=50 --swish_act

python3 EBM_test_modl.py --exp=siat256 --resume_iter=164250 --step_lr=10 --swish_act
```

# Acknowledgement
The implementation is based on this repository: https://github.com/openai/ebm_code_release.
