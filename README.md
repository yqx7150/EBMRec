# EBMRec
MRI Reconstruction Using Energy-Based Model

## Visual illustration of the invertible medical image synthesis and fusion in variable augmentation manner
 <div align="center"><img src="https://github.com/yqx7150/EBMRec/blob/main/Figs/Fig1.png"  width = "400" height = "450"> </div>
 
## The training pipeline of iVAN
 <div align="center"><img src="https://github.com/yqx7150/EBMRec/blob/main/Figs/Fig2.png"> </div>
 
## Two visualization results of synthesizing from T1 to T2
 <div align="center"><img src="https://github.com/yqx7150/EBMRec/blob/main/Figs/Fig5.png"> </div>
 
## Three fusion results of T2-weighted MR and CT images
 <div align="center"><img src="https://github.com/yqx7150/EBMRec/blob/main/Figs/Fig6.png"> </div>

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
```


# Acknowledgement
The implementation is based on this repository: https://github.com/openai/ebm_code_release.
