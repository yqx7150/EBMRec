# EBMRec
MRI Reconstruction Using Energy-Based Model

# Pretrained Models
We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1spFtJLw-5GFwg9rHB015yA). key number is "gygy "and unzip into the folder cachedir.

# Train
if you want to train the code，please
```bash
python3 EBM_train.py --exp=fastMRI256 --dataset=fastMRI --num_steps=50 --batch_size=16 --step_lr=100 --lr=3e-4 --zero_kl --replay_batch --ResNet128_model --cclass --swish_act

```

# Acknowledgement
The implementation is based on this repository: https://github.com/openai/ebm_code_release.
