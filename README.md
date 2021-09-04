# EBMRec
MRI Reconstruction Using Energy-Based Model

# Pretrained Models
We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1spFtJLw-5GFwg9rHB015yA). key number is "gygy "and unzip into the folder cachedir.

# Train
if you want to train the code
python3 EBM_train.py --exp=fastMRI256 --dataset=fastMRI --num_steps=50 --batch_size=16 --step_lr=100 --lr=3e-4 --zero_kl --replay_batch --ResNet128_model --cclass --swish_act


## Test
if you want to test the code, please 

```bash
not-p
python3.5 separate_ImageNet.py --model ncsn --runner Test_6ch_inpainting_rgbrgb_256 --config anneal_bedroom_6ch_256.yml --doc your-checkpoint --test --image_folder your-save-path

or
python3.5 separate_ImageNet.py --model ncsn --runner Test_3ch_inpainting_rgbrgb_256 --config anneal_bedroom_3ch_256.yml --doc your-checkpoint --test --image_folder your-save-path
python3.5 separate_ImageNet.py --model ncsn --runner Test_6ch_inpainting_rgbrgb_256 --config anneal_bedroom_6ch_256.yml --doc your-checkpoint --test --image_folder your-save-path
....
```
key number is "HGM " 

# Acknowledgement
The implementation is based on this repository: https://github.com/openai/ebm_code_release.
