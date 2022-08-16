# multi-output-deblur
The official implementation of paper "Multi-Outputs Is All You Need For Deblur".

## Environment

- python 3.7+

- pytorch 1.7+

## Install & Usage

### NAFNet & HINet

This part of code is cloned and modified from https://github.com/megvii-research/NAFNet, and some scripts are copied from https://github.com/megvii-model/HINet. For more information and guide, please see the docs in those repos.

#### Install

```
cd NAFNet-main/
python setup.py develop --no_cuda_ext
```

#### Train & test

```
# train
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/GoPro/MHNAFNet-width64-combinate-heads.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/GoPro/MHHINet.yml --launcher pytorch

# test
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/test.py -opt options/test/GoPro/MHNAFNet-width64.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/test.py -opt options/test/GoPro/MHHINet.yml --launcher pytorch
```

### Restormer

This part of code is cloned and modified from https://github.com/swz30/Restormer. For more information and guide, please see the docs in that repo.

#### Install

```
cd Restormer-main/
python setup.py develop --no_cuda_ext
```

#### Train & test

```
# train
./train.sh Motion_Deblurring/Options/Deblurring_MHRestormer.yml
# test
cd Motion_Deblurring
python test_multi_head.py --dataset GoPro
```

### MPRNet

This part of code is cloned and modified from https://github.com/swz30/MPRNet. For more information and guide, please see the docs in that repo.

#### Train & test

```
cd MPRNet-main/Deblurring
# train
python train_multi_head.py
# test
python test_multi_head.py --dataset GoPro
```
