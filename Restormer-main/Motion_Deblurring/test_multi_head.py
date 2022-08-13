## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr_restormer.models.archs.mhrestormer_arch import MHRestormer
from skimage import img_as_ubyte
from pdb import set_trace as stx
from psnr_ssim import calculate_psnr_mh, calculate_ssim_mh

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/pdl631/Codes/Restormer-main/experiments/Deblurring_MHRestormer/models/net_g_latest.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/Deblurring_MHRestormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = MHRestormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

record_single_head_psnr = []
record_single_head_ssim = []
record_best_psnr = []
record_best_ssim = []

inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
tar_dir = os.path.join(args.input_dir, 'test', dataset, 'target')
image_names = os.listdir(inp_dir)
# files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
with torch.no_grad():
    # for file_ in tqdm(files):
    for image_name_ in tqdm(image_names):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        # img = np.float32(utils.load_img(file_))/255.
        img = np.float32(utils.load_img(os.path.join(inp_dir, image_name_)))/255.
        tar = utils.load_img(os.path.join(tar_dir, image_name_))
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:,:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 1, 3, 4, 2).squeeze(0).numpy()

        if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
            for h, restored_ in enumerate(restored):
                utils.save_img((os.path.join(result_dir, image_name_.split('.')[0]+f'_h{h}.png')), img_as_ubyte(restored_))
        else:
            restored_img = img_as_ubyte(restored)
            gt_img = tar

            psnrs, psnr_best, _ = calculate_psnr_mh(restored_img, gt_img, 0)
            ssims, ssim_best, _ = calculate_ssim_mh(restored_img, gt_img, 0)
            record_single_head_psnr.append(psnrs)
            record_single_head_ssim.append(ssims)
            record_best_psnr.append(psnr_best)
            record_best_ssim.append(ssim_best)

    print(np.mean(record_single_head_psnr, axis=0),
          np.mean(record_single_head_ssim, axis=0),
          np.mean(record_best_psnr),
          np.mean(record_best_ssim))
