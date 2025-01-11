import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor, Normalize
from torchvision.utils import save_image

from skimage.metrics import structural_similarity
from skimage.io import imread
import matplotlib.pyplot as plt



import os

import argparse
import numpy as np
import random
from PIL import Image

from configs import get_img_size, get_crop_ratio
from utils import Recorder
from pathlib import Path

def count_files_in_directory(file_path):
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"The provided path does not exist: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"The provided path is not a file: {file_path}")

    parent_dir = file_path.parent
    
    files_count = sum(1 for item in parent_dir.iterdir() if item.is_file())
    
    return files_count

def calculate_psnr(original, reconstructed, data_range=1.0):
    original = original.float()
    reconstructed = reconstructed.float()

    mse = torch.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float('inf')

    data_range_tensor = torch.tensor(data_range, dtype=mse.dtype, device=mse.device)

    psnr_value = 20 * torch.log10(data_range_tensor) - 10 * torch.log10(mse)
    
    return psnr_value.item()

def calculate_color_ssim(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")    
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    ssim_value = structural_similarity(img1, img2, multichannel=True, channel_axis=2, data_range=1.0)
    return ssim_value


def main():
   

    parser = argparse.ArgumentParser(description='MoWE Inference for One Image')

    parser.add_argument('--task', default='derain', type=str, help='task name')
    parser.add_argument('--dataset', default='mawsim', type=str, help='dataset name')
    parser.add_argument('--img-path', type=str, required=True, help='image path')
    parser.add_argument('--img-size', type=int, nargs='+', default=None, help='img size')
    parser.add_argument('--crop-ratio', type=float, nargs='+', default=None, help='crop ratio')
    parser.add_argument('--model-path', default=None, type=str, help='inference model path')
    parser.add_argument('--model-name', default='mowe', type=str, help='inference model name')
    parser.add_argument('--exp', default='infer', type=str, help='output dir')
    parser.add_argument('--clean-path', type=str, required=True, help='image path')
    parser.add_argument('--visualization', default=None, choices=['attn'], 
                                            type=str, help='Visualization choice')
    parser.add_argument('--gpu-list', type=int, nargs='+', default=None, help='cuda list')
    parser.add_argument('--workers', default=15, type=int, help='num-workers')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()
    device = 'cuda' if len(args.gpu_list) > 0 else 'cpu'
    print('{}:'.format(device), args.gpu_list)
    gpus = ','.join([str(i) for i in args.gpu_list])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!\n")
    output_dir = args.exp
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('==> Output dir:')
    print(output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print('\n==> Building model..')
    print('\n==> Load inference model from {}..'.format(args.model_path))
    model = torch.load(args.model_path, map_location=torch.device('cpu'))['net']
    if args.visualization == 'attn':
        model = Recorder(model)
    model = model.to(device)
    if 'cuda' in device:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        cudnn.benchmark = True
    model.eval()
    sum_psnr = 0
    sum_ssim = 0
    ran = count_files_in_directory(args.img_path)
    for i in range(ran):
        print('\n==> Preparing data..')
        im_path = args.img_path
        im_path = im_path.split('/')[:-1] + ["{:04d}".format(i) + ".jpg"]
        im_path = '/'.join(im_path)
        print(f"im_path is {im_path}.")
        clean_path = args.clean_path
        clean_path = clean_path.split('/')[:-1] + ["{:04d}".format(i) + ".jpg"]
        clean_path = '/'.join(clean_path)
        snow = Image.open(im_path).resize((720, 480))
        input_name = f"{im_path.split('/')[-2]}_{im_path.split('/')[-1]}"
        print(f"input name is {input_name}.")
        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        snow = normalize(to_tensor(snow)).to(device).unsqueeze(dim=0)
        clean_image = Image.open(clean_path).resize((720, 480))
        clean_image = to_tensor(clean_image).to(device)
        if i == 0:
            reference = torch.zeros((1, 3, 480, 720)).to(device)
        else:
            original_img_path = args.img_path
            sub_dir = original_img_path.split('/')[-2]  
            new_filename = "{:04d}.jpg".format(i-1)
            new_filename = sub_dir + "_" + new_filename
            print(f"new filename is {new_filename}.")
            new_path = os.path.join(output_dir, new_filename)
            reference = Image.open(new_path).resize((720, 480))
            reference = normalize(to_tensor(reference)).to(device).unsqueeze(dim=0)
        inputs = torch.cat((snow, reference), dim=1)
        inputs = inputs.to(device)
        count_inputs = torch.zeros_like(inputs)
        outputs = torch.zeros((1, 3, 480, 720)).to(device)
        count_outputs = torch.zeros((1, 3, 480, 720)).to(device)
        h, w = inputs.shape[-2], inputs.shape[-1]
        h_out, w_out = outputs.shape[-2], outputs.shape[-1]
        if args.crop_ratio is not None:
            crop_ratio = [1.0/args.crop_ratio[1], 1.0/args.crop_ratio[0]]
        else:
            crop_ratio = get_crop_ratio(dataset_name=args.dataset)
        h_d, w_d = int(h * crop_ratio[0]), int(w * crop_ratio[1])
        h_d_out, w_d_out = int(h_out * crop_ratio[0]), int(w_out * crop_ratio[1])
        num_h = int(1.0/crop_ratio[0])
        num_w = int(1.0/crop_ratio[1])
        for i in np.arange(0, num_h-0.5, 0.5):
            for j in np.arange(0, num_w-0.5, 0.5):
                i, j = float(i), float(j)
                if i == 0.0 and j == 0.0:
                    inputs_crop = inputs[:, :, 0:h_d, 0:w_d]
                    count_inputs[:, :, 0:h_d, 0:w_d] += 1
                    continue
                else:
                    inputs_crop_temp = inputs[:, :, int(i*h_d): int((i+1)*h_d), int(j*w_d): int((j+1)*w_d)]
                    count_inputs[:, :, int(i*h_d):int((i+1)*h_d), int(j*w_d):int((j+1)*w_d)] += 1
                    inputs_crop = torch.cat([inputs_crop, inputs_crop_temp], dim=0)
        task_idx = None
        if args.visualization == 'attn':
            (_outputs, cls_outputs, weights_list, l_aux), attns = model(inputs_crop, task_idx)
            print(attns.shape)
        else:
            _outputs, cls_outputs, weights_list, l_aux = model(inputs_crop, task_idx)
        bs = outputs.shape[0]
        for _i, i in enumerate(np.arange(0, num_h-0.5, 0.5)):  # _i: 个数, i: 位置
            for _j, j in enumerate(np.arange(0, num_w-0.5, 0.5)):
                i, j = float(i), float(j)
                outputs[:, :, int(i*h_d_out): int((i+1)*h_d_out), int(j*w_d_out): int((j+1)*w_d_out)] += \
                    _outputs[int((_i*(2*num_w-1)+_j)*bs): int((_i*(2*num_w-1)+_j+1)*bs), :, :, :]
                count_outputs[:, :, int(i*h_d_out): int((i+1)*h_d_out), int(j*w_d_out): int((j+1)*w_d_out)] += 1
        outputs = torch.div(outputs, count_outputs)
        rebuild_image = outputs[0]
        psnr = calculate_psnr(clean_image, rebuild_image)
        ssim = calculate_color_ssim(clean_image, rebuild_image)
        sum_psnr += psnr
        sum_ssim += ssim
        print(f"psnr is {psnr}.")
        print(f"sum_psnr is {sum_psnr}.")
        save_image(outputs[0], os.path.join(output_dir, input_name))
    ave_psnr = sum_psnr / float(ran)
    ave_ssim = sum_ssim / float(ran)
    print(f"ave_psnr is {ave_psnr}.")
    print(f"ave_ssim is {ave_ssim}.")

    # print(weights_list[0].shape)  # [35, 1024, 16] = [(2h-1)*(2w-1), h*w/p*p, num_expert]
    # print(weights_list[1].shape)

    # torch.set_printoptions(precision=4, sci_mode=False)
    # weight_0 = weights_list[0].mean(dim=0).mean(dim=0)
    # print(torch.round(weight_0.detach().cpu(), decimals=4))

    # weight_1 = weights_list[1].mean(dim=0).mean(dim=0)
    # print(torch.round(weight_1.detach().cpu(), decimals=4))

if __name__ == '__main__':
    main()