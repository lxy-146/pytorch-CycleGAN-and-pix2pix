import numpy as np
import math
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 同ssim
def lxy_psnr(img1, img2):
    # mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    # if mse == 0:
    #     return 100
    # PIXEL_MAX = 255.0
    # return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)
    psnr_value = psnr(img1, img2)
    return psnr_value

# 图像每个元素是0-255
def lxy_ssim(img1, img2):
    ssim_value = ssim(img1, img2, gaussian_weights=True, use_sample_covariance=False, data_range=255.0)
    return ssim_value
# 输入是两个图像列表，numpy格式。返回是图像列表的平均指标
def lxy_ssim_psnr(gtr_imgs, gen_imgs):
    ssims, psnrs = [], []
    for gtr_img, gen_img in zip(gtr_imgs, gen_imgs):
        ssims.append(lxy_ssim(gtr_img, gen_img))
        psnrs.append(lxy_psnr(gtr_img, gen_img))
    return np.mean(np.array(ssims)), np.mean(np.array(psnrs))

def main():
    img1 = []
    img2 = []
    logdir = 'test2_norm'
    files = os.listdir(f'./output/{logdir}/A')
    for file in files:
        filename = file.split('.')[0]
        x1 = Image.open(f'./output/{logdir}/B/{filename}.jpg')
        # x1 = Image.open(f'./dataset/US/test/A/{filename}.png')
        x2 = Image.open(f'./dataset/US/test/B/{filename}.png')
        x1 = x1.convert('L')
        x2 = x2.convert('L')
        x1 = np.array(x1)
        x2 = np.array(x2)
        # print(x1)
        # x1 = np.int32(x1)
        # x2 = np.int32(x2)
                # x1 = np.array(x1) / 255
                # x2 = np.array(x2) / 255
        img1.append(np.array(x1))
        img2.append(np.array(x2))
    ssim, psnr = lxy_ssim_psnr(img1, img2)
    print(ssim, psnr)

if __name__ == '__main__':
    main()