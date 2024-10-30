import math, torch
import numpy as np
import jax.numpy as jnp
import imgaug.augmenters as iaa
from PIL import Image
from pathlib import Path
from torchvision import transforms
import imgaug.augmenters as iaa
import numpy as np
import torch
import math




THIS_DIR = Path(__file__).parent.resolve()
IMG_SIZE = 576
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_img = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ColorJitter(0, 0, 0),
    transforms.RandomHorizontalFlip(0),
    transforms.RandomVerticalFlip(0),
    transforms.RandomGrayscale(0),
    transforms.RandomAffine(0,
                            translate=(0, 0),
                            scale=(1.0, 1.0),
                            interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

transform_mask = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])




def generate_thr(img_shape, min=0, max=4):
    """
    SH: This function generates a random Perlin mask. The resulting mask has the
    same shape as the input image and consists of ones and zeros.
    """
    min_perlin_scale = min
    max_perlin_scale = max
    perlin_scalex = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scaley = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_noise_np = rand_perlin_2d_np((img_shape[1], img_shape[2]), (perlin_scalex, perlin_scaley))
    threshold = 0.5
    perlin_noise_np = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise_np)
    perlin_thr = np.where(perlin_noise_np > threshold, np.ones_like(perlin_noise_np), np.zeros_like(perlin_noise_np))
    return perlin_thr


def perlin_mask(img_shape, feat_size, min, max, mask_fg, flag=0):
    mask = np.zeros((feat_size, feat_size))
    while np.max(mask) == 0:
        # SH: variables `perlin_thr_1` and `perlin_thr_2` are the $m_1$ and $m_2$ from the paper (Sec. 3.3).
        perlin_thr_1 = generate_thr(img_shape, min, max)
        perlin_thr_2 = generate_thr(img_shape, min, max)
        temp = torch.rand(1).numpy()[0]
        # SH: the variable `temp` here is the $\alpha$ from the paper (Eq. 3).
        if temp > 2 / 3:
            perlin_thr = perlin_thr_1 + perlin_thr_2
            perlin_thr = np.where(perlin_thr > 0, np.ones_like(perlin_thr), np.zeros_like(perlin_thr))
        elif temp > 1 / 3:
            perlin_thr = perlin_thr_1 * perlin_thr_2
        else:
            perlin_thr = perlin_thr_1
        
        perlin_thr = torch.from_numpy(perlin_thr)
        perlin_thr_fg = perlin_thr * mask_fg
        down_ratio_y = int(img_shape[1] / feat_size)
        down_ratio_x = int(img_shape[2] / feat_size)
        mask_ = perlin_thr_fg
        mask = torch.nn.functional.max_pool2d(perlin_thr_fg.unsqueeze(0).unsqueeze(0), (down_ratio_y, down_ratio_x)).float()
        mask = mask.numpy()[0, 0]
    mask_s = mask
    if flag != 0:
        mask_l = mask_.numpy()
    if flag == 0:
        return mask_s
    else:
        return mask_s, mask_l


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    # SH: Not used:
    # tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1],
                                                  axis=1)
    dot = lambda grad, shift: (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])




if __name__ == '__main__':
    """
    Let's test creating a LAS-style anomaly using the images nominal, fg, and texture{.png}.
    """
    seed = 1

    nominal = Image.open(THIS_DIR.joinpath('./nominal.png')).convert('RGB')
    nominal = transform_img(nominal)

    texture = Image.open(THIS_DIR.joinpath('./texture.png')).convert('RGB')
    texture = transform_img(texture)

    fg = Image.open(THIS_DIR.joinpath('./fg.png'))
    fg = torch.ceil(transform_mask(fg)[0])

    
    mask_all = perlin_mask(nominal.shape, IMG_SIZE // 8, 0, 6, fg, 1)
    mask_s = torch.from_numpy(mask_all[0])
    mask_l = torch.from_numpy(mask_all[1])

    beta = np.random.normal(loc=0.5, scale=0.1)
    beta = np.clip(beta, .2, .8)

    # SH: This is Eq. 4 in the paper
    aug_image = nominal * (1 - mask_l) + (1 - beta) * texture * mask_l + beta * nominal * mask_l
    
    print(5)
