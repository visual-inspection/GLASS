import math, torch, random
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import flax.linen as nn
import imgaug.augmenters as iaa
from jax._src.basearray import ArrayLike
from typing import Callable
from PIL import Image
from pathlib import Path



THIS_DIR = Path(__file__).parent.resolve()





class Perlin_generator:
    def __init__(self, seed: int) -> None:
        """
        Initializes all randoms (Python, Numpy, Jax) using the given seed.
        Also creates an initial key for JAX.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._key_threshold, self._key_perlin2d, self._key_alpha = jrand.split(key=jrand.PRNGKey(seed), num=3)

        torch.use_deterministic_algorithms(mode=True)
    
    @property
    def key_threshold(self) -> jrand.PRNGKey:
        self._key_threshold, new_key = jrand.split(key=self._key_threshold, num=2)
        return new_key
    
    @property
    def key_perlin2d(self) -> jrand.PRNGKey:
        self._key_perlin2d, new_key = jrand.split(key=self._key_perlin2d, num=2)
        return new_key
    
    @property
    def key_alpha(self) -> jrand.PRNGKey:
        self._key_alpha, new_key = jrand.split(key=self._key_alpha, num=2)
        return new_key
    

    def generate_threshold_mask(self, img_shape: tuple[int, int, int], min: int=0, max: int=4, threshold: float=0.5):
        """
        Generates 2D Perlin noise, applies randomly a +/-90° rotation to it, then returns a mask
        (ones and zeros) of it. The mask is equal to one where the noise is greater than the threshold.
        """
        assert len(img_shape) == 3

        key = self.key_threshold
        scale_x, scale_y = (2 ** jrand.randint(key=key, shape=(2,), minval=min, maxval=max)).astype(jnp.int32)
        noise = self.rand_perlin_2d(shape=img_shape[1:], res=(scale_x.item(), scale_y.item()))

        key = self.key_threshold # Make a new key.
        # SH: I refactored this part. We rotate the array by 0 or +/-90°.
        choice = jrand.choice(key=key, a=jnp.asarray([0,1,3]), replace=False, shape=()).item()
        if choice > 0:
            noise = jnp.rot90(m=noise, k=choice)
        
        perlin_thr = jnp.where(noise > threshold, jnp.ones_like(noise), jnp.zeros_like(noise))
        return perlin_thr
    

    def rand_perlin_2d(self, shape: tuple[int, int], res: tuple[int, int], fade: Callable[[ArrayLike], ArrayLike] = lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3) -> ArrayLike:
        """
        https://www.wolframalpha.com/input?i=plot+6+*+t+**+5+-+15+*+t+**+4+%2B+10+*+t+**+3+from+-10+to+10
        """
        
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = jnp.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

        key = self.key_perlin2d
        angles = 2. * math.pi * jrand.uniform(key=key, shape=(res[0] + 1, res[1] + 1))
        gradients = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)

        def tile_grads(slice1: tuple[int, int|None], slice2: tuple[int, int|None]) -> jnp.ndarray:
            return jnp.repeat(a=jnp.repeat(a=gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], repeats=d[0], axis=0), repeats=d[1], axis=1)

        def dot(grad: jnp.ndarray, shift: tuple[int, int]):
            return (jnp.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]), axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)
        
        def lin_interp(x: ArrayLike, y: ArrayLike, w: ArrayLike) -> ArrayLike:
            return (y - x) * w + x

        n00 = dot(tile_grads((0, -1), (0, -1)), (0, 0))
        n10 = dot(tile_grads((1, None), (0, -1)), (-1, 0))
        n01 = dot(tile_grads((0, -1), (1, None)), (0, -1))
        n11 = dot(tile_grads((1, None), (1, None)), (-1, -1))
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * lin_interp(lin_interp(n00, n10, t[..., 0]), lin_interp(n01, n11, t[..., 0]), t[..., 1])
    

    def perlin_mask(self, img_shape: tuple[int, int, int], feat_size: int, min: int, max: int, mask_fg: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        mask_s = jnp.zeros((feat_size, feat_size))
        while jnp.max(mask_s) == 0:
            # SH: variables `perlin_thr_1` and `perlin_thr_2` are the $m_1$ and $m_2$ from the paper (Sec. 3.3).
            perlin_thr_1 = self.generate_threshold_mask(img_shape=img_shape, min=min, max=max)
            perlin_thr_2 = self.generate_threshold_mask(img_shape=img_shape, min=min, max=max)
            
            
            # This is the variable $\alpha$ from the paper.
            alpha = jrand.uniform(key=self.key_alpha, shape=(1,)).item()
            if alpha > 2. / 3.:
                perlin_thr = perlin_thr_1 + perlin_thr_2
                perlin_thr = jnp.where(perlin_thr > 0, jnp.ones_like(perlin_thr), np.zeros_like(perlin_thr))
            elif alpha > 1. / 3.:
                perlin_thr = perlin_thr_1 * perlin_thr_2
            else:
                perlin_thr = perlin_thr_1

            # Create a mask that is a complementry of the foreground- and noise-mask.
            perlin_thr_fg = perlin_thr * mask_fg
            down_ratio_y = int(img_shape[1] / feat_size)
            down_ratio_x = int(img_shape[2] / feat_size)
            mask_l = perlin_thr_fg.copy()
            kernel = (down_ratio_y, down_ratio_x)
            mask_s = nn.max_pool(jnp.expand_dims(perlin_thr_fg, 2), window_shape=kernel, strides=kernel).astype(jnp.float32).squeeze()
        
        return mask_s, mask_l



if __name__ == '__main__':
    """
    Let's test creating a LAS-style anomaly using the images nominal, fg, and texture{.png}.
    """
    seed = 1

    nominal = jnp.asarray(Image.open(THIS_DIR.joinpath('./nominal.png')).convert('RGB')).T
    fg = jnp.asarray(Image.open(THIS_DIR.joinpath('./fg.png'))).T
    texture = jnp.asarray(Image.open(THIS_DIR.joinpath('./texture.png')).convert('RGB')).T

    pgen = Perlin_generator(seed=seed)
    mask_s, mask_l = pgen.perlin_mask(img_shape=nominal.shape, feat_size=576//8, min=0, max=6, mask_fg=fg)
    
    # 0.5 is the default mean-value in mvtec, 0.1 is the std-dev
    beta = 0.5 + 0.1 * jrand.normal(key=jrand.PRNGKey(seed), shape=(1,)).clip(0.2, 0.8)
    augmented = nominal * (1. - mask_l) + (1. - beta) * texture * mask_l + beta * nominal * mask_l

    res = Image.fromarray(np.asarray(jnp.uint8(augmented)*255).T)
    print(5)
    
