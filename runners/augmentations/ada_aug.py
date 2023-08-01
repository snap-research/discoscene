# python3.7
"""Defines the augmentation pipeline from StyleGAN2-ADA.

Basically, this pipeline executes a series of augmentations one by one, based
on an adjustable probability. More details can be found at paper

https://arxiv.org/pdf/2006.06676.pdf

This code is borrowed from the official implementation

https://github.com/NVlabs/stylegan2-ada-pytorch
"""

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.stylegan2_official_ops import misc
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import conv2d_gradfix
from third_party.stylegan2_official_ops import grid_sample_gradfix

__all__ = ['AdaAug']

# Coefficients of various wavelet decomposition low-pass filters.
WAVELETS = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1': [0.7071067811865476, 0.7071067811865476],
    'db2': [-0.12940952255092145, 0.22414386804185735,
            0.836516303737469, 0.48296291314469025],
    'db3': [0.035226291882100656, -0.08544127388224149,
            -0.13501102001039084, 0.4598775021193313,
            0.8068915093133388, 0.3326705529509569],
    'db4': [-0.010597401784997278, 0.032883011666982945,
            0.030841381835986965, -0.18703481171888114,
            -0.02798376941698385, 0.6308807679295904,
            0.7148465705525415, 0.23037781330885523],
    'db5': [0.003335725285001549, -0.012580751999015526,
            -0.006241490213011705, 0.07757149384006515,
            -0.03224486958502952, -0.24229488706619015,
            0.13842814590110342, 0.7243085284385744,
            0.6038292697974729, 0.160102397974125],
    'db6': [-0.00107730108499558, 0.004777257511010651,
            0.0005538422009938016, -0.031582039318031156,
            0.02752286553001629, 0.09750160558707936,
            -0.12976686756709563, -0.22626469396516913,
            0.3152503517092432, 0.7511339080215775,
            0.4946238903983854, 0.11154074335008017],
    'db7': [0.0003537138000010399, -0.0018016407039998328,
            0.00042957797300470274, 0.012550998556013784,
            -0.01657454163101562, -0.03802993693503463,
            0.0806126091510659, 0.07130921926705004,
            -0.22403618499416572, -0.14390600392910627,
            0.4697822874053586, 0.7291320908465551,
            0.39653931948230575, 0.07785205408506236],
    'db8': [-0.00011747678400228192, 0.0006754494059985568,
            -0.0003917403729959771, -0.00487035299301066,
            0.008746094047015655, 0.013981027917015516,
            -0.04408825393106472, -0.01736930100202211,
            0.128747426620186, 0.00047248457399797254,
            -0.2840155429624281, -0.015829105256023893,
            0.5853546836548691, 0.6756307362980128,
            0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735,
             0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149,
             -0.13501102001039084, 0.4598775021193313,
             0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851,
             0.49761866763201545, 0.8037387518059161,
             0.29785779560527736, -0.09921954357684722,
             -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643,
             -0.039134249302383094, 0.1993975339773936,
             0.7234076904024206, 0.6339789634582119,
             0.01660210576452232, -0.17532808990845047,
             -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702,
             -0.11799011114819057, -0.048311742585633,
             0.4910559419267466, 0.787641141030194,
             0.3379294217276218, -0.07263752278646252,
             -0.021060292512300564, 0.04472490177066578,
             0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163,
             -0.01263630340325193, 0.03051551316596357,
             0.0678926935013727, -0.049552834937127255,
             0.017441255086855827, 0.5361019170917628,
             0.767764317003164, 0.2886296317515146,
             -0.14004724044296152, -0.10780823770381774,
             0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481,
             0.03169508781149298, 0.007607487324917605,
             -0.1432942383508097, -0.061273359067658524,
             0.4813596512583722, 0.7771857517005235,
             0.3644418948353314, -0.05194583810770904,
             -0.027219029917056003, 0.049137179673607506,
             0.003808752013890615, -0.01495225833704823,
             -0.0003029205147213668, 0.0018899503327594609]
}


def matrix(*rows, device=None):
    """Constructs a matrix with given rows."""
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x
             if isinstance(x, torch.Tensor)
             else misc.constant(x, shape=ref[0].shape, device=ref[0].device)
             for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def translate2d(tx, ty, device=None):
    """Gets a matrix for 2D translation."""
    return matrix([1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1],
                  device=device)


def translate3d(tx, ty, tz, device=None):
    """Gets a matrix for 3D translation."""
    return matrix([1, 0, 0, tx],
                  [0, 1, 0, ty],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1],
                  device=device)


def scale2d(sx, sy, device=None):
    """Gets a matrix for 2D scaling."""
    return matrix([sx, 0,  0],
                  [0,  sy, 0],
                  [0,  0,  1],
                  device=device)


def scale3d(sx, sy, sz, device=None):
    """Gets a matrix for 3D scaling."""
    return matrix([sx, 0,  0,  0],
                  [0,  sy, 0,  0],
                  [0,  0,  sz, 0],
                  [0,  0,  0,  1],
                  device=device)


def rotate2d(theta, device=None):
    """Gets a matrix for 2D rotation."""
    return matrix([torch.cos(theta), torch.sin(-theta), 0],
                  [torch.sin(theta), torch.cos(theta),  0],
                  [0,                0,                 1],
                  device=device)


def rotate3d(v, theta, device=None):
    """Gets a matrix for 3D rotation."""
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]
    s = torch.sin(theta)
    c = torch.cos(theta)
    cc = 1 - c
    return matrix(
        [vx*vx*cc + c,    vx*vy*cc - vz*s, vx*vz*cc + vy*s, 0],
        [vy*vx*cc + vz*s, vy*vy*cc + c,    vy*vz*cc - vx*s, 0],
        [vz*vx*cc - vy*s, vz*vy*cc + vx*s, vz*vz*cc + c,    0],
        [0,               0,               0,               1],
        device=device)


def translate2d_inv(tx, ty, device=None):
    """Gets a matrix for inverse 2D translation."""
    return translate2d(-tx, -ty, device=device)


def scale2d_inv(sx, sy, device=None):
    """Gets a matrix for inverse 2D scaling."""
    return scale2d(1 / sx, 1 / sy, device=device)


def rotate2d_inv(theta, device=None):
    """Gets a matrix for inverse 2D rotation."""
    return rotate2d(-theta, device=device)

# pylint: disable=missing-function-docstring

class AdaAug(nn.Module):
    """Defines an adaptive augmentation pipeline.

    This pipeline is introduced in paper "Training Generative Adversarial
    Networks with Limited Data" (https://arxiv.org/pdf/2006.06676.pdf).

    Basically, this class supports versatile image augmentations which are
    executed one by one. Each augmentation will be executed on a probability.
    User can initialize this class with a probability multiplier for each
    augmentation, and also adjust the variable `self.p` to control the
    probability for all augmentations.
    """
    def __init__(self,
                 xflip=0,
                 rotate90=0,
                 xint=0,
                 xint_max=0.125,
                 scale=0,
                 rotate=0,
                 aniso=0,
                 xfrac=0,
                 scale_std=0.2,
                 rotate_max=1,
                 aniso_std=0.2,
                 xfrac_std=0.125,
                 brightness=0,
                 contrast=0,
                 lumaflip=0,
                 hue=0,
                 saturation=0,
                 brightness_std=0.2,
                 contrast_std=0.5,
                 hue_max=1,
                 saturation_std=1,
                 imgfilter=0,
                 imgfilter_bands=(1,1,1,1),
                 imgfilter_std=1,
                 noise=0,
                 cutout=0,
                 noise_std=0.1,
                 cutout_size=0.5):
        """Initializes with probability multipliers for each augmentation.

        For all probability multipliers, `0` means disabling a particular
        augmentation while `1` means enabling.

        Augmentation settings include:

        - Pixel blitting:

        (1) xflip: Probability multiplier for x-flip. (default: 0)
        (2) rotate90: Probability multiplier for 90 degree rotations.
            (default: 0)
        (3) xint: Probability multiplier for integer translation.
            (default: 0)
        (4) xint_max: Range of integer translation, relative to image
            dimensions. (default: 0.125)

        - General geometric transformation:

        (1) scale: Probability multiplier for isotropic scaling. (default: 0)
        (2) rotate: Probability multiplier for arbitrary rotation. (default: 0)
        (3) aniso: Probability multiplier for anisotropic scaling. (default: 0)
        (4) xfrac: Probability multiplier for fractional translation.
            (default: 0)
        (5) scale_std: Log2 standard deviation of isotropic scaling.
            (default: 0.2)
        (6) rotate_max: Range of arbitrary rotation, `1` means full circle.
            (default: 1)
        (7) aniso_std: Log2 standard deviation of anisotropic scaling.
            (default: 0.2)
        (8) xfrac_std: Standard deviation of fractional translation, relative to
            image dimensions. (default: 0.125)

        - Color transformation:

        (1) brightness: Probability multiplier for brightness. (default: 0)
        (2) contrast: Probability multiplier for contrast. (default: 0)
        (3) lumaflip: Probability multiplier for luma flip. (default: 0)
        (4) hue: Probability multiplier for hue rotation. (default: 0)
        (5) saturation: Probability multiplier for saturation. (default: 0)
        (6) brightness_std: Standard deviation of brightness. (default: 0.2)
        (7) contrast_std: Log2 standard deviation of contrast. (default: 0.5)
        (8) hue_max: Range of hue rotation, `1` means full circle. (default: 1)
        (9) saturation_std: Log2 standard deviation of saturation. (default: 1)

        - Image-space filtering:

        (1) imgfilter: Probability multiplier for image-space filtering.
            (default: 0)
        (2) imgfilter_bands: Probability multipliers for individual frequency
            bands. (default: (1, 1, 1, 1))
        (3) imgfilter_std: Log2 standard deviation of image-space filter
            amplification. (default: 1)

        - Image-space corruption:

        (1) noise: Probability multiplier for additive RGB noise. (default: 0)
        (2) cutout: Probability multiplier for cutout. (default: 0)
        (3) noise_std: Standard deviation of additive RGB noise. (default: 0.1)
        (4) cutout_size: Size of the cutout rectangle, relative to image
            dimensions. (default: 0.5)
        """
        super().__init__()

        # Overall probability multiplier for all augmentations.
        self.register_buffer('p', torch.ones([]))

        # Pixel blitting.
        self.xflip = float(xflip)
        self.rotate90 = float(rotate90)
        self.xint = float(xint)
        self.xint_max = float(xint_max)

        # General geometric transformations.
        self.scale = float(scale)
        self.rotate = float(rotate)
        self.aniso = float(aniso)
        self.xfrac = float(xfrac)
        self.scale_std = float(scale_std)
        self.rotate_max = float(rotate_max)
        self.aniso_std = float(aniso_std)
        self.xfrac_std = float(xfrac_std)

        # Color transformations.
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.lumaflip = float(lumaflip)
        self.hue = float(hue)
        self.saturation = float(saturation)
        self.brightness_std = float(brightness_std)
        self.contrast_std = float(contrast_std)
        self.hue_max = float(hue_max)
        self.saturation_std = float(saturation_std)

        # Image-space filtering.
        self.imgfilter = float(imgfilter)
        self.imgfilter_bands = list(imgfilter_bands)
        self.imgfilter_std = float(imgfilter_std)

        # Image-space corruptions.
        self.noise = float(noise)
        self.cutout = float(cutout)
        self.noise_std = float(noise_std)
        self.cutout_size = float(cutout_size)

        # Setup orthogonal lowpass filter for geometric augmentations.
        self.register_buffer(
            'Hz_geom', upfirdn2d.setup_filter(WAVELETS['sym6']))

        # Construct filter bank for image-space filtering.
        Hz_lo = np.asarray(WAVELETS['sym2'])            # H(z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
        Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)])
            Hz_fbank = Hz_fbank.reshape(Hz_fbank.shape[0], -1)[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            start = (Hz_fbank.shape[1] - Hz_hi2.size) // 2
            end = (Hz_fbank.shape[1] + Hz_hi2.size) // 2
            Hz_fbank[i, start:end] += Hz_hi2
        self.register_buffer(
            'Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))

    def forward(self, images, debug_percentile=None, impl='cuda'):
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device
        if debug_percentile is not None:
            debug_percentile = torch.as_tensor(
                debug_percentile, dtype=torch.float32, device=device)

        ##################
        # Pixel Blitting #
        ##################

        # Initialize inverse homogeneous 2D transform:
        #   G_inv @ pixel_out ==> pixel_in
        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        # Apply x-flip with probability (self.xflip * self.p).
        if self.xflip > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 2)
            prob = torch.rand([batch_size], device=device)
            i = torch.where(
                prob < self.xflip * self.p,
                i,
                torch.zeros_like(i)
            )
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 2))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)

        # Apply 90 degree rotations with probability (self.rotate90 * self.p).
        if self.rotate90 > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 4)
            prob = torch.rand([batch_size], device=device)
            i = torch.where(
                prob < self.rotate90 * self.p,
                i,
                torch.zeros_like(i)
            )
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 4))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)

        # Apply integer translation with probability (self.xint * self.p).
        if self.xint > 0:
            t = torch.rand([batch_size, 2], device=device) * 2 - 1
            t = t * self.xint_max
            prob = torch.rand([batch_size, 1], device=device)
            t = torch.where(
                prob < self.xint * self.p,
                t,
                torch.zeros_like(t)
            )
            if debug_percentile is not None:
                t = torch.full_like(
                    t,
                    (debug_percentile * 2 - 1) * self.xint_max
                )
            _translate_matrix = translate2d_inv(
                torch.round(t[:, 0] * width), torch.round(t[:, 1] * height))
            G_inv = G_inv @ _translate_matrix

        ##########################################################
        # Select Parameters for General Geometric Transformation #
        ##########################################################

        # Apply isotropic scaling with probability (self.scale * self.p).
        if self.scale > 0:
            s = torch.exp2(
                torch.randn([batch_size], device=device) * self.scale_std)
            prob = torch.rand([batch_size], device=device)
            s = torch.where(
                prob < self.scale * self.p,
                s,
                torch.ones_like(s)
            )
            if debug_percentile is not None:
                s = torch.full_like(
                    s,
                    torch.exp2(
                        torch.erfinv(debug_percentile * 2 - 1) * self.scale_std
                    )
                )
            G_inv = G_inv @ scale2d_inv(s, s)

        # Apply pre-rotation with probability (p_rot).
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1))
        if self.rotate > 0:
            theta = torch.rand([batch_size], device=device) * 2 - 1
            theta = theta * np.pi * self.rotate_max
            prob = torch.rand([batch_size], device=device)
            theta = torch.where(
                prob < p_rot,
                theta,
                torch.zeros_like(theta)
            )
            if debug_percentile is not None:
                theta = torch.full_like(
                    theta,
                    (debug_percentile * 2 - 1) * np.pi * self.rotate_max
                )
            G_inv = G_inv @ rotate2d_inv(-theta)  # Before anisotropic scaling.

        # Apply anisotropic scaling with probability (self.aniso * self.p).
        if self.aniso > 0:
            s = torch.exp2(
                torch.randn([batch_size], device=device) * self.aniso_std)
            prob = torch.rand([batch_size], device=device)
            s = torch.where(
                prob < self.aniso * self.p,
                s,
                torch.ones_like(s)
            )
            if debug_percentile is not None:
                s = torch.full_like(
                    s,
                    torch.exp2(
                        torch.erfinv(debug_percentile * 2 - 1) * self.aniso_std
                    )
                )
            G_inv = G_inv @ scale2d_inv(s, 1 / s)

        # Apply post-rotation with probability (p_rot).
        if self.rotate > 0:
            theta = torch.rand([batch_size], device=device) * 2 - 1
            theta = theta * np.pi * self.rotate_max
            prob = torch.rand([batch_size], device=device)
            theta = torch.where(
                prob < p_rot,
                theta,
                torch.zeros_like(theta)
            )
            if debug_percentile is not None:
                theta = torch.zeros_like(theta)
            G_inv = G_inv @ rotate2d_inv(-theta)  # After anisotropic scaling.

        # Apply fractional translation with probability (self.xfrac * self.p).
        if self.xfrac > 0:
            t = torch.randn([batch_size, 2], device=device) * self.xfrac_std
            prob = torch.rand([batch_size, 1], device=device)
            t = torch.where(
                prob < self.xfrac * self.p,
                t,
                torch.zeros_like(t)
            )
            if debug_percentile is not None:
                t = torch.full_like(
                    t,
                    torch.erfinv(debug_percentile * 2 - 1) * self.xfrac_std
                )
            G_inv = G_inv @ translate2d_inv(t[:, 0] * width, t[:, 1] * height)

        ####################################
        # Execute Geometric Transformation #
        ####################################

        # Execute if the transform is not identity.
        if G_inv is not I_3:
            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1],
                        [cx, -cy, 1],
                        [cx, cy, 1],
                        [-cx, cy, 1],
                        device=device)  # [idx, xyz]
            cp = G_inv @ cp.t()  # [N, xyz, idx]
            Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1)  # [xy, N * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values
            margin = margin + misc.constant(
                [Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(misc.constant([0, 0] * 2, device=device))
            margin = margin.min(misc.constant(
                [width - 1, height - 1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = F.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            images = upfirdn2d.upsample2d(
                x=images, f=self.Hz_geom, up=2, impl=impl)
            G_inv = (scale2d(2, 2, device=device) @
                     G_inv @
                     scale2d_inv(2, 2, device=device))
            G_inv = (translate2d(-0.5, -0.5, device=device) @
                     G_inv @
                     translate2d_inv(-0.5, -0.5, device=device))

            # Execute transformation.
            shape = [batch_size,
                     num_channels,
                     (height + Hz_pad * 2) * 2,
                     (width + Hz_pad * 2) * 2]
            _scale_matrix = scale2d(2 / images.shape[3],
                                    2 / images.shape[2],
                                    device=device)
            _scale_inv_matrix = scale2d_inv(2 / shape[3],
                                            2 / shape[2],
                                            device=device)
            G_inv = _scale_matrix @ G_inv @ _scale_inv_matrix
            grid = F.affine_grid(theta=G_inv[:, :2, :],
                                 size=shape,
                                 align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid, impl=impl)

            # Downsample and crop.
            images = upfirdn2d.downsample2d(x=images,
                                            f=self.Hz_geom,
                                            down=2,
                                            padding=-Hz_pad * 2,
                                            flip_filter=True,
                                            impl=impl)

        ##############################################
        # Select Parameters for Color Transformation #
        ##############################################

        # Initialize homogeneous 3D transformation matrix:
        #   C @ color_in ==> color_out
        I_4 = torch.eye(4, device=device)
        C = I_4

        # Apply brightness with probability (self.brightness * self.p).
        if self.brightness > 0:
            b = torch.randn([batch_size], device=device) * self.brightness_std
            prob = torch.rand([batch_size], device=device)
            b = torch.where(
                prob < self.brightness * self.p,
                b,
                torch.zeros_like(b)
            )
            if debug_percentile is not None:
                b = torch.full_like(
                    b,
                    torch.erfinv(
                        debug_percentile * 2 - 1
                    ) * self.brightness_std
                )
            C = translate3d(b, b, b) @ C

        # Apply contrast with probability (self.contrast * self.p).
        if self.contrast > 0:
            c = torch.exp2(
                torch.randn([batch_size], device=device) * self.contrast_std)
            prob = torch.rand([batch_size], device=device)
            c = torch.where(
                prob < self.contrast * self.p,
                c,
                torch.ones_like(c)
            )
            if debug_percentile is not None:
                c = torch.full_like(
                    c,
                    torch.exp2(
                        torch.erfinv(
                            debug_percentile * 2 - 1
                        ) * self.contrast_std
                    )
                )
            C = scale3d(c, c, c) @ C

        # Apply luma flip with probability (self.lumaflip * self.p).
        v = misc.constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device)
        if self.lumaflip > 0:
            i = torch.floor(torch.rand([batch_size, 1, 1], device=device) * 2)
            prob = torch.rand([batch_size, 1, 1], device=device)
            i = torch.where(
                prob < self.lumaflip * self.p,
                i,
                torch.zeros_like(i)
            )
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 2))
            C = (I_4 - 2 * v.ger(v) * i) @ C  # Householder reflection.

        # Apply hue rotation with probability (self.hue * self.p).
        if self.hue > 0 and num_channels > 1:
            theta = (torch.rand([batch_size], device=device) * 2 - 1)
            theta = theta * np.pi * self.hue_max
            prob = torch.rand([batch_size], device=device)
            theta = torch.where(
                prob < self.hue * self.p,
                theta,
                torch.zeros_like(theta)
            )
            if debug_percentile is not None:
                theta = torch.full_like(
                    theta,
                    (debug_percentile * 2 - 1) * np.pi * self.hue_max
                )
            C = rotate3d(v, theta) @ C  # Rotate around v.

        # Apply saturation with probability (self.saturation * self.p).
        if self.saturation > 0 and num_channels > 1:
            s = torch.randn([batch_size, 1, 1], device=device)
            s = torch.exp2(s * self.saturation_std)
            prob = torch.rand([batch_size, 1, 1], device=device)
            s = torch.where(
                prob < self.saturation * self.p,
                s,
                torch.ones_like(s)
            )
            if debug_percentile is not None:
                s = torch.full_like(
                    s,
                    torch.exp2(
                        torch.erfinv(
                            debug_percentile * 2 - 1
                        ) * self.saturation_std
                    )
                )
            C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

        ################################
        # Execute Color Transformation #
        ################################

        # Execute if the transform is not identity.
        if C is not I_4:
            images = images.reshape([batch_size, num_channels, height * width])
            if num_channels % 3 != 0:
                masks = images[:, -1:]
                images = images[:, :-1]
            _num_channels = images.shape[1]
            if _num_channels == 3:
                images = C[:, :3, :3] @ images + C[:, :3, 3:]
                if num_channels>3:
                    images = torch.cat([images, masks], dim=1)
            elif _num_channels == 6:
                images1 = images[:, :3]
                images2 = images[:, 3:6]
                images1 = C[:, :3, :3] @ images1 + C[:, :3, 3:]
                images2 = C[:, :3, :3] @ images2 + C[:, :3, 3:]
                images = torch.cat([images1, images2], dim=1)
                if num_channels > 6:
                    images = torch.cat([images, masks], dim=1)
                else:
                    pass
            elif _num_channels == 1:
                images = C[:, :3, :3] @ images + C[:, :3, 3:]
                C = C[:, :3, :].mean(dim=1, keepdims=True)
                images = (images * C[:, :, :3].sum(dim=2, keepdims=True) +
                          C[:, :, 3:])
            else:
                raise ValueError(
                    'Image must be RGB (3 channels) or L (1 channel)')
            images = images.reshape([batch_size, num_channels, height, width])

        #########################
        # Image-space Filtering #
        #########################

        if self.imgfilter > 0:
            num_bands = self.Hz_fbank.shape[0]
            assert len(self.imgfilter_bands) == num_bands

            # Expected power spectrum (1/f).
            expected_power = misc.constant(
                np.array([10, 1, 1, 1]) / 13, device=device)

            # Apply amplification for each band with probability
            # (self.imgfilter * self.p * band_strength).
            # Global gain vector (identity).
            g = torch.ones([batch_size, num_bands], device=device)
            for i, band_strength in enumerate(self.imgfilter_bands):
                t_i = torch.randn([batch_size], device=device)
                t_i = torch.exp2(t_i * self.imgfilter_std)
                prob = torch.rand([batch_size], device=device)
                t_i = torch.where(
                    prob < self.imgfilter * self.p * band_strength,
                    t_i,
                    torch.ones_like(t_i)
                )
                if debug_percentile is not None:
                    if band_strength > 0:
                        t_i = torch.full_like(
                            t_i,
                            torch.exp2(
                                torch.erfinv(
                                    debug_percentile * 2 - 1
                                ) * self.imgfilter_std
                            )
                        )
                    else:
                        t_i = torch.ones_like(t_i)
                # Temporary gain vector.
                t = torch.ones([batch_size, num_bands], device=device)
                # Replace i-th element.
                t[:, i] = t_i
                # Normalize power.
                _temp_norm = expected_power * t.square()
                t = t / _temp_norm.sum(dim=-1, keepdims=True).sqrt()
                # Accumulate into global gain.
                g = g * t

            # Construct combined amplification filter.
            Hz_prime = g @ self.Hz_fbank  # [batch, tap]
            Hz_prime = Hz_prime.unsqueeze(1).repeat([1, num_channels, 1])
            Hz_prime = Hz_prime.reshape([batch_size * num_channels, 1, -1])

            # Apply filter.
            p = self.Hz_fbank.shape[1] // 2
            images = images.reshape(
                [1, batch_size * num_channels, height, width])
            images = F.pad(input=images, pad=[p, p, p, p], mode='reflect')
            images = conv2d_gradfix.conv2d(input=images,
                                           weight=Hz_prime.unsqueeze(2),
                                           groups=batch_size * num_channels,
                                           impl=impl)
            images = conv2d_gradfix.conv2d(input=images,
                                           weight=Hz_prime.unsqueeze(3),
                                           groups=batch_size * num_channels,
                                           impl=impl)
            images = images.reshape([batch_size, num_channels, height, width])

        ##########################
        # Image-space Corruption #
        ##########################

        # Apply additive RGB noise with probability (self.noise * self.p).
        if self.noise > 0:
            sigma = torch.randn([batch_size, 1, 1, 1], device=device)
            sigma = sigma.abs() * self.noise_std
            prob = torch.rand([batch_size, 1, 1, 1], device=device)
            sigma = torch.where(
                prob < self.noise * self.p,
                sigma,
                torch.zeros_like(sigma)
            )
            if debug_percentile is not None:
                sigma = torch.full_like(
                    sigma,
                    torch.erfinv(debug_percentile) * self.noise_std
                )
            _noise = torch.randn(
                [batch_size, num_channels, height, width], device=device)
            images = images + _noise * sigma

        # Apply cutout with probability (self.cutout * self.p).
        if self.cutout > 0:
            size = torch.full(
                [batch_size, 2, 1, 1, 1], self.cutout_size, device=device)
            prob = torch.rand([batch_size, 1, 1, 1, 1], device=device)
            size = torch.where(
                prob < self.cutout * self.p,
                size,
                torch.zeros_like(size)
            )
            center = torch.rand([batch_size, 2, 1, 1, 1], device=device)
            if debug_percentile is not None:
                size = torch.full_like(size, self.cutout_size)
                center = torch.full_like(center, debug_percentile)
            coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
            mask_x = (
                ((coord_x + 0.5) / width - center[:, 0]).abs() >=
                size[:, 0] / 2
            )
            mask_y = (
                ((coord_y + 0.5) / height - center[:, 1]).abs() >=
                size[:, 1] / 2
            )
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            images = images * mask

        return images

# pylint: enable=missing-function-docstring
