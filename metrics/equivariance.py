# python3.7
"""Contains the class to evaluate GANs with Equivariance.

EQ-T and EQ-R are introduced in paper https://arxiv.org/pdf/2106.12423.pdf

Most functions in this file are borrowed from the original implementation:

https://github.com/NVlabs/stylegan3/blob/main/metrics/equivariance.py

Basically, the generator should support a customizable input transform matrix to
control the output synthesis. Equivariance evaluates how much the synthesis
changes if we (1) apply a transform to the input, and (2) apply an inverse
transform to the output. Such a metric measures how well the generator handles
transformations.
"""

import os.path
import time
import numpy as np

import torch
import torch.fft
import torch.nn.functional as F

from third_party.stylegan3_official_ops import upfirdn2d
from .base_gan_metric import BaseGANMetric

__all__ = [
    'EquivarianceMetric', 'EQTMetric', 'EQT50K', 'EQTFracMetric', 'EQTFrac50K',
    'EQRMetric', 'EQR50K'
]


def sinc(x):
    """Applies the `sinc` function."""
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)


def lanczos_window(x, a):
    """Applies Lanczos window.

    Args:
        x: The input signal.
        a: Kernel size.
    """
    x = x.abs() / a
    return torch.where(x < 1, sinc(x), torch.zeros_like(x))


def rotation_matrix(angle):
    """Gets a transformation matrix for 2D rotation."""
    angle = torch.as_tensor(angle).to(torch.float32)
    mat = torch.eye(3, device=angle.device)
    mat[0, 0] = angle.cos()
    mat[0, 1] = angle.sin()
    mat[1, 0] = -angle.sin()
    mat[1, 1] = angle.cos()
    return mat


def apply_integer_translation(x, tx, ty):
    """Applies integer translation to a batch of 2D images.

    The images will only be translated with integer number of pixels.

    Args:
        x: Input images, with shape [N, C, H, W].
        tx: Translation along x axis.
        ty: Translation along y axis.
    """
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.round().to(torch.int64)
    iy = ty.round().to(torch.int64)

    z = torch.zeros_like(x)
    m = torch.zeros_like(x)
    if abs(ix) < W and abs(iy) < H:
        y = x[:, :, max(-iy, 0):H + min(-iy, 0), max(-ix, 0):W + min(-ix, 0)]
        z[:, :, max(iy, 0):H + min(iy, 0), max(ix, 0):W + min(ix, 0)] = y
        m[:, :, max(iy, 0):H + min(iy, 0), max(ix, 0):W + min(ix, 0)] = 1
    return z, m


def apply_fractional_translation(x, tx, ty, a=3, impl='cuda'):
    """Applies integer translation to a batch of 2D images.

    Different from function `apply_integer_translation()`, the images can be
    translated with subpixels.

    Args:
        x: Input images, with shape [N, C, H, W].
        tx: Translation along x axis.
        ty: Translation along y axis.
        a: A factor to control filter length. (default: 3)
        impl: Implementation mode of filtering. (default: `cuda`)
    """
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.floor().to(torch.int64)
    iy = ty.floor().to(torch.int64)
    fx = tx - ix
    fy = ty - iy
    b = a - 1

    z = torch.zeros_like(x)
    zx0 = max(ix - b, 0)
    zy0 = max(iy - b, 0)
    zx1 = min(ix + a, 0) + W
    zy1 = min(iy + a, 0) + H
    if zx0 < zx1 and zy0 < zy1:
        taps = torch.arange(a * 2, device=x.device) - b
        filter_x = (sinc(taps - fx) * sinc((taps - fx) / a)).unsqueeze(0)
        filter_y = (sinc(taps - fy) * sinc((taps - fy) / a)).unsqueeze(1)
        y = x
        y = upfirdn2d.filter2d(x=y,
                               f=filter_x / filter_x.sum(),
                               padding=[b, a, 0, 0],
                               impl=impl)
        y = upfirdn2d.filter2d(x=y,
                               f=filter_y / filter_y.sum(),
                               padding=[0, 0, b, a],
                               impl=impl)
        y = y[:,
              :,
              max(b - iy, 0):H + b + a + min(-iy - a, 0),
              max(b - ix, 0):W + b + a + min(-ix - a, 0)]
        z[:, :, zy0:zy1, zx0:zx1] = y

    m = torch.zeros_like(x)
    mx0 = max(ix + a, 0)
    my0 = max(iy + a, 0)
    mx1 = min(ix - b, 0) + W
    my1 = min(iy - b, 0) + H
    if mx0 < mx1 and my0 < my1:
        m[:, :, my0:my1, mx0:mx1] = 1
    return z, m


def construct_affine_bandlimit_filter(mat,
                                      a=3,
                                      amax=16,
                                      aflt=64,
                                      up=4,
                                      cutoff_in=1,
                                      cutoff_out=1):
    """Constructs an oriented low-pass filter.

    This filter applies the appropriate bandlimit with respect to the input and
    output of the given 2D affine transformation.

    Args:
        mat: The transformation matrix.
        a: Kernel size for Lanczos window. (default: 3)
        amax: Maximum kernel size. (default: 16)
        aflt: Length of filter. (default: 64)
        up: Upsampling factor for filtering. (default: 4)
        cutoff_in: Cutoff frequency of the input. (default: 1)
        cutoff_out: Cutoff frequency of the output. (default: 1)
    """
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = torch.arange(aflt * up * 2 - 1, device=mat.device)
    taps = ((taps + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = sinc(xi * cutoff_in) * sinc(yi * cutoff_in)
    fo = sinc(xo * cutoff_out) * sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = lanczos_window(xi, a) * lanczos_window(yi, a)
    wo = lanczos_window(xo, a) * lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0, 1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1])
    f = f.reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0, 2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f


def apply_affine_transformation(x, mat, up=4, impl='cuda', **filter_kwargs):
    """Applies affine transformation to a batch of 2D images.

    Args:
        x: Input images, with shape [N, C, H, W].
        mat: The transformation matrix.
        up: Upsampling factor used to construct the bandlimit filter. See
            function `construct_affine_bandlimit_filter()`. (default: 4)
        impl: Implementation mode of filtering. (default: `cuda`)
        **filter_kwargs: Additional arguments for constructing the bandlimit
            filter. See function `construct_affine_bandlimit_filter()`.
    """
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p, impl=impl)
    z = torch.nn.functional.grid_sample(input=y,
                                        grid=g,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(input=m,
                                        grid=g,
                                        mode='nearest',
                                        padding_mode='zeros',
                                        align_corners=False)
    return z, m


def apply_fractional_rotation(x, angle, a=3, impl='cuda', **filter_kwargs):
    """Applies fractional rotation to a batch of 2D images.

    Args:
        x: Input images, with shape [N, C, H, W].
        angle: The rotation angle.
        a: Kernel size for Lanczos window. See function
            `construct_affine_bandlimit_filter()`. (default: 3)
        impl: Implementation mode of filtering. (default: `cuda`)
    """
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(angle)
    return apply_affine_transformation(
        x, mat, a=a, amax=a*2, impl=impl, **filter_kwargs)


def apply_fractional_pseudo_rotation(x,
                                     angle,
                                     a=3,
                                     impl='cuda',
                                     **filter_kwargs):
    """Applies fractional pseudo rotation to a batch of 2D images.

    This function modifies the frequency content of the input images as if they
    had undergo fractional rotation, but WITHOUT actually rotating them.

    Args:
        x: Input images, with shape [N, C, H, W].
        angle: The rotation angle.
        a: Kernel size for Lanczos window. See function
            `construct_affine_bandlimit_filter()`. (default: 3)
        impl: Implementation mode of filtering. (default: `cuda`)
    """
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(-angle)
    f = construct_affine_bandlimit_filter(
        mat, a=a, amax=a*2, up=1, **filter_kwargs)
    y = upfirdn2d.filter2d(x=x, f=f, impl=impl)
    m = torch.zeros_like(y)
    c = f.shape[0] // 2
    m[:, :, c:-c, c:-c] = 1
    return y, m


class EquivarianceMetric(BaseGANMetric):
    """Defines the base class for evaluating equivariance."""

    def __init__(self,
                 name='Equivariance',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 input_transformation_name='synthesis.early_layer.transform',
                 translate_max=0.125,
                 rotate_max=1,
                 test_eqt=False,
                 test_eqt_frac=False,
                 test_eqr=False):
        """Initializes the class with some hyper-parameters.

        Args:
            input_transformation_name: Name of the customizable input
                transformation. (default `synthesis.early_layer.transform` for
                `models/stylegan3_generator.py`)
            translate_max: Maximum relative translation. (default: 0.125)
            rotate_max: Maximum rotation. (default: 1)
            test_eqt: Whether to evaluate EQ-T metric. (default: False)
            test_eqt_frac: Whether to evaluate EQ-T_frac metric.
                (default: False)
            test_eqr: Whether to evaluate EQ-R metric. (default: False)
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=latent_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed)
        self.input_transformation_name = input_transformation_name
        self.translate_max = translate_max
        self.rotate_max = rotate_max
        self.test_eqt = test_eqt
        self.test_eqt_frac = test_eqt_frac
        self.test_eqr = test_eqr
        self.requires_test = test_eqt or test_eqt_frac or test_eqr

    def compute_equivariance_diff(self, generator, generator_kwargs):
        """Computes the equivariance difference with the generator."""
        latent_num = self.latent_num
        batch_size = self.batch_size
        if self.random_latents:
            g1 = torch.Generator(device=self.device)
            g1.manual_seed(self.seed)
        else:
            latent_codes = np.load(self.latent_file)[self.replica_indices]
            latent_codes = torch.from_numpy(latent_codes).to(torch.float32)
        if self.random_labels:
            g2 = torch.Generator(device=self.device)
            g2.manual_seed(self.seed)
        else:
            labels = np.load(self.label_file)[self.replica_indices]
            labels = torch.from_numpy(labels).to(torch.float32)

        G = generator
        G_kwargs = generator_kwargs
        impl = generator_kwargs.get('impl', 'cuda')
        G_mode = G.training  # save model training mode.
        G.eval()

        I = torch.eye(3, device=self.device)
        M = G
        try:
            for key in self.input_transformation_name.split('.'):
                M = getattr(M, key)
        except AttributeError as e:
            raise ValueError(f'Cannot find customizable transformation '
                             f'`{self.input_transformation_name}` from given '
                             f'generator, hence, equivariance metric cannot be '
                             f'evaluated!') from e
        if not isinstance(M, torch.Tensor) or M.shape != (3, 3):
            raise ValueError(f'`{self.input_transformation_name}` from given '
                             f'generator is an invalid transformation matrix!')

        # Seed for evaluating EQ-T.
        if self.test_eqt:
            g3 = torch.Generator(device=self.device)
            g3.manual_seed(self.seed)
        # Seed for evaluating EQ-T_frac.
        if self.test_eqt_frac:
            g4 = torch.Generator(device=self.device)
            g4.manual_seed(self.seed)
        # Seed for evaluating EQ-R.
        if self.test_eqr:
            g5 = torch.Generator(device=self.device)
            g5.manual_seed(self.seed)

        self.logger.info(f'Synthesizing {latent_num} reference images '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Synthesis', total=latent_num)
        all_results = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = min(start + batch_size, self.replica_latent_num)
            with torch.no_grad():
                # Prepare latents and labels.
                if self.random_latents:
                    batch_codes = torch.randn((end - start, *self.latent_dim),
                                              generator=g1, device=self.device)
                else:
                    batch_codes = latent_codes[start:end].cuda().detach()
                if self.random_labels:
                    if self.label_dim == 0:
                        batch_labels = torch.zeros((end - start, 0),
                                                   device=self.device)
                    else:
                        rnd_labels = torch.randint(
                            low=0, high=self.label_dim, size=(end - start,),
                            generator=g2, device=self.device)
                        batch_labels = F.one_hot(
                            rnd_labels, num_classes=self.label_dim)
                else:
                    batch_labels = labels[start:end].cuda().detach()

                # Original synthesis without any transformation.
                M[:] = I
                ori = G(batch_codes, batch_labels, **G_kwargs)['image']
                batch_results = torch.zeros((batch_codes.shape[0], 6),
                                            dtype=torch.float64,
                                            device=self.device)
                # Evaluate EQ-T.
                if self.test_eqt:
                    t = torch.rand(2, device=self.device, generator=g3)
                    t = (t * 2 - 1) * self.translate_max
                    t = (t * G.resolution).round() / G.resolution
                    M[:] = I
                    M[:2, 2] = -t
                    img = G(batch_codes, batch_labels, **G_kwargs)['image']
                    ref, mask = apply_integer_translation(ori, t[0], t[1])
                    diff = (ref - img).square() * mask
                    batch_results[:, 0] += diff.to(torch.float64).sum(
                        dim=(1, 2, 3))
                    batch_results[:, 1] += mask.to(torch.float64).sum(
                        dim=(1, 2, 3))
                # Evaluate EQ-T_frac.
                if self.test_eqt_frac:
                    t = torch.rand(2, device=self.device, generator=g4)
                    t = (t * 2 - 1) * self.translate_max
                    M[:] = I
                    M[:2, 2] = -t
                    img = G(batch_codes, batch_labels, **G_kwargs)['image']
                    ref, mask = apply_fractional_translation(
                        ori, t[0], t[1], impl=impl)
                    diff = (ref - img).square() * mask
                    batch_results[:, 2] += diff.to(torch.float64).sum(
                        dim=(1, 2, 3))
                    batch_results[:, 3] += mask.to(torch.float64).sum(
                        dim=(1, 2, 3))
                # Rotation EQ-R.
                if self.test_eqr:
                    angle = torch.rand([], device=self.device, generator=g5)
                    angle = (angle * 2 - 1) * (self.rotate_max * np.pi)
                    M[:] = rotation_matrix(-angle)
                    img = G(batch_codes, batch_labels, **G_kwargs)['image']
                    ref, ref_mask = apply_fractional_rotation(
                        ori, angle, impl=impl)
                    pseudo, pseudo_mask = apply_fractional_pseudo_rotation(
                        img, angle, impl=impl)
                    mask = ref_mask * pseudo_mask
                    diff = (ref - pseudo).square() * mask
                    batch_results[:, 4] += diff.to(torch.float64).sum(
                        dim=(1, 2, 3))
                    batch_results[:, 5] += mask.to(torch.float64).sum(
                        dim=(1, 2, 3))
                gathered_results = self.gather_batch_results(batch_results)
                self.append_batch_results(gathered_results, all_results)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_results = self.gather_all_results(all_results)[:latent_num]

        if self.is_chief:
            assert all_results.shape == (latent_num, 6)
        else:
            assert len(all_results) == 0
            all_results = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_results

    def evaluate(self, _data_loader, generator, generator_kwargs):
        if not self.requires_test:
            self.sync()
            return None

        results = self.compute_equivariance_diff(generator, generator_kwargs)
        if self.is_chief:
            result = dict()
            if self.test_eqt:
                eqt_diff = np.sum(results[:, 0]) / np.sum(results[:, 1])
                eqt_psnr = np.log10(2) * 20 - np.log10(eqt_diff) * 10
                result[f'{self.name}_eqt'] = float(eqt_psnr)
            if self.test_eqt:
                eqt_frac_diff = np.sum(results[:, 2]) / np.sum(results[:, 3])
                eqt_frac_psnr = np.log10(2) * 20 - np.log10(eqt_frac_diff) * 10
                result[f'{self.name}_eqt_frac'] = float(eqt_frac_psnr)
            if self.test_eqt:
                eqr_diff = np.sum(results[:, 4]) / np.sum(results[:, 5])
                eqr_psnr = np.log10(2) * 20 - np.log10(eqr_diff) * 10
                result[f'{self.name}_eqr'] = float(eqr_psnr)
        else:
            assert results is None
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """Higher EQ-T, EQ-T_frac, EQ-R are better."""
        if metric_name == f'{self.name}_eqt':
            return ref is None or new > ref
        if metric_name == f'{self.name}_eqt_frac':
            return ref is None or new > ref
        if metric_name == f'{self.name}_eqr':
            return ref is None or new > ref
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief or not self.requires_test:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        msg = f'Evaluating `{self.name}`: '
        if self.test_eqt:
            eqt_psnr = result[f'{self.name}_eqt']
            assert isinstance(eqt_psnr, float)
            msg += f'EQ-T {eqt_psnr:.3f}, '
        if self.test_eqt_frac:
            eqt_frac_psnr = result[f'{self.name}_eqt_frac']
            assert isinstance(eqt_frac_psnr, float)
            msg += f'EQ-T_frac {eqt_frac_psnr:.3f}, '
        if self.test_eqr:
            eqr_psnr = result[f'{self.name}_eqr']
            assert isinstance(eqr_psnr, float)
            msg += f'EQ-R {eqr_psnr:.3f}, '
        if log_suffix is None:
            msg = msg[:-2] + '.'
        else:
            msg = msg + log_suffix + '.'
        self.logger.info(msg)

        with open(os.path.join(self.work_dir, f'{self.name}.txt'), 'a+') as f:
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{date}] {msg}\n')

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning('`Tag` is missing when writing data to '
                                    'TensorBoard, hence, the data may be mixed '
                                    'up!')
            if self.test_eqt:
                self.tb_writer.add_scalar(
                    f'Metrics/{self.name}_eqt', eqt_psnr, tag)
            if self.test_eqt_frac:
                self.tb_writer.add_scalar(
                    f'Metrics/{self.name}_eqt_frac', eqt_frac_psnr, tag)
            if self.test_eqr:
                self.tb_writer.add_scalar(
                    f'Metrics/{self.name}_eqr', eqr_psnr, tag)
            self.tb_writer.flush()
        self.sync()

    def info(self):
        metric_info = super().info()
        metric_info['Transformation param name (e.g., model buffer)'] = (
            self.input_transformation_name)
        metric_info['Max translation'] = self.translate_max
        metric_info['Max rotation'] = self.rotate_max
        metric_info['Test translation equivariance'] = self.test_eqt
        metric_info['Test fractional translation equivariance'] = (
            self.test_eqt_frac)
        metric_info['Test rotation equivariance'] = self.test_eqr
        return metric_info


class EQTMetric(EquivarianceMetric):
    """Defines the class for EQ-T metric computation.

    This is a shortcut of `EquivarianceMetric`.
    """

    def __init__(self,
                 name='EQTMetric',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 input_transformation_name='synthesis.early_layer.transform',
                 translate_max=0.125):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=latent_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         input_transformation_name=input_transformation_name,
                         translate_max=translate_max,
                         test_eqt=True,
                         test_eqt_frac=False,
                         test_eqr=False)


class EQT50K(EQTMetric):
    """Defines the class for EQ-T (50K) metric computation.

    50_000 synthesis will be used as reference.
    """

    def __init__(self,
                 name='EQT50K',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 input_transformation_name='synthesis.early_layer.transform',
                 translate_max=0.125):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=50_000,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         input_transformation_name=input_transformation_name,
                         translate_max=translate_max)


class EQTFracMetric(EquivarianceMetric):
    """Defines the class for EQ-T_frac metric computation.

    This is a shortcut of `EquivarianceMetric`.
    """

    def __init__(self,
                 name='EQTFracMetric',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 input_transformation_name='synthesis.early_layer.transform',
                 translate_max=0.125):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=latent_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         input_transformation_name=input_transformation_name,
                         translate_max=translate_max,
                         test_eqt=False,
                         test_eqt_frac=True,
                         test_eqr=False)


class EQTFrac50K(EQTFracMetric):
    """Defines the class for EQ-T_frac (50K) metric computation.

    50_000 synthesis will be used as reference.
    """

    def __init__(self,
                 name='EQTFrac50K',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 input_transformation_name='synthesis.early_layer.transform',
                 translate_max=0.125):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=50_000,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         input_transformation_name=input_transformation_name,
                         translate_max=translate_max)


class EQRMetric(EquivarianceMetric):
    """Defines the class for EQ-R metric computation.

    This is a shortcut of `EquivarianceMetric`.
    """

    def __init__(self,
                 name='EQRMetric',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 input_transformation_name='synthesis.early_layer.transform',
                 rotate_max=1):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=latent_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         input_transformation_name=input_transformation_name,
                         rotate_max=rotate_max,
                         test_eqt=False,
                         test_eqt_frac=False,
                         test_eqr=True)


class EQR50K(EQRMetric):
    """Defines the class for EQ-R (50K) metric computation.

    50_000 synthesis will be used as reference.
    """

    def __init__(self,
                 name='EQR50K',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 input_transformation_name='synthesis.early_layer.transform',
                 rotate_max=1):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=50_000,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         input_transformation_name=input_transformation_name,
                         rotate_max=rotate_max)
