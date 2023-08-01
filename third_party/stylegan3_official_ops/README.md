# Operators for StyleGAN2

All files in this directory are borrowed from repository [stylegan3](https://github.com/NVlabs/stylegan3). Basically, these files implement customized operators, which are faster than the native operators from PyTorch, especially for second-derivative computation, including

- `bias_act.bias_act()`: Fuse adding bias and then performing activation as one operator.
- `upfirdn2d.setup_filter()`: Set up the kernel used for filtering.
- `upfirdn2d.filter2d()`: Filtering a 2D feature map with given kernel.
- `upfirdn2d.upsample2d()`: Upsampling a 2D feature map.
- `upfirdn2d.downsample2d()`: Downsampling a 2D feature map.
- `upfirdn2d.upfirdn2d()`: Upsampling, filtering, and then downsampling a 2D feature map.
- `filtered_lrelu.filtered_lrelu()`: Leaky ReLU layer, wrapped with upsampling and downsampling for anti-aliasing.
- `conv2d_gradfix.conv2d()`: Convolutional layer, supporting arbitrarily high order gradients and fixing gradient when computing penalty.
- `conv2d_gradfix.conv_transpose2d()`: Transposed convolutional layer, supporting arbitrarily high order gradients and fixing gradient when computing penalty.
- `conv2d_resample.conv2d_resample()`: Wraps `upfirdn2d()` and `conv2d()` (or `conv_transpose2d()`). This is not used in our network implementation (*i.e.*, `models/stylegan2_generator.py` and `models/stylegan2_discriminator.py`)

We make following slight modifications beyond disabling some lint warnings:

- Line 24 of file `misc.py`: Use `EasyDict` from module `easydict` to replace that from `dnnlib` from [stylegan3](https://github.com/NVlabs/stylegan3).
- Line 36 of file `custom_ops.py`: Disable log message when setting up customized operators.
- Line 54/109 of file `custom_ops.py`: Add necessary CUDA compiler path. (***NOTE**: If your cuda binary does not locate at `/usr/local/cuda/bin`, please specify in function `_find_compiler_bindir_posix()`.*)
- Line 21 of file `bias_act.py`: Use `EasyDict` from module `easydict` to replace that from `dnnlib` from [stylegan3](https://github.com/NVlabs/stylegan3).
- Line 162-165 of file `filtered_lrelu.py`: Change some implementations in `_filtered_lrelu_ref()` to `ref`.
- Line 31 of file `grid_sample_gradfix.py`: Enable customized grid sampling operator by default.
- Line 35 of file `grid_sample_gradfix.py`:  Use `impl` to disable customized grid sample operator.
- Line 34 of file `conv2d_gradfix.py`: Enable customized convolution operators by default.
- Line 48/53 of file `conv2d_gradfix.py`: Use `impl` to disable customized convolution operators.
- Line 36/53 of file `conv2d_resample.py`: Use `impl` to disable customized convolution operators.
- Line 23 of file `fma.py`: Use `impl` to disable customized add-multiply operator.

Please use `ref` or `cuda` to choose which implementation to use. `ref` refers to native PyTorch operators while `cuda` refers to the customized operators from the official repository. `cuda` is used by default.
