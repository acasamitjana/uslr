import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VecInt(nn.Module):
    """
    Vector Integration Layer

    Enables vector integration via several methods
    (ode or quadrature for time-dependent vector fields,
    scaling and squaring for stationary fields)

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self, field_shape, int_steps=7, **kwargs):
        """
        Parameters:
            int_steps is the number of integration steps
        """
        super().__init__()
        self.int_steps = int_steps
        self.scale = 1 / (2 ** self.int_steps)
        self.transformer = SpatialTransformer(field_shape)

    def forward(self, field, **kwargs):

        output = field
        output = output * self.scale
        nsteps = self.int_steps
        if 'nsteps' in kwargs:
            nsteps = nsteps - kwargs['nsteps']

        for _ in range(nsteps):
            a = self.transformer(output, output)
            output = output + a

        return output

class SpatialInterpolation(nn.Module):
    """
    [SpatialInterpolation] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, mode='bilinear', padding_mode='zeros'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, src, new_locs, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        if 'padding_mode' in kwargs:
            self.padding_mode = kwargs['padding_mode']
        if 'mode' in kwargs:
            self.mode = kwargs['mode']

        shape = src.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode=self.padding_mode, align_corners=True)

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, size, mode='bilinear', padding_mode='border'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, src, flow, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        padding_mode = kwargs['padding_mode'] if 'padding_mode' in kwargs else self.padding_mode
        mode = kwargs['mode'] if 'mode' in kwargs else self.mode

        new_locs = self.grid + flow
        shape = src.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            # new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode, align_corners=True)


class RescaleTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    Credit to voxelmorph: https://github.com/voxelmorph/voxelmorph/blob/redesign/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, factor=None, target_size=None, gaussian_filter_flag=False):
        '''

        :param vol_size:
        :param factor:
                :param latent_size: it only applies if factor is None

        '''
        super().__init__()

        self.ndims = len(inshape)
        self.mode = 'linear'
        self.inshape = inshape
        self.gaussian_filter_flag = gaussian_filter_flag

        if factor is None:
            assert target_size is not None
            self.factor = tuple([b/a for a, b in zip(inshape, target_size)])
        elif isinstance(factor, list) or isinstance(factor, tuple):
            self.factor = list(factor)
        else:
            self.factor = [factor for _ in range(self.ndims)]

        if self.ndims == 2:
            self.mode = 'bi' + self.mode
        elif self.ndims == 3:
            self.mode = 'tri' + self.mode

        if self.factor[0] < 1 and self.gaussian_filter_flag:
            kernel_sigma = [0.44 * 1 / f for f in self.factor]

            if self.ndims == 2:
                kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)
            elif self.ndims == 3:
                kernel = self.gaussian_filter_3d(kernel_sigma=kernel_sigma)
            else:
                raise ValueError('[RESCALE TF] No valid kernel found.')
            self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def gaussian_filter_3d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX, ZZ = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
        xyz_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis], ZZ[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1, 1, 1, 1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1, 1, 1, 1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)
        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xyz_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1],kernel_size[2])

        # Total kernel

        total_kernel = np.zeros((3,3) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel
        total_kernel[2, 2] = kernel


        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x):

        x = x.clone()
        if self.factor[0] < 1:
            if self.gaussian_filter_flag:
                padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
                if self.ndims == 2:
                    x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)
                else:
                    x = F.conv3d(x, self.kernel, stride=(1, 1, 1), padding=padding)

            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]

        elif self.factor[0] > 1:
            # multiply first to save memory
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    Credit to voxelmorph: https://github.com/voxelmorph/voxelmorph/blob/redesign/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, target_size=None, factor=None, gaussian_filter_flag=True):
        '''

        :param vol_size:
        :param factor: if factor<1 the shape is reduced and viceversa.
        :param latent_size: it only applies if factor is None
        '''
        super().__init__()

        self.ndims = len(inshape)
        self.mode = 'linear'
        self.inshape = inshape
        self.gaussian_filter_flag = gaussian_filter_flag
        if self.ndims == 2:
            self.mode = 'bi' + self.mode
        elif self.ndims == 3:
            self.mode = 'tri' + self.mode

        if target_size is None:
            self.factor = factor
            if isinstance(factor, float) or isinstance(factor, int):
                self.factor = [factor for _ in range(self.ndims)]
        else:
            self.factor = tuple([b/a for a, b in zip(inshape, target_size)])

        if self.factor[0] < 1 and self.gaussian_filter_flag:

            kernel_sigma = [0.44 / f for f in self.factor]
            if self.ndims == 2:
                kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)
            elif self.ndims == 3:
                kernel = self.gaussian_filter_3d(kernel_sigma=kernel_sigma)
            else:
                raise ValueError('[RESCALE TF] No valid kernel found.')
            self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def gaussian_filter_3d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX, ZZ = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
        xyz_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis], ZZ[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1, 1, 1, 1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1, 1, 1, 1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)
        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xyz_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1],kernel_size[2])

        # Total kernel

        total_kernel = np.zeros((3,3) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel
        total_kernel[2, 2] = kernel


        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x, *args, **kwargs):

        x = x.clone()
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode=self.mode

        if self.gaussian_filter_flag and self.factor[0] < 1:
            padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
            if self.ndims == 2:
                x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)
            else:
                x = F.conv3d(x, self.kernel, stride=(1, 1, 1), padding=padding)

        x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=mode)

        return x


