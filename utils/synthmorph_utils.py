#!/usr/bin/env python
import copy
import os
import argparse
import pdb
import numpy as np
import nibabel as nib
from itertools import combinations
from scipy.ndimage import label as scipy_label
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, distance_transform_edt, binary_fill_holes

import tensorflow as tf
from tensorflow import keras
import tensorflow.python.keras.initializers as KI
import tensorflow.python.keras.layers as KL
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import neurite as ne
import torch

from utils import io_utils

# set tensorflow logging
K.set_image_data_format('channels_last')
tf.get_logger().setLevel('ERROR')
# tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


repo_home = os.environ.get('PYTHONPATH')

path_model_registration = os.path.join(repo_home, 'data', 'models', 'brains-dice-vel-0.5-res-16-256f.h5')
labels_segmentation = os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_labels_2.0.npy')
labels_parcellation = os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_labels.npy')
labels_registration = os.path.join(repo_home, 'data', 'labels_classes_priors', 'label_list_registration.npy')
atlas_cog_file = os.path.join(repo_home, 'data', 'atlas', 'atlas_synthmorph.cog.npy')
atlas_seg_file = os.path.join(repo_home, 'data', 'atlas', 'atlas_seg.nii.gz')
atlas_mask_file = os.path.join(repo_home, 'data', 'atlas', 'atlas_mask.nii.gz')
atlas_file = os.path.join(repo_home, 'data', 'atlas', 'atlas.nii.gz')


def getM(ref, mov, use_L1=False):

    zmat = np.zeros(ref.shape[::-1])
    zcol = np.zeros([ref.shape[1], 1])
    ocol = np.ones([ref.shape[1], 1])
    zero = np.zeros(zmat.shape)

    A = np.concatenate([
        np.concatenate([np.transpose(ref), zero, zero, ocol, zcol, zcol], axis=1),
        np.concatenate([zero, np.transpose(ref), zero, zcol, ocol, zcol], axis=1),
        np.concatenate([zero, zero, np.transpose(ref), zcol, zcol, ocol], axis=1)], axis=0)

    b = np.concatenate([np.transpose(mov[0, :]), np.transpose(mov[1, :]), np.transpose(mov[2, :])], axis=0)

    x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A), b))

    # If L1 minimization: we use L2 solution as initialization and miminize L1
    if use_L1:
        from scipy.optimize import linprog
        Apos = np.concatenate([A, -np.eye(A.shape[0])], axis=1)
        Aneg = np.concatenate([-A, -np.eye(A.shape[0])], axis=1)
        A_ub = np.concatenate([Apos, Aneg], axis=0)
        b_ub = np.concatenate([b, -b])
        c = np.concatenate([np.zeros(12), np.ones(A.shape[0])])

        aux = linprog(c, A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=[None, None],
                      options={'disp': True, 'autoscale': True},
                      x0=np.concatenate([x, 0.1 + np.abs(np.matmul(A, x) - b)]))
        x = aux.x[0:12]


    M = np.stack([
        [x[0], x[1], x[2], x[9]],
        [x[3], x[4], x[5], x[10]],
        [x[6], x[7], x[8], x[11]],
        [0, 0, 0, 1]])

    return M

def validate_affine_shape(shape):
    """
    Validates whether the given input shape represents a valid affine matrix.
    Throws error if the shape is valid.

    Parameters:
        shape: List of integers of the form [..., N, N+1].
    """
    ndim = shape[-1] - 1
    actual = tuple(shape[-2:])
    if ndim not in (2, 3) or actual != (ndim, ndim + 1):
        raise ValueError(f'Affine matrix must be of shape (2, 3) or (3, 4), got {actual}.')


def affine_to_dense_shift(matrix, shape, shift_center=True, indexing='ij'):
    """
    Transforms an affine matrix to a dense location shift.

    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply affine matrix to each index.
        3. Subtract grid.

    Parameters:
        matrix: affine matrix of shape (N, N+1).
        shape: ND shape of the target warp.
        shift_center: Shift grid to image center.
        indexing: Must be 'xy' or 'ij'.

    Returns:
        Dense shift (warp) of shape (*shape, N).
    """

    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
        matrix = tf.cast(matrix, tf.float32)

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix.shape)

    # list of volume ndgrid
    # N-long list, each entry of shape
    mesh = ne.utils.volshape_to_meshgrid(shape, indexing=indexing)
    mesh = [f if f.dtype == matrix.dtype else tf.cast(f, matrix.dtype) for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (shape[f] - 1) / 2 for f in range(len(shape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [ne.utils.flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype=matrix.dtype))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:ndims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(shape) + [ndims])  # *shape x N

    # get shifts and return
    return loc - tf.stack(mesh, axis=ndims)


def crop_volume_with_idx(volume, crop_idx, aff=None, n_dims=None, return_copy=True):
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: croppping indices, in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if aff is specified, this function returns an updated affine matrix of the volume after
    cropping.
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    n_dims = int(np.array(crop_idx).shape[0] / 2) if n_dims is None else n_dims

    # crop image
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return new_volume, aff
    else:
        return new_volume


def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()

def mask_volume(volume, mask=None, threshold=0.1, dilate=0, erode=0, fill_holes=False, masking_value=0,
                return_mask=False, return_copy=True):
    """Mask a volume, either with a given mask, or by keeping only the values above a threshold.
    :param volume: a numpy array, possibly with several channels
    :param mask: (optional) a numpy array to mask volume with.
    Mask doesn't have to be a 0/1 array, all strictly positive values of mask are considered for masking volume.
    Mask should have the same size as volume. If volume has several channels, mask can either be uni- or multi-channel.
     In the first case, the same mask is applied to all channels.
    :param threshold: (optional) If mask is None, masking is performed by keeping thresholding the input.
    :param dilate: (optional) number of voxels by which to dilate the provided or computed mask.
    :param erode: (optional) number of voxels by which to erode the provided or computed mask.
    :param fill_holes: (optional) whether to fill the holes in the provided or computed mask.
    :param masking_value: (optional) masking value
    :param return_mask: (optional) whether to return the applied mask
    :return: the masked volume, and the applied mask if return_mask is True.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    vol_shape = list(new_volume.shape)
    n_dims, n_channels = get_dims(vol_shape)

    # get mask and erode/dilate it
    if mask is None:
        mask = new_volume >= threshold
    else:
        assert list(mask.shape[:n_dims]) == vol_shape[:n_dims], 'mask should have shape {0}, or {1}, had {2}'.format(
            vol_shape[:n_dims], vol_shape[:n_dims] + [n_channels], list(mask.shape))
        mask = mask > 0
    if dilate > 0:
        dilate_struct = build_binary_structure(dilate, n_dims)
        mask_to_apply = binary_dilation(mask, dilate_struct)
    else:
        mask_to_apply = mask
    if erode > 0:
        erode_struct = build_binary_structure(erode, n_dims)
        mask_to_apply = binary_erosion(mask_to_apply, erode_struct)
    if fill_holes:
        mask_to_apply = binary_fill_holes(mask_to_apply)

    # replace values outside of mask by padding_char
    if mask_to_apply.shape == new_volume.shape:
        new_volume[np.logical_not(mask_to_apply)] = masking_value
    else:
        new_volume[np.stack([np.logical_not(mask_to_apply)] * n_channels, axis=-1)] = masking_value

    if return_mask:
        return new_volume, mask_to_apply
    else:
        return new_volume

def build_binary_structure(connectivity, n_dims, shape=None):
    """Return a dilation/erosion element with provided connectivity"""
    if shape is None:
        shape = [connectivity * 2 + 1] * n_dims
    else:
        shape = reformat_to_list(shape, length=n_dims)
    dist = np.ones(shape)
    center = tuple([tuple([int(s / 2)]) for s in shape])
    dist[center] = 0
    dist = distance_transform_edt(dist)
    struct = (dist <= connectivity) * 1
    return struct


def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


def _conv_block(x, nfeat, strides=1, name=None, do_res=False, hyp_tensor=None,
                include_activation=True, kernel_initializer='he_normal'):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims

    extra_conv_params = {}
    if hyp_tensor is not None:
        Conv = getattr(ne.layers, 'HyperConv%dDFromDense' % ndims)
        conv_inputs = [x, hyp_tensor]
    else:
        Conv = getattr(KL, 'Conv%dD' % ndims)
        extra_conv_params['kernel_initializer'] = kernel_initializer
        conv_inputs = x

    convolved = Conv(nfeat, kernel_size=3, padding='same',
                     strides=strides, name=name, **extra_conv_params)(conv_inputs)

    if do_res:
        # assert nfeat == x.get_shape()[-1], 'for residual number of features should be constant'
        add_layer = x
        print('note: this is a weird thing to do, since its not really residual training anymore')
        if nfeat != x.get_shape().as_list()[-1]:
            add_layer = Conv(nfeat, kernel_size=3, padding='same',
                             name='resfix_' + name, **extra_conv_params)(conv_inputs)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])

    if include_activation:
        name = name + '_activation' if name else None
        convolved = KL.LeakyReLU(0.2, name=name)(convolved)

    return convolved

def _upsample_block(x, connection, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

    size = (factor,) * ndims if ndims > 1 else factor
    upsampled = UpSampling(size=size, name=name)(x)
    name = name + '_concat' if name else None
    return KL.concatenate([upsampled, connection], name=name)

def resample_volume(volume, aff, new_vox_size, interpolation='linear'):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = RegularGridInterpolator((x, y, z), volume_filt, method=interpolation)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def mkdir(path_dir):
    """Recursively creates the current dir as well as its parent folders if they do not already exist."""
    if path_dir[-1] == '/':
        path_dir = path_dir[:-1]
    if not os.path.isdir(path_dir):
        list_dir_to_create = [path_dir]
        while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
            list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
        for dir_to_create in reversed(list_dir_to_create):
            os.mkdir(dir_to_create)

def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is None, where
    n_dims is automatically inferred.
    """

    mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty.set_data_dtype(dtype)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)




def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume



def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)

def transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.

    Args:
        vol (Tensor): volume with size vol_shape or [*vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [*new_vol_shape, D] or [*new_vol_shape, C, D]
            where C is the number of channels, and D is the dimentionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes.
    # location volshape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing=indexing)  # volume mesh
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = tf.cast(m, loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


def integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    """
    Integrate (stationary of time-dependent) vector field (N-D Tensor) in tensorflow

    Aside from directly using tensorflow's numerical integration odeint(), also implements
    "scaling and squaring", and quadrature. Note that the diff. equation given to odeint
    is the one used in quadrature.

    Parameters:
        vec: the Tensor field to integrate.
            If vol_size is the size of the intrinsic volume, and vol_ndim = len(vol_size),
            then vector shape (vec_shape) should be
            [vol_size, vol_ndim] (if stationary)
            [vol_size, vol_ndim, nb_time_steps] (if time dependent)
        time_dep: bool whether vector is time dependent
        method: 'scaling_and_squaring' or 'ss' or 'ode' or 'quadrature'

        if using 'scaling_and_squaring': currently only supports integrating to time point 1.
            nb_steps: int number of steps. Note that this means the vec field gets broken
            down to 2**nb_steps. so nb_steps of 0 means integral = vec.

        if using 'ode':
            out_time_pt (optional): a time point or list of time points at which to evaluate
                Default: 1
            init (optional): if using 'ode', the initialization method.
                Currently only supporting 'zero'. Default: 'zero'
            ode_args (optional): dictionary of all other parameters for
                tf.contrib.integrate.odeint()

    Returns:
        int_vec: integral of vector field.
        Same shape as the input if method is 'scaling_and_squaring', 'ss', 'quadrature',
        or 'ode' with out_time_pt not a list. Will have shape [*vec_shape, len(out_time_pt)]
        if method is 'ode' with out_time_pt being a list.

    Todo:
        quadrature for more than just intrinsically out_time_pt = 1
    """

    int_end = 1
    if 'int_end' in kwargs: int_end=kwargs['int_end']
    if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
        raise ValueError("method has to be 'scaling_and_squaring' or 'ode'. found: %s" % method)

    if method in ['ss', 'scaling_and_squaring']:
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

        if time_dep:
            svec = K.permute_dimensions(vec, [-1, *range(0, vec.shape[-1] - 1)])
            assert 2**nb_steps == svec.shape[0], "2**nb_steps and vector shape don't match"

            svec = svec / (2**nb_steps)
            for _ in range(nb_steps):
                svec = svec[0::2] + tf.map_fn(transform, svec[1::2, :], svec[0::2, :])

            disp = svec[0, :]

        else:
            vec = vec / (2**nb_steps) * int_end
            for _ in range(nb_steps):
                vec += transform(vec, vec)
            disp = vec

    elif method == 'quadrature':
        # TODO: could output more than a single timepoint!
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 1, 'nb_steps should be >= 1, found: %d' % nb_steps

        vec = vec / nb_steps

        if time_dep:
            disp = vec[..., 0]
            for si in range(nb_steps - 1):
                disp += transform(vec[..., si + 1], disp)
        else:
            disp = vec
            for _ in range(nb_steps - 1):
                disp += transform(vec, disp)

    else:
        assert not time_dep, "odeint not implemented with time-dependent vector field"
        fn = lambda disp, _: transform(vec, disp)

        # process time point.
        out_time_pt = kwargs['out_time_pt'] if 'out_time_pt' in kwargs.keys() else 1
        out_time_pt = tf.cast(K.flatten(out_time_pt), tf.float32)
        len_out_time_pt = out_time_pt.get_shape().as_list()[0]
        assert len_out_time_pt is not None, 'len_out_time_pt is None :('
        # initializing with something like tf.zeros(1) gives a control flow issue.
        z = out_time_pt[0:1] * 0.0
        K_out_time_pt = K.concatenate([z, out_time_pt], 0)

        # enable a new integration function than tf.contrib.integrate.odeint
        odeint_fn = tf.contrib.integrate.odeint
        if 'odeint_fn' in kwargs.keys() and kwargs['odeint_fn'] is not None:
            odeint_fn = kwargs['odeint_fn']

        # process initialization
        if 'init' not in kwargs.keys() or kwargs['init'] == 'zero':
            disp0 = vec * 0  # initial displacement is 0
        else:
            raise ValueError('non-zero init for ode method not implemented')

        # compute integration with odeint
        if 'ode_args' not in kwargs.keys():
            kwargs['ode_args'] = {}
        disp = odeint_fn(fn, disp0, K_out_time_pt, **kwargs['ode_args'])
        disp = K.permute_dimensions(disp[1:len_out_time_pt + 1, :], [*range(1, len(disp.shape)), 0])

        # return
        if len_out_time_pt == 1:
            disp = disp[..., 0]

    return disp



def crop_volume(volume, cropping_margin=None, cropping_shape=None, aff=None, return_crop_idx=False, mode='center'):
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume. The cropping margin is applied on both sides.
    Can be an int, sequence or 1d numpy array of size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped. Can be an int, sequence or 1d numpy
    array of size n_dims. Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
    If not None, this function also returns an updated version of the affine matrix for the cropped volume.
    :param return_crop_idx: (optional) whether to return the cropping indices used to crop the given volume.
    :param mode: (optional) if cropping_shape is not None, whether to extract the centre of the image (mode='center'),
    or to randomly crop the volume to the provided shape (mode='random'). Default is 'center'.
    :return: cropped volume, corresponding affine matrix if aff is not None, and cropping indices if return_crop_idx is
    True (in that order).
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = reformat_to_list(cropping_margin, length=n_dims)
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [vol_shape[i] - cropping_margin[i] if do_cropping[i] else vol_shape[i] for i in range(n_dims)]
    else:
        cropping_shape = reformat_to_list(cropping_shape, length=n_dims)
        if mode == 'center':
            min_crop_idx = np.maximum([int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0)
            max_crop_idx = np.minimum([min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                                      np.array(vol_shape)[:n_dims])
        elif mode == 'random':
            crop_max_val = np.maximum(np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0)
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims])
        else:
            raise ValueError('mode should be either "center" or "random", had %s' % mode)
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]

    # sort outputs
    output = [new_volume]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)
    return output[0] if len(output) == 1 else tuple(output)



def rescale_volume(volume, new_min=0, new_max=255, min_percentile=2., max_percentile=98., use_positive_only=False):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)




def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :return: padded volume, and updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = get_dims(vol_shape)
    padding_shape = reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # check if need to pad
    if np.any(np.array(padding_shape, dtype='int32') > np.array(vol_shape[:n_dims], dtype='int32')):

        # get padding margins
        min_margins = np.maximum(np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        max_margins = np.maximum(np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])

        # pad volume
        new_volume = np.pad(new_volume, pad_margins, mode='constant', constant_values=padding_value)

        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins

    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])

    # sort outputs
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)



def add_axis(x, axis=0):
    """Add axis to a numpy array.
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    axis = reformat_to_list(axis)
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x



def build_seg_model(model_file_segmentation,
                model_file_parcellation,
                labels_segmentation,
                labels_parcellation):

    if not os.path.isfile(model_file_segmentation):
        raise ValueError("The provided model path does not exist.")

    # get labels
    n_labels_seg = len(labels_segmentation)

    # build UNet
    net = unet(nb_features=24,
               input_shape=[None, None, None, 1],
               nb_levels=5,
               conv_size=3,
               nb_labels=n_labels_seg,
               feat_mult=2,
               activation='elu',
               nb_conv_per_level=2,
               batch_norm=-1,
               name='unet')
    net.load_weights(model_file_segmentation, by_name=True)
    input_image = net.inputs[0]
    name_segm_prediction_layer = 'unet_prediction'

    # smooth posteriors
    last_tensor = net.output
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    last_tensor = GaussianBlur(sigma=0.5)(last_tensor)
    net = keras.Model(inputs=net.inputs, outputs=last_tensor)

    # add aparc segmenter
    n_labels_parcellation = len(labels_parcellation)

    last_tensor = net.output
    last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(last_tensor)
    last_tensor = ConvertLabels(np.arange(n_labels_seg), labels_segmentation)(last_tensor)
    parcellation_masking_values = np.array([1 if ((ll == 3) | (ll == 42)) else 0 for ll in labels_segmentation])
    last_tensor = ConvertLabels(labels_segmentation, parcellation_masking_values)(last_tensor)
    last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=2, axis=-1))(last_tensor)
    last_tensor = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
    net = keras.Model(inputs=net.inputs, outputs=last_tensor)

    # build UNet
    net = unet(nb_features=24,
               input_shape=[None, None, None, 3],
               nb_levels=5,
               conv_size=3,
               nb_labels=n_labels_parcellation,
               feat_mult=2,
               activation='elu',
               nb_conv_per_level=2,
               batch_norm=-1,
               name='unet_parc',
               input_model=net)
    net.load_weights(model_file_parcellation, by_name=True)

    # smooth predictions
    last_tensor = net.output
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    last_tensor = GaussianBlur(sigma=0.5)(last_tensor)
    net = keras.Model(inputs=net.inputs, outputs=[net.get_layer(name_segm_prediction_layer).output, last_tensor])

    return net


def unet(nb_features,
         input_shape,
         nb_levels,
         conv_size,
         nb_labels,
         name='unet',
         prefix=None,
         feat_mult=1,
         pool_size=2,
         padding='same',
         dilation_rate_mult=1,
         activation='elu',
         skip_n_concatenations=0,
         use_residuals=False,
         final_pred_activation='softmax',
         nb_conv_per_level=1,
         layer_nb_feats=None,
         conv_dropout=0,
         batch_norm=None,
         input_model=None):
    """
    Unet-style keras model with an overdose of parametrization. Copied with permission
    from github.com/adalca/neurite.
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # get encoding model
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         layer_nb_feats=layer_nb_feats,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm,
                         input_model=input_model)

    # get decoder
    # use_skip_connections=True makes it a u-net
    lnf = layer_nb_feats[(nb_levels * nb_conv_per_level):] if layer_nb_feats is not None else None
    dec_model = conv_dec(nb_features,
                         None,
                         nb_levels,
                         conv_size,
                         nb_labels,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         use_skip_connections=True,
                         skip_n_concatenations=skip_n_concatenations,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         final_pred_activation=final_pred_activation,
                         nb_conv_per_level=nb_conv_per_level,
                         batch_norm=batch_norm,
                         layer_nb_feats=lnf,
                         conv_dropout=conv_dropout,
                         input_model=enc_model)
    final_model = dec_model

    return final_model



def conv_enc(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             dilation_rate_mult=1,
             padding='same',
             activation='elu',
             layer_nb_feats=None,
             use_residuals=False,
             nb_conv_per_level=2,
             conv_dropout=0,
             batch_norm=None,
             input_model=None):
    """
    Fully Convolutional Encoder. Copied with permission from github.com/adalca/neurite.
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # first layer: input
    name = '%s_input' % prefix
    if input_model is None:
        input_tensor = keras.layers.Input(shape=input_shape, name=name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.inputs
        last_tensor = input_model.outputs
        if isinstance(last_tensor, list):
            last_tensor = last_tensor[0]

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # prepare layers
    convL = getattr(keras.layers, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation, 'data_format': 'channels_last'}
    maxpool = getattr(keras.layers, 'MaxPooling%dD' % ndims)

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    lfidx = 0  # level feature index
    for level in range(nb_levels):
        lvl_first_tensor = last_tensor
        nb_lvl_feats = np.round(nb_features * feat_mult ** level).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult ** level

        for conv in range(nb_conv_per_level):  # does several conv per level, max pooling applied at the end
            if layer_nb_feats is not None:  # None or List of all the feature numbers
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_downarm_%d_%d' % (prefix, level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:  # no activation
                last_tensor = convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                # conv dropout along feature space only
                name = '%s_dropout_downarm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

        if use_residuals:
            convarm_layer = last_tensor

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            nb_feats_in = lvl_first_tensor.get_shape()[-1]
            nb_feats_out = convarm_layer.get_shape()[-1]
            add_layer = lvl_first_tensor
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_down_merge_%d' % (prefix, level)
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(lvl_first_tensor)
                add_layer = last_tensor

                if conv_dropout > 0:
                    name = '%s_dropout_down_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]

            name = '%s_res_down_merge_%d' % (prefix, level)
            last_tensor = keras.layers.add([add_layer, convarm_layer], name=name)

            name = '%s_res_down_merge_act_%d' % (prefix, level)
            last_tensor = keras.layers.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_down_%d' % (prefix, level)
            last_tensor = keras.layers.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

        # max pool if we're not at the last level
        if level < (nb_levels - 1):
            name = '%s_maxpool_%d' % (prefix, level)
            last_tensor = maxpool(pool_size=pool_size, name=name, padding=padding)(last_tensor)

    # create the model and return
    model = keras.Model(inputs=input_tensor, outputs=[last_tensor], name=model_name)
    return model


def conv_dec(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             nb_labels,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             use_skip_connections=False,
             skip_n_concatenations=0,
             padding='same',
             dilation_rate_mult=1,
             activation='elu',
             use_residuals=False,
             final_pred_activation='softmax',
             nb_conv_per_level=2,
             layer_nb_feats=None,
             batch_norm=None,
             conv_dropout=0,
             input_model=None):
    """
    Fully Convolutional Decoder. Copied with permission from github.com/adalca/neurite.

    Parameters:
        ...
        use_skip_connections (bool): if true, turns an Enc-Dec to a U-Net.
            If true, input_tensor and tensors are required.
            It assumes a particular naming of layers. conv_enc...
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # if using skip connections, make sure need to use them.
    if use_skip_connections:
        assert input_model is not None, "is using skip connections, tensors dictionary is required"

    # first layer: input
    input_name = '%s_input' % prefix
    if input_model is None:
        input_tensor = keras.layers.Input(shape=input_shape, name=input_name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.input
        last_tensor = input_model.output
        input_shape = last_tensor.shape.as_list()[1:]

    # vol size info
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        if ndims > 1:
            pool_size = (pool_size,) * ndims

    # prepare layers
    convL = getattr(keras.layers, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    upsample = getattr(keras.layers, 'UpSampling%dD' % ndims)

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    lfidx = 0
    for level in range(nb_levels - 1):
        nb_lvl_feats = np.round(nb_features * feat_mult ** (nb_levels - 2 - level)).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult ** (nb_levels - 2 - level)

        # upsample matching the max pooling layers size
        name = '%s_up_%d' % (prefix, nb_levels + level)
        last_tensor = upsample(size=pool_size, name=name)(last_tensor)
        up_tensor = last_tensor

        # merge layers combining previous layer
        if use_skip_connections & (level < (nb_levels - skip_n_concatenations - 1)):
            conv_name = '%s_conv_downarm_%d_%d' % (prefix, nb_levels - 2 - level, nb_conv_per_level - 1)
            cat_tensor = input_model.get_layer(conv_name).output
            name = '%s_merge_%d' % (prefix, nb_levels + level)
            last_tensor = keras.layers.concatenate([cat_tensor, last_tensor], axis=ndims + 1, name=name)

        # convolution layers
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:
                last_tensor = convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                name = '%s_dropout_uparm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

        # residual block
        if use_residuals:

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            add_layer = up_tensor
            nb_feats_in = add_layer.get_shape()[-1]
            nb_feats_out = last_tensor.get_shape()[-1]
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_up_merge_%d' % (prefix, level)
                add_layer = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(add_layer)

                if conv_dropout > 0:
                    name = '%s_dropout_up_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                    last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

            name = '%s_res_up_merge_%d' % (prefix, level)
            last_tensor = keras.layers.add([last_tensor, add_layer], name=name)

            name = '%s_res_up_merge_act_%d' % (prefix, level)
            last_tensor = keras.layers.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_up_%d' % (prefix, level)
            last_tensor = keras.layers.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # Compute likelyhood prediction (no activation yet)
    name = '%s_likelihood' % prefix
    last_tensor = convL(nb_labels, 1, activation=None, name=name)(last_tensor)
    like_tensor = last_tensor

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location
    if final_pred_activation == 'softmax':
        name = '%s_prediction' % prefix
        softmax_lambda_fcn = lambda x: keras.activations.softmax(x, axis=ndims + 1)
        pred_tensor = keras.layers.Lambda(softmax_lambda_fcn, name=name)(last_tensor)

    # otherwise create a layer that does nothing.
    else:
        name = '%s_prediction' % prefix
        pred_tensor = keras.layers.Activation('linear', name=name)(like_tensor)

    # create the model and retun
    model = keras.Model(inputs=input_tensor, outputs=pred_tensor, name=model_name)
    return model


def is_affine_shape(shape):
    """
    Determins whether the given shape (single-batch) represents an
    affine matrix.

    Parameters:
        shape:  List of integers of the form [N, N+1], assuming an affine.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False

def rescale_affine(mat, factor):
    """
    Rescales affine matrix by some factor.

    Parameters:
        mat: Affine matrix of shape [..., N, N+1].
        factor: Zoom factor.
    """
    scaled_translation = tf.expand_dims(mat[..., -1] * factor, -1)
    scaled_matrix = tf.concat([mat[..., :-1], scaled_translation], -1)
    return scaled_matrix

def rescale_dense_transform(transform, factor, interp_method='linear'):
    """
    Rescales a dense transform. this involves resizing and rescaling the vector field.

    Parameters:
        transform: A dense warp of shape [..., D1, ..., DN, N].
        factor: Scaling factor.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
    """

    def single_batch(trf):
        if factor < 1:
            trf = ne.utils.resize(trf, factor, interp_method=interp_method)
            trf = trf * factor
        else:
            # multiply first to save memory (multiply in smaller space)
            trf = trf * factor
            trf = ne.utils.resize(trf, factor, interp_method=interp_method)
        return trf

    # enable batched or non-batched input
    if len(transform.shape) > (transform.shape[-1] + 1):
        rescaled = tf.map_fn(single_batch, transform)
    else:
        rescaled = single_batch(transform)

    return rescaled


def get_list_labels(label_list=None, save_label_list=None, FS_sort=False):
    """This function reads or computes a list of all label values used in a set of label maps.
    It can also sort all labels according to FreeSurfer lut.
    :param label_list: (optional) already computed label_list. Can be a sequence, a 1d numpy array, or the path to
    a numpy 1d array.
    :param labels_dir: (optional) if path_label_list is None, the label list is computed by reading all the label maps
    in the given folder. Can also be the path to a single label map.
    :param save_label_list: (optional) path where to save the label list.
    :param FS_sort: (optional) whether to sort label values according to the FreeSurfer classification.
    If true, the label values will be ordered as follows: neutral labels first (i.e. non-sided), left-side labels,
    and right-side labels. If FS_sort is True, this function also returns the number of neutral labels in label_list.
    :return: the label list (numpy 1d array), and the number of neutral (i.e. non-sided) labels if FS_sort is True.
    If one side of the brain is not represented at all in label_list, all labels are considered as neutral, and
    n_neutral_labels = len(label_list).
    """

    # load label list if previously computed
    label_list = np.array(reformat_to_list(label_list, load_as_numpy=True, dtype='int'))


    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    if FS_sort:
        neutral_FS_labels = [0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                             109, 165, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
                             251, 252, 253, 254, 255, 258, 259, 260, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
                             502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 530,
                             531, 532, 533, 534, 535, 536, 537]
        neutral = list()
        left = list()
        right = list()
        for la in label_list:
            if la in neutral_FS_labels:
                if la not in neutral:
                    neutral.append(la)
            elif (0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (135 < la < 139) | (1000 <= la <= 1035) | \
                    (la == 865) | (20100 < la < 20110):
                if la not in left:
                    left.append(la)
            elif (39 < la < 72) | (162 < la < 165) | (2000 <= la <= 2035) | (20000 < la < 20010) | (la == 139) | \
                    (la == 866):
                if la not in right:
                    right.append(la)
            else:
                raise Exception('label {} not in our current FS classification, '
                                'please update get_list_labels in utils.py'.format(la))
        label_list = np.concatenate([sorted(neutral), sorted(left), sorted(right)])
        if ((len(left) > 0) & (len(right) > 0)) | ((len(left) == 0) & (len(right) == 0)):
            n_neutral_labels = len(neutral)
        else:
            n_neutral_labels = len(label_list)

    # save labels if specified
    if save_label_list is not None:
        np.save(save_label_list, np.int32(label_list))

    if FS_sort:
        return np.int32(label_list), n_neutral_labels
    else:
        return np.int32(label_list), None


def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var


def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var




def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)

def meshgrid(*args, **kwargs):
    """
    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    """

    indexing = kwargs.pop("indexing", "xy")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                        "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype
    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow.
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):
        stack_sz = [*sz[:i], 1, *sz[(i + 1):]]
        if indexing == 'xy' and ndim > 1 and i < 2:
            stack_sz[0], stack_sz[1] = stack_sz[1], stack_sz[0]
        output[i] = tf.tile(output[i], tf.stack(stack_sz))
    return output

def gaussian_kernel(sigma, max_sigma=None, blur_range=None, separable=True):
    """Build gaussian kernels of the specified standard deviation. The outputs are given as tensorflow tensors.
    :param sigma: standard deviation of the tensors. Can be given as a list/numpy array or as tensors. In each case,
    sigma must have the same length as the number of dimensions of the volume that will be blurred with the output
    tensors (e.g. sigma must have 3 values for 3D volumes).
    :param max_sigma:
    :param blur_range:
    :param separable:
    :return:
    """
    # convert sigma into a tensor
    if not tf.is_tensor(sigma):
        sigma_tens = tf.convert_to_tensor(reformat_to_list(sigma), dtype='float32')
    else:
        assert max_sigma is not None, 'max_sigma must be provided when sigma is given as a tensor'
        sigma_tens = sigma
    shape = sigma_tens.get_shape().as_list()

    # get n_dims and batchsize
    if shape[0] is not None:
        n_dims = shape[0]
        batchsize = None
    else:
        n_dims = shape[1]
        batchsize = tf.split(tf.shape(sigma_tens), [1, -1])[0]

    # reformat max_sigma
    if max_sigma is not None:  # dynamic blurring
        max_sigma = np.array(reformat_to_list(max_sigma, length=n_dims))
    else:  # sigma is fixed
        max_sigma = np.array(reformat_to_list(sigma, length=n_dims))

    # randomise the burring std dev and/or split it between dimensions
    if blur_range is not None:
        if blur_range != 1:
            sigma_tens = sigma_tens * tf.random.uniform(tf.shape(sigma_tens), minval=1 / blur_range, maxval=blur_range)

    # get size of blurring kernels
    windowsize = np.int32(np.ceil(2.5 * max_sigma) / 2) * 2 + 1

    if separable:

        split_sigma = tf.split(sigma_tens, [1] * n_dims, axis=-1)

        kernels = list()
        comb = np.array(list(combinations(list(range(n_dims)), n_dims - 1))[::-1])
        for (i, wsize) in enumerate(windowsize):

            if wsize > 1:

                # build meshgrid and replicate it along batch dim if dynamic blurring
                locations = tf.cast(tf.range(0, wsize), 'float32') - (wsize - 1) / 2
                if batchsize is not None:
                    locations = tf.tile(tf.expand_dims(locations, axis=0),
                                        tf.concat([batchsize, tf.ones(tf.shape(tf.shape(locations)), dtype='int32')],
                                                  axis=0))
                    comb[i] += 1

                # compute gaussians
                exp_term = -K.square(locations) / (2 * split_sigma[i] ** 2)
                g = tf.exp(exp_term - tf.math.log(np.sqrt(2 * np.pi) * split_sigma[i]))
                g = g / tf.reduce_sum(g)

                for axis in comb[i]:
                    g = tf.expand_dims(g, axis=axis)
                kernels.append(tf.expand_dims(tf.expand_dims(g, -1), -1))

            else:
                kernels.append(None)

    else:

        # build meshgrid
        mesh = [tf.cast(f, 'float32') for f in volshape_to_meshgrid(windowsize, indexing='ij')]
        diff = tf.stack([mesh[f] - (windowsize[f] - 1) / 2 for f in range(len(windowsize))], axis=-1)

        # replicate meshgrid to batch size and reshape sigma_tens
        if batchsize is not None:
            diff = tf.tile(tf.expand_dims(diff, axis=0),
                           tf.concat([batchsize, tf.ones(tf.shape(tf.shape(diff)), dtype='int32')], axis=0))
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=1)
        else:
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=0)

        # compute gaussians
        sigma_is_0 = tf.equal(sigma_tens, 0)
        exp_term = -K.square(diff) / (2 * tf.where(sigma_is_0, tf.ones_like(sigma_tens), sigma_tens)**2)
        norms = exp_term - tf.math.log(tf.where(sigma_is_0, tf.ones_like(sigma_tens), np.sqrt(2 * np.pi) * sigma_tens))
        kernels = K.sum(norms, -1)
        kernels = tf.exp(kernels)
        kernels /= tf.reduce_sum(kernels)
        kernels = tf.expand_dims(tf.expand_dims(kernels, -1), -1)

    return kernels



def get_mapping_lut(source, dest=None):
    """This functions returns the look-up table to map a list of N values (source) to another list (dest).
    If the second list is not given, we assume it is equal to [0, ..., N-1]."""

    # initialise
    source = np.array(reformat_to_list(source), dtype='int32')
    n_labels = source.shape[0]

    # build new label list if neccessary
    if dest is None:
        dest = np.arange(n_labels, dtype='int32')
    else:
        assert len(source) == len(dest), 'label_list and new_label_list should have the same length'
        dest = np.array(reformat_to_list(dest, dtype='int'))

    # build look-up table
    lut = np.zeros(np.max(source) + 1, dtype='int32')
    for source, dest in zip(source, dest):
        lut[source] = dest

    return lut

def fast_3D_interp_torch(X, II, JJ, KK, mode='linear'):
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        Y = X[IIr, JJr, KKr]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]

        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx

        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy

        c = c0 * wfz + c1 * wcz

        Y = torch.zeros(II.shape, device='cpu')
        Y[ok] = c.float()

    else:
        raise Exception('mode must be linear or nearest')

    return Y

def fast_3D_interp_field_torch(X, II, JJ, KK):
    input_shape = X.shape
    if len(input_shape) > 4:
        X = X.reshape(input_shape[:3] + (-1, ))
    num_channels = X.shape[-1]

    ok = (II > 0) & (JJ > 0) & (KK > 0) & (II <= X.shape[0] - 1) & (JJ <= X.shape[1] - 1) & (KK <= X.shape[2] - 1)
    IIv = II[ok]
    JJv = JJ[ok]
    KKv = KK[ok]

    fx = torch.floor(IIv).long()
    cx = fx + 1
    cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
    wcx = IIv - fx
    wfx = 1 - wcx

    fy = torch.floor(JJv).long()
    cy = fy + 1
    cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    wcy = JJv - fy
    wfy = 1 - wcy

    fz = torch.floor(KKv).long()
    cz = fz + 1
    cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
    wcz = KKv - fz
    wfz = 1 - wcz

    Y = torch.zeros([*II.shape, num_channels], device='cpu')
    for channel in range(num_channels):

        Xc = X[:, :, :, channel]

        c000 = Xc[fx, fy, fz]
        c100 = Xc[cx, fy, fz]
        c010 = Xc[fx, cy, fz]
        c110 = Xc[cx, cy, fz]
        c001 = Xc[fx, fy, cz]
        c101 = Xc[cx, fy, cz]
        c011 = Xc[fx, cy, cz]
        c111 = Xc[cx, cy, cz]

        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx

        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy

        c = c0 * wfz + c1 * wcz

        Yc = torch.zeros(II.shape, device='cpu')
        Yc[ok] = c.float()

        Y[:, :, :, channel] = Yc

    if len(input_shape) > 4:
        Y = Y.reshape(input_shape)

    return Y


def instance_register(ref, flo, svf, inshape, epochs=5, grad_penalty=1, int_resolution=2):
    model = InstanceDense(
        inshape,
        nb_feats=1,
        int_steps=7,
        int_resolution=int_resolution
    )
    model.set_flow(svf)
    image_loss_func = NCC().loss
    #
    losses = [image_loss_func, Grad('l2', loss_mult=int_resolution).loss]
    weights = [1, grad_penalty]
    #
    zeros = np.zeros((1, *inshape, len(inshape)), dtype='float32')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=weights)
    model.fit(
        [flo],
        [ref, zeros],
        batch_size=None,
        epochs=epochs,
        steps_per_epoch=1,
        verbose=1
    )

    return model

def compute_atlas_alignment(im_file, mask_file, atlas_mri, M, mode='linear', normalize=True):
    atlas_volsize = atlas_mri.shape
    atlas_aff = atlas_mri.affine

    R, Raff, Rh = io_utils.load_volume(im_file, im_only=False, squeeze=True, dtype=None, aff_ref=None)
    Rm, Rmaff, Rmh = io_utils.load_volume(mask_file, im_only=False, squeeze=True, dtype=None, aff_ref=None)

    II, JJ, KK = np.meshgrid(np.arange(atlas_volsize[0]), np.arange(atlas_volsize[1]), np.arange(atlas_volsize[2]), indexing='ij')
    II = torch.tensor(II)
    JJ = torch.tensor(JJ)
    KK = torch.tensor(KK)

    # Reference
    affine = torch.tensor(np.matmul(np.linalg.inv(Raff), np.matmul(M, atlas_aff)))
    II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
    JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
    KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
    Rlin = fast_3D_interp_torch(torch.tensor(R.copy()), II2, JJ2, KK2, mode)

    affine = torch.tensor(np.matmul(np.linalg.inv(Rmaff), np.matmul(M, atlas_aff)))
    II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
    JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
    KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
    RSlin = fast_3D_interp_torch(torch.tensor(Rm.copy()), II2, JJ2, KK2, 'nearest')

    Rlin[RSlin == 0] = 0
    if normalize:
        Rlin = Rlin / torch.max(Rlin)

    return Rlin, Raff, Rh

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

class GaussianBlur(keras.layers.Layer):
    """Applies gaussian blur to an input image."""

    def __init__(self, sigma, random_blur_range=None, use_mask=False, **kwargs):
        self.sigma = reformat_to_list(sigma)
        assert np.all(np.array(self.sigma) >= 0), 'sigma should be superior or equal to 0'
        self.use_mask = use_mask

        self.n_dims = None
        self.n_channels = None
        self.blur_range = random_blur_range
        self.stride = None
        self.separable = None
        self.kernels = None
        self.convnd = None
        super(GaussianBlur, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["sigma"] = self.sigma
        config["random_blur_range"] = self.blur_range
        config["use_mask"] = self.use_mask
        return config

    def build(self, input_shape):

        # get shapes
        if self.use_mask:
            assert len(input_shape) == 2, 'please provide a mask as second layer input when use_mask=True'
            self.n_dims = len(input_shape[0]) - 2
            self.n_channels = input_shape[0][-1]
        else:
            self.n_dims = len(input_shape) - 2
            self.n_channels = input_shape[-1]

        # prepare blurring kernel
        self.stride = [1]*(self.n_dims+2)
        self.sigma = reformat_to_list(self.sigma, length=self.n_dims)
        self.separable = np.linalg.norm(np.array(self.sigma)) > 5
        if self.blur_range is None:  # fixed kernels
            self.kernels = gaussian_kernel(self.sigma, separable=self.separable)
        else:
            self.kernels = None

        # prepare convolution
        self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)

        self.built = True
        super(GaussianBlur, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if self.use_mask:
            image = inputs[0]
            mask = tf.cast(inputs[1], 'bool')
        else:
            image = inputs
            mask = None

        # redefine the kernels at each new step when blur_range is activated
        if self.blur_range is not None:
            self.kernels = gaussian_kernel(self.sigma, blur_range=self.blur_range, separable=self.separable)

        if self.separable:
            for k in self.kernels:
                if k is not None:
                    image = tf.concat([self.convnd(tf.expand_dims(image[..., n], -1), k, self.stride, 'SAME')
                                       for n in range(self.n_channels)], -1)
                    if self.use_mask:
                        maskb = tf.cast(mask, 'float32')
                        maskb = tf.concat([self.convnd(tf.expand_dims(maskb[..., n], -1), k, self.stride, 'SAME')
                                           for n in range(self.n_channels)], -1)
                        image = image / (maskb + keras.backend.epsilon())
                        image = tf.where(mask, image, tf.zeros_like(image))
        else:
            if any(self.sigma):
                image = tf.concat([self.convnd(tf.expand_dims(image[..., n], -1), self.kernels, self.stride, 'SAME')
                                   for n in range(self.n_channels)], -1)
                if self.use_mask:
                    maskb = tf.cast(mask, 'float32')
                    maskb = tf.concat([self.convnd(tf.expand_dims(maskb[..., n], -1), self.kernels, self.stride, 'SAME')
                                       for n in range(self.n_channels)], -1)
                    image = image / (maskb + keras.backend.epsilon())
                    image = tf.where(mask, image, tf.zeros_like(image))

        return image

class ConvertLabels(keras.layers.Layer):
    """Convert all labels in a tensor by the corresponding given set of values.
    labels_converted = ConvertLabels(source_values, dest_values)(labels).
    labels must be an int32 tensor, and labels_converted will also be int32.

    :param source_values: list of all the possible values in labels. Must be a list or a 1D numpy array.
    :param dest_values: list of all the target label values. Must be ordered the same as source values:
    labels[labels == source_values[i]] = dest_values[i].
    If None (default), dest_values is equal to [0, ..., N-1], where N is the total number of values in source_values,
    which enables to remap label maps to [0, ..., N-1].
    """

    def __init__(self, source_values, dest_values=None, **kwargs):
        self.source_values = source_values
        self.dest_values = dest_values
        self.lut = None
        super(ConvertLabels, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["source_values"] = self.source_values
        config["dest_values"] = self.dest_values
        return config

    def build(self, input_shape):
        self.lut = tf.convert_to_tensor(get_mapping_lut(self.source_values, dest=self.dest_values), dtype='int32')
        self.built = True
        super(ConvertLabels, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.gather(self.lut, tf.cast(inputs, dtype='int32'))

class Unet(tf.keras.Model):
    """
    A unet architecture that builds off either an input keras model or input shape. Layer features
    can be specified directly as a list of encoder and decoder features or as a single integer along
    with a number of unet levels. The default network features per layer (when no options are
    specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(self,
                 inshape=None,
                 input_model=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 do_res=False,
                 half_res=False,
                 hyp_input=None,
                 hyp_tensor=None,
                 final_activation_function=None,
                 name='unet'):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the unet before concatenation
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
            hyp_input: Hypernetwork input tensor. Enables HyperConvs if provided. Default is None.
            hyp_tensor: Hypernetwork final tensor. Enables HyperConvs if provided. Default is None.
            final_activation_function: if not None add an activation layer at end
            name: Model name - also used as layer name prefix. Default is 'unet'.
        """

        # have the option of specifying input shape or input model
        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            unet_input = KL.Input(shape=inshape, name='%s_input' % name)
            model_inputs = [unet_input]
        else:
            unet_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
            model_inputs = input_model.inputs

        # add hyp_input tensor if provided
        if hyp_input is not None:
            model_inputs = model_inputs + [hyp_input]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        ndims = len(unet_input.get_shape()) - 2
        assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * nb_levels

        # configure encoder (down-sampling path)
        enc_layers = []
        last = unet_input
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor)
            enc_layers.append(last)

            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

        # configure decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_dec_conv_%d_%d' % (name, real_level, conv)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor)
            if not half_res or level < (nb_levels - 2):
                layer_name = '%s_dec_upsample_%d' % (name, real_level)
                last = _upsample_block(last, enc_layers.pop(), factor=max_pool[real_level],
                                       name=layer_name)

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            layer_name = '%s_dec_final_conv_%d' % (name, num)
            last = _conv_block(last, nf, name=layer_name, hyp_tensor=hyp_tensor)

        if final_activation_function is not None:
            last = KL.Activation(final_activation_function, name='%s_final_activation' % name)(last)

        super().__init__(inputs=model_inputs, outputs=last, name=name)

class RescaleTransform(Layer):
    """
    Rescale transform layer

    Rescales a dense or affine transform. If dense, this involves resizing and
    rescaling the vector field.
    """

    def __init__(self, zoom_factor, interp_method='linear', **kwargs):
        """
        Parameters:
            zoom_factor: Scaling factor.
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'zoom_factor': self.zoom_factor,
            'interp_method': self.interp_method,
        })
        return config

    def build(self, input_shape):
        # check if transform is affine
        self.is_affine = is_affine_shape(input_shape[1:])
        self.ndims = input_shape[-1] - 1 if self.is_affine else input_shape[-1]

    def compute_output_shape(self, input_shape):
        if self.is_affine:
            return (input_shape[0], self.ndims, self.ndims + 1)
        else:
            shape = [int(d * self.zoom_factor) for d in input_shape[1:-1]]
            return (input_shape[0], *shape, self.ndims)

    def call(self, transform):
        """
        Parameters
            transform: Transform to rescale. Either a dense warp of shape [B, D1, ..., DN, N]
            or an affine matrix of shape [B, N, N+1].
        """
        if self.is_affine:
            return rescale_affine(transform, self.zoom_factor)
        else:
            return rescale_dense_transform(transform, self.zoom_factor,
                                                 interp_method=self.interp_method)

class VxmDenseOriginalSynthmorph(ne.modelio.LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 input_model=None,
                 fill_value=None,
                 name='vxm_dense'):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
            input_model: Model to replace default input layer before concatenation. Default is None.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        unet_model = Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            name='%s_unet' % name
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                         kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                         name='%s_flow' % name)(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                                 kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                                 bias_initializer=KI.Constant(value=-10),
                                 name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow_inputs = [flow_mean, flow_logsigma]
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)(flow_inputs)
        else:
            flow_params = flow_mean
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow
        if bidir:
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = VecInt(method='ss',
                              name='%s_flow_int' % name,
                              int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = VecInt(method='ss',
                                         name='%s_neg_flow_int' % name,
                                         int_steps=int_steps)(neg_flow)

            # resize to final resolution
            if int_downsize > 1:
                pos_flow = RescaleTransform(
                    int_downsize, name='%s_diffflow' % name)(pos_flow)
                if bidir:
                    neg_flow = RescaleTransform(
                        int_downsize, name='%s_neg_diffflow' % name)(neg_flow)

        # warp image with flow field
        y_source = SpatialTransformer(
            interp_method='linear',
            indexing='ij',
            fill_value=fill_value,
            name='%s_transformer' % name)([source, pos_flow])
        if bidir:
            st_inputs = [target, neg_flow]
            y_target = SpatialTransformer(interp_method='linear',
                                                 indexing='ij',
                                                 fill_value=fill_value,
                                                 name='%s_neg_transformer' % name)(st_inputs)

        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source]

        if use_probs:
            # compute loss on flow probabilities
            outputs += [flow_params]
        else:
            # compute smoothness loss on pre-integrated warp
            outputs += [preint_flow]

        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None
        self.references.preint_flow = preint_flow


    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.preint_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        st_input = [img_input, warp_model.output]
        y_img = SpatialTransformer(interp_method=interp_method)(st_input)
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class InstanceDense(ne.modelio.LoadableModel):
    """
    VoxelMorph network to perform instance-specific optimization.
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_feats=1,
                 mult=1000,
                 int_steps=7,
                 int_downsize=None,
                 int_resolution=2):
        """
        Parameters:
            inshape: Input shape of moving image. e.g. (192, 192, 192)
            nb_feats: Number of source image features. Default is 1.
            mult: Bias multiplier for local parameter layer. Default is 1000.
            int_steps: Number of flow integration steps.
                The warp is non-diffeomorphic when this value is 0.
            int_resolution: Resolution (relative voxel size) of the flow field during
                vector integration. Default is 2.
            int_downsize: Deprecated - use int_resolution instead.
        """

        # downsample warp shape
        ds_warp_shape = [int(dim / float(int_resolution)) for dim in inshape]

        source = tf.keras.Input(shape=(*inshape, nb_feats))
        flow_layer = ne.layers.LocalParamWithInput(shape=(*ds_warp_shape, len(inshape)), mult=mult)
        preint_flow = flow_layer(source)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        pos_flow = preint_flow
        if int_steps > 0:
            pos_flow = VecInt(method='ss', name='flow_int', int_steps=int_steps)(pos_flow)

            # resize to final resolution
            if int_resolution > 1:
                pos_flow = RescaleTransform(int_resolution, name='diffflow')(pos_flow)

        # warp image with flow field
        y_source = SpatialTransformer(interp_method='linear', indexing='ij',
                                      name='transformer')([source, pos_flow])

        # initialize the keras model
        super().__init__(name='vxm_instance_dense',
                         inputs=[source],
                         outputs=[y_source, preint_flow])

        # cache pointers to important layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.pos_flow = pos_flow
        self.references.flow_layer = flow_layer
        self.references.mult = mult

    def set_flow(self, warp):
        '''
        Sets the networks flow field weights. Scales the warp to
        accommodate the local weight multiplier.
        '''
        warp = warp / self.references.mult
        self.references.flow_layer.set_weights(warp)

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict(src)

class VecInt(Layer):
    """
    Vector integration layer

    Enables vector integration via several methods (ode or quadrature for
    time-dependent vector fields and scaling-and-squaring for stationary fields)

    If you find this function useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self,
                 indexing='ij',
                 method='ss',
                 int_steps=7,
                 out_time_pt=1,
                 ode_args=None,
                 odeint_fn=None,
                 int_end=1,
                 **kwargs):
        """
        Parameters:
            indexing: Must be 'xy' or 'ij'.
            method: Must be any of the methods in neuron.utils.integrate_vec.
            int_steps: Number of integration steps.
            out_time_pt: Time point at which to output if using odeint integration.
        """

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        self.out_time_pt = out_time_pt
        self.odeint_fn = odeint_fn  # if none then will use a tensorflow function
        self.ode_args = ode_args
        self.int_end = int_end
        if ode_args is None:
            self.ode_args = {'rtol': 1e-6, 'atol': 1e-12}
        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'indexing': self.indexing,
            'method': self.method,
            'int_steps': self.int_steps,
            'out_time_pt': self.out_time_pt,
            'ode_args': self.ode_args,
            'odeint_fn': self.odeint_fn,
        })
        return config

    def build(self, input_shape):
        # confirm built
        self.built = True

        trf_shape = input_shape
        if isinstance(input_shape[0], (list, tuple)):
            trf_shape = input_shape[0]
        self.inshape = trf_shape

        if trf_shape[-1] != len(trf_shape) - 2:
            raise Exception('transform ndims %d does not match expected ndims %d'
                            % (trf_shape[-1], len(trf_shape) - 2))

    def call(self, inputs, *kwargs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        loc_shift = inputs[0]

        # necessary for multi-gpu models
        loc_shift = tf.reshape(loc_shift, [-1, *self.inshape[1:]])
        if hasattr(inputs[0], '_keras_shape'):
            loc_shift._keras_shape = inputs[0]._keras_shape

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            loc_shift_split = tf.split(loc_shift, loc_shift.shape[-1], axis=-1)
            loc_shift_lst = [loc_shift_split[1], loc_shift_split[0], *loc_shift_split[2:]]
            loc_shift = tf.concat(loc_shift_lst, -1)

        if len(inputs) > 1:
            assert self.out_time_pt is None, \
                'out_time_pt should be None if providing batch_based out_time_pt'

        # map transform across batch
        # out = MapVecInt()([loc_shift, inputs[1:]])
        out = tf.map_fn(self._single_int,
                        [loc_shift] + inputs[1:],
                        fn_output_signature=loc_shift.dtype)

        if hasattr(inputs[0], '_keras_shape'):
            out._keras_shape = inputs[0]._keras_shape
        return out

    def _single_int(self, inputs, **kwargs):

        vel = inputs[0]
        out_time_pt = self.out_time_pt
        if len(inputs) == 2:
            out_time_pt = inputs[1]
        return integrate_vec(vel, method=self.method,
                             nb_steps=self.int_steps,
                             ode_args=self.ode_args,
                             out_time_pt=out_time_pt,
                             odeint_fn=self.odeint_fn,
                             int_end=self.int_end)


class MapVecInt(Layer):
    def call(self, inputs):


        return tf.map_fn(self._single_int,
                        [inputs[0]] + inputs[1],
                        fn_output_signature=inputs[0].dtype)

    def _single_int(self, inputs, **kwargs):

        vel = inputs[0]
        out_time_pt = self.out_time_pt
        if len(inputs) == 2:
            out_time_pt = inputs[1]
        return integrate_vec(vel, method=self.method,
                             nb_steps=self.int_steps,
                             ode_args=self.ode_args,
                             out_time_pt=out_time_pt,
                             odeint_fn=self.odeint_fn,
                             int_end=self.int_end)

class SpatialTransformer(Layer):
    """
    ND spatial transformer layer

    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.

    If you find this layer useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network.

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 **kwargs):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            indexing: Must be 'ij' (matrix) or 'xy' (cartesian). 'xy' indexing will
                have the first two entries of the flow (along last axis) flipped
                compared to 'ij' indexing.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms.
        """
        self.interp_method = interp_method
        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):

        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]
        self.is_affine = is_affine_shape(input_shape[1][1:])

        # make sure inputs are reasonable shapes
        if self.is_affine:
            expected = (self.ndims, self.ndims + 1)
            actual = tuple(self.trfshape[-2:])
            if expected != actual:
                raise ValueError(f'Expected {expected} affine matrix, got {actual}.')
        else:
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                import warnings
                warnings.warn(f'Dense transform shape {dense_shape} does not match '
                              f'image shape {image_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """

        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))

        # convert affine matrix to warp field
        if self.is_affine:
            fun = lambda x: affine_to_dense_shift(x, vol.shape[1:-1],
                                                        shift_center=self.shift_center,
                                                        indexing=self.indexing)
            trf = tf.map_fn(fun, trf)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]), vol)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], fn_output_signature=vol.dtype)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method,
                               fill_value=self.fill_value)

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5, signed=False):
        self.win = win
        self.eps = eps
        self.signed = signed

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        in_ch = Ji.get_shape().as_list()[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(Ii, sum_filt, strides, padding)
        J_sum = conv_fn(Ji, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # TODO: simplify this
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        if self.signed:
            cc = cross / tf.sqrt(I_var * J_var + self.eps)
        else:
            # cc = (cross * cross) / (I_var * J_var)
            cc = (cross / I_var) * (cross / J_var)

        return cc

    def loss(self, y_true, y_pred, reduce='mean'):
        # compute cc
        cc = self.ncc(y_true, y_pred)
        # reduce
        if reduce == 'mean':
            cc = tf.reduce_mean(K.batch_flatten(cc), axis=-1)
        elif reduce == 'max':
            cc = tf.reduce_max(K.batch_flatten(cc), axis=-1)
        elif reduce is not None:
            raise ValueError(f'Unknown NCC reduction type: {reduce}')
        # loss
        return -cc

class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad

