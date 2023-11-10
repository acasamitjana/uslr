import csv
import pdb

import nibabel as nib
import numpy as np
from scipy.special import softmax
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.interpolate import RegularGridInterpolator as rgi
from munkres import Munkres


def align_with_identity_vox2ras0(V, vox2ras0):

    COST = np.zeros((3,3))
    for i in range(3):
        for j in range(3):

            # worker is the vector
            b = vox2ras0[:3,i]

            # task is j:th axis
            a = np.zeros((3,1))
            a[j] = 1

            COST[i, j] = - np.abs(np.dot(a.T, b))/np.linalg.norm(a, 2)/np.linalg.norm(b, 2)

    m = Munkres()
    indexes = m.compute(COST)

    v2r = np.zeros_like(vox2ras0)
    for idx in indexes:
        v2r[:, idx[1]] = vox2ras0[:, idx[0]]
    v2r[:, 3] = vox2ras0[:, 3]
    V = np.transpose(V, axes=[idx[1] for idx in indexes])

    for d in range(3):
        if v2r[d,d] < 0:
            v2r[:3, d] = -v2r[:3, d]
            v2r[:3, 3] = v2r[:3, 3] - v2r[:3, d] * (V.shape[d] -1)
            V = np.flip(V, axis=d)

    return V, v2r

def rescale_volume(volume, new_min=0, new_max=255, min_percentile=2, max_percentile=98, use_positive_only=True):
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
    robust_max = np.max(intensities) if max_percentile == 0 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * new_max
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)

def rescale_flow(flow_vol, aff, new_vox_size):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """
    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    f_factor = pixdim / new_vox_size
    pdb.set_trace()

    flow_vol[..., 0] *= f_factor[0]
    flow_vol[..., 1] *= f_factor[1]
    flow_vol[..., 2] *= f_factor[2]

    flow_vol, flow_aff = rescale_voxel_size(flow_vol, aff, new_vox_size, not_aliasing=True)

    return flow_vol, flow_aff

def rescale_voxel_size(volume, aff, new_vox_size, not_aliasing=False):
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

    if len(volume.shape) > 3:
        sigmas = np.concatenate((sigmas, [0]))

    if all(sigmas == 0) or not_aliasing:
        volume_filt = volume
    else:
        volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape[:3] * factor)

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

def gaussian_antialiasing(volume, aff, new_voxel_size):
    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_voxel_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    return gaussian_filter(volume, sigmas)


def get_rigid_params(matrix, proxyref, cog=None):
    ry = -np.asin(matrix[2, 0])
    rx = np.atan2(matrix[2, 1] / np.cos(ry), matrix[2, 2] / np.cos(ry))
    rz = np.atan2(matrix[1, 0] / np.cos(ry), matrix[0, 0] / np.cos(ry))
    angles = np.array([rx, ry, rz])

    T_center = np.zeros((4, 4)).to(matrix.device)
    T_center[0, 0] = 1
    T_center[1, 1] = 1
    T_center[2, 2] = 1
    T_center[3, 3] = 1
    if cog is None:
        T_center[:3, 3] = (-proxyref.affine @ np.asarray([i / 2 for i in proxyref.shape] + [1]))[:3]
    else:
        T_center[:3, 3] = -cog


    T_center_inv = np.zeros((4, 4)).to(matrix.device)
    T_center_inv[0, 0] = 1
    T_center_inv[1, 1] = 1
    T_center_inv[2, 2] = 1
    T_center_inv[3, 3] = 1
    if cog is None:
        T_center_inv[:3, 3] = (proxyref.affine @ np.asarray([i / 2 for i in proxyref.shape] + [1]))[:3]
    else:
        T_center_inv[:3, 3] = cog

    T_rot = np.eye(4)
    T_rot[:3, :3] = matrix[:3, :3]
    T_trans = matrix @ T_center_inv @ np.linalg.inv(T_rot) @ T_center
    translation = T_trans[:3, 3]

    return angles, translation
def one_hot_encoding(target, num_classes=None, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or dict): existing categories as a LUT. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (d1, d2, ..., dN, num_classes)

    '''

    if categories is None and num_classes is None:
        categories = {cls: it_cls for it_cls, cls in enumerate(np.sort(np.unique(target)))}
        num_classes = len(categories)

    elif categories is not None:
        if isinstance(categories, list) or isinstance(categories, np.ndarray):
            categories = {cls: it_cls for it_cls, cls in enumerate(categories)}

        num_classes = len(np.unique(list(categories.values())))

    else:
        categories = {cls: cls for cls in np.arange(num_classes)}

    labels = np.zeros((num_classes,) + target.shape, dtype='uint16')
    for cls, it_cls in categories.items():
        idx_class = np.where(target == cls)
        idx = (it_cls,) + idx_class
        labels[idx] = 1

    return np.transpose(labels, axes=(1,2,3,0))

def label_log_odds(target, num_classes=None, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or list): existing categories. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (num_classes, d1, d2, ..., dN)

    '''

    if categories is None and num_classes is None:
        categories = np.sort(np.unique(target))
        num_classes = len(categories)

    elif categories is not None:
        num_classes = len(categories)

    else:
        categories = np.arange(num_classes)

    labels = -10000 * np.ones((num_classes,) + target.shape, dtype='int')
    for it_cls, cls in enumerate(categories):
        mask_label = target == cls
        bbox_label, crop_coord = crop_label(mask_label, margin=10)

        d_in = (distance_transform_edt(bbox_label))
        d_out = -distance_transform_edt(~bbox_label)
        d = np.zeros_like(d_in)
        d[bbox_label] = d_in[bbox_label]
        d[~bbox_label] = d_out[~bbox_label]

        labels[it_cls, crop_coord[0][0]: crop_coord[0][1], crop_coord[1][0]: crop_coord[1][1], crop_coord[2][0]: crop_coord[2][1]] = d

    return labels

def crop_label(mask, margin=10, threshold=0):

    ndim = len(mask.shape)
    if isinstance(margin, int):
        margin=[margin]*ndim

    crop_coord = []
    idx = np.where(mask>threshold)
    for it_index, index in enumerate(idx):
        clow = max(0, np.min(idx[it_index]) - margin[it_index])
        chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
        crop_coord.append([clow, chigh])

    mask_cropped = mask[
                   crop_coord[0][0]: crop_coord[0][1],
                   crop_coord[1][0]: crop_coord[1][1],
                   crop_coord[2][0]: crop_coord[2][1]
                   ]

    return mask_cropped, crop_coord

def apply_crop(image, crop_coord):
    return image[crop_coord[0][0]: crop_coord[0][1],
                 crop_coord[1][0]: crop_coord[1][1],
                 crop_coord[2][0]: crop_coord[2][1]
           ]

def compute_centroids_ras(seg_file, labelfile):
    seg_proxy = nib.load(seg_file)
    seg_buffer = np.array(seg_proxy.dataobj)
    labels = np.load(labelfile)

    nlab = len(labels)
    refCOG = np.zeros([4, nlab])

    ok = np.ones(nlab)
    for l in range(nlab):
        aux = np.where(seg_buffer == labels[l])
        if len(aux[0]) > 50:
            refCOG[0, l] = np.median(aux[0])
            refCOG[1, l] = np.median(aux[1])
            refCOG[2, l] = np.median(aux[2])
            refCOG[3, l] = 1
        else:
            ok[l] = 0

    refCOG = np.matmul(seg_proxy.affine, refCOG)[:-1, :]

    return refCOG, ok


def compute_distance_map_nongrid(labelmap, sampling_grid, labels_lut=None):
    if labels_lut is None:
        labels_lut = {ul: it_ul for it_ul, ul in enumerate(np.unique(labelmap))}

    sampling_grid_nn = np.round(sampling_grid).astype('int')
    sampling_grid_nn[0] = np.clip(sampling_grid_nn[0], 0, labelmap.shape[0] - 1)
    sampling_grid_nn[1] = np.clip(sampling_grid_nn[1], 0, labelmap.shape[1] - 1)
    sampling_grid_nn[2] = np.clip(sampling_grid_nn[2], 0, labelmap.shape[2] - 1)
    distancemap = -200 * np.ones(sampling_grid_nn.shape[1:] + (len(labels_lut.keys()),), dtype='float32')
    for ul, it_ul in labels_lut.items():

        mask_label = labelmap == ul
        mask_label_reg = mask_label[sampling_grid_nn[0], sampling_grid_nn[1], sampling_grid_nn[2]]
        if np.sum(mask_label) == 0:
            continue
        else:

            idx_in = distance_transform_edt(mask_label, return_distances=False, return_indices=True)
            d_in = np.sqrt(np.sum((idx_in[:, sampling_grid_nn[0], sampling_grid_nn[1], sampling_grid_nn[2]] - sampling_grid)**2, axis=0))
            idx_out = distance_transform_edt(~mask_label,  return_distances=False, return_indices=True)
            d_out = -np.sqrt(np.sum((idx_out[:, sampling_grid_nn[0], sampling_grid_nn[1], sampling_grid_nn[2]] - sampling_grid)**2, axis=0))

            # With crop (it is approximate near the boundaries)
            # bbox_label, crop_coord = crop_label(mask_label, margin=5)
            # idx_in = distance_transform_edt(bbox_label, return_distances=False, return_indices=True)
            # d_in = np.sqrt(np.sum((idx_in[:,
            #                        np.clip(sampling_grid_nn[0]-crop_coord[0][0], 0, bbox_label.shape[0]-1),
            #                        np.clip(sampling_grid_nn[1]-crop_coord[1][0], 0, bbox_label.shape[1]-1),
            #                        np.clip(sampling_grid_nn[2]-crop_coord[2][0], 0, bbox_label.shape[2]-1)]
            #                        - sampling_grid + np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0]]).reshape((3, 1, 1, 1))) ** 2, axis=0))
            # idx_out = distance_transform_edt(~bbox_label,  return_distances=False, return_indices=True)
            # d_out = - np.sqrt(np.sum((idx_out[:,
            #                           np.clip(sampling_grid_nn[0]-crop_coord[0][0], 0, bbox_label.shape[0]-1),
            #                           np.clip(sampling_grid_nn[1]-crop_coord[1][0], 0, bbox_label.shape[1]-1),
            #                           np.clip(sampling_grid_nn[2]-crop_coord[2][0], 0, bbox_label.shape[2]-1)]
            #                           - sampling_grid + np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0]]).reshape((3, 1, 1, 1))) ** 2, axis=0))

            d = np.zeros_like(d_in)
            d[mask_label_reg] = d_in[mask_label_reg]
            d[~mask_label_reg] = d_out[~mask_label_reg]
            distancemap[... , it_ul] = d

    return distancemap



def compute_distance_map(labelmap, soft_seg=True, labels_lut=None):
    if labels_lut is None:
        labels_lut = {ul: it_ul for it_ul, ul in enumerate(np.unique(labelmap))}

    distancemap = -200 * np.ones(labelmap.shape + (len(labels_lut.keys()),), dtype='float32')
    for ul, it_ul in labels_lut.items():

        mask_label = labelmap == ul
        if np.sum(mask_label) == 0:
            continue
        else:
            mask_label, crop_coord = crop_label(mask_label, margin=5)

            d_in = (distance_transform_edt(mask_label))
            d_out = -distance_transform_edt(~mask_label)
            d = np.zeros_like(d_in)
            d[mask_label] = d_in[mask_label]
            d[~mask_label] = d_out[~mask_label]

            distancemap[
                crop_coord[0][0]: crop_coord[0][1],
                crop_coord[1][0]: crop_coord[1][1],
                crop_coord[2][0]: crop_coord[2][1], it_ul
            ] = d

    # distancemap = np.clip(distancemap, -3, np.max(distancemap))

    if soft_seg:
        distancemap = softmax(distancemap, axis=-1)

    return distancemap



def compute_distance_map_crop(labelmap, soft_seg=True, labels_lut=None):
    if labels_lut is None:
        labels_lut = {ul: it_ul for it_ul, ul in enumerate(np.unique(labelmap))}

    distancemap = -200 * np.ones(labelmap.shape + (len(labels_lut.keys()),), dtype='float32')
    for ul, it_ul in labels_lut.items():

        mask_label = labelmap == ul
        if np.sum(mask_label) == 0:
            continue
        else:
            bbox_label, crop_coord = crop_label(mask_label, margin=5)

            d_in = (distance_transform_edt(bbox_label))
            d_out = -distance_transform_edt(~bbox_label)
            d = np.zeros_like(d_in)
            d[bbox_label] = d_in[bbox_label]
            d[~bbox_label] = d_out[~bbox_label]

            distancemap[crop_coord[0][0]: crop_coord[0][1],
                        crop_coord[1][0]: crop_coord[1][1],
                        crop_coord[2][0]: crop_coord[2][1], it_ul] = d

    if soft_seg:
        distancemap = softmax(distancemap, axis=-1)

    return distancemap