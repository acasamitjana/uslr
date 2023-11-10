import pdb
from os.path import join
import copy

import nibabel as nib
import numpy as np

from setup import *
from utils import io_utils, synthmorph_utils, def_utils

# -------------- #
# Initialization #
# -------------- #

def initialize_graph_linear(pairwise_centroids, affine_filepath, pairwise_timepoints=None, ok_centr=None):
    # https://www.cse.sc.edu/~songwang/CourseProj/proj2004/ross/ross.pdf

    refCent, floCent = pairwise_centroids

    if ok_centr is not None:
        refCent = refCent[:, ok_centr > 0]
        floCent = floCent[:, ok_centr > 0]

    trans_ref = np.mean(refCent, axis=1, keepdims=True)
    trans_flo = np.mean(floCent, axis=1, keepdims=True)

    refCent_tx = refCent - trans_ref
    floCent_tx = floCent - trans_flo

    cov = refCent_tx @ floCent_tx.T
    u, s, vt = np.linalg.svd(cov)
    I = np.eye(3)
    if np.prod(np.diag(s)) < 0:
        I[-1, -1] = -1

    Q = vt.T @ I @ u.T

    # Full transformation

    Tr = np.eye(4)
    Tr[:3, 3] = -trans_ref.squeeze()

    Tf = np.eye(4)
    Tf[:3, 3] = trans_flo.squeeze()

    R = np.eye(4)
    R[:3,:3] = Q

    aff = Tf @ R @ Tr

    np.save(affine_filepath, aff)

    if pairwise_timepoints is not None:
        ref_proxy, flo_proxy = pairwise_timepoints
        data = np.array(flo_proxy.dataobj)
        v2r = np.linalg.inv(aff) @ flo_proxy.affine
        proxy_reg = nib.Nifti1Image(data, v2r)
        proxy_reg = def_utils.vol_resample(ref_proxy, proxy_reg, mode='nearest')
        nib.save(proxy_reg, affine_filepath + '.nii.gz')

def initialize_graph_nonlinear(pairwise_timepoints, Msubject, results_dir, filename,
                               instance_refinement=True, epochs=10, grad_penalty=1, full_size=False):
    from tensorflow.python import keras
    import tensorflow as tf
    tp_ref, tp_flo = pairwise_timepoints

    if isinstance(tp_ref, dict) and isinstance(tp_flo, dict):
        ref_filepath = tp_ref['image']
        flo_filepath = tp_flo['image']
        ref_mask_filepath = tp_ref['mask']
        flo_mask_filepath = tp_flo['mask']

    else:
        conditions_image_res = {'suffix': 'T1w', 'acquisition': '1', 'scope': 'sreg-lin', 'space': 'SUBJECT',
                                'extension': '.nii.gz', 'run': '01'}
        conditions_mask_res = {'suffix': 'mask', 'acquisition': '1', 'scope': 'sreg-lin', 'space': 'SUBJECT',
                               'extension': '.nii.gz', 'run': '01'}

        ref_filepath = join(tp_ref.data_dir['sreg-lin'], tp_ref.get_files(**conditions_image_res)[0])
        flo_filepath = join(tp_flo.data_dir['sreg-lin'], tp_flo.get_files(**conditions_image_res)[0])

        ref_mask_filepath = join(tp_ref.data_dir['sreg-lin'], tp_ref.get_files(**conditions_mask_res)[0])
        flo_mask_filepath = join(tp_flo.data_dir['sreg-lin'], tp_flo.get_files(**conditions_mask_res)[0])

    A, Aaff, Ah = io_utils.load_volume(synthmorph_utils.atlas_file, im_only=False, squeeze=True, dtype=None, aff_ref=None)
    Amri = nib.Nifti1Image(A, Aaff)
    Aaff = Aaff.astype('float32')

    SVFaff_net = Aaff.copy()
    for c in range(3):
        SVFaff_net[:-1, c] = SVFaff_net[:-1, c] * 2

    SVFaff_net[:-1, -1] = SVFaff_net[:-1, -1] - np.matmul(SVFaff_net[:-1, :-1], 0.5 * (np.array([0.5, 0.5, 0.5]) - 1))

    Rlin, Raff, Rh = synthmorph_utils.compute_atlas_alignment(ref_filepath, ref_mask_filepath, Amri, Msubject)
    Flin, Faff, Fh = synthmorph_utils.compute_atlas_alignment(flo_filepath, flo_mask_filepath, Amri, Msubject)


    cnn = synthmorph_utils.VxmDenseOriginalSynthmorph.load(synthmorph_utils.path_model_registration)
    svf1 = cnn.register(Flin.detach().numpy()[np.newaxis, ..., np.newaxis],
                        Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])

    svf2 = cnn.register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                        Flin.detach().numpy()[np.newaxis, ..., np.newaxis])
    svf = 0.5 * svf1 - 0.5 * svf2

    if instance_refinement and grad_penalty > 0:
        instance_model = synthmorph_utils.instance_register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                                            Flin.detach().numpy()[np.newaxis, ..., np.newaxis], svf,
                                                            inshape=A.shape, epochs=epochs, grad_penalty=grad_penalty)

        svf = instance_model.references.flow_layer(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])
        svf = svf.numpy()

    # pdb.set_trace()
    if full_size:
        upscaler = keras.Sequential([synthmorph_utils.RescaleTransform(2)])
        svf = upscaler(tf.convert_to_tensor(svf))
        SVFaff_net = Aaff

    SVFmri_net = nib.Nifti1Image(np.squeeze(svf), Msubject @ SVFaff_net)
    nib.save(SVFmri_net, join(results_dir, filename + '.svf.nii.gz'))

    if DEBUG:
        integrator = keras.Sequential([synthmorph_utils.VecInt(method='ss', int_steps=7)])
        upscaler = keras.Sequential([synthmorph_utils.RescaleTransform(2)])

        warp_pos_small = integrator(tf.convert_to_tensor(svf))
        f2r_field = np.squeeze(upscaler(warp_pos_small))
        flow_mri = nib.Nifti1Image(f2r_field, (Msubject @ Aaff).astype('float32'))

        refproxy = nib.load(ref_filepath)
        floproxy = nib.load(flo_filepath)

        t1w_reg = def_utils.vol_resample(refproxy, floproxy, proxyflow=flow_mri)
        nib.save(t1w_reg, join(results_dir, filename + '.reg.nii.gz'))


def initialize_graph_nonlinear_multimodal(pairwise_modality, Msubject, results_dir, filename, instance_refinement=True,
                                          epochs=10, grad_penalty=1, full_size=False, int_resolution=2):
    from tensorflow.python import keras
    import tensorflow as tf
    mod_ref, mod_flo = pairwise_modality

    ref_filepath = mod_ref['image']
    flo_filepath = mod_flo['image']
    ref_mask_filepath = mod_ref['mask']
    flo_mask_filepath = mod_flo['mask']

    A, Aaff, Ah = io_utils.load_volume(synthmorph_utils.atlas_file, im_only=False, squeeze=True, dtype=None, aff_ref=None)
    Amri = nib.Nifti1Image(A, Aaff)
    Aaff = Aaff.astype('float32')

    SVFaff_net = Aaff.copy()
    for c in range(3):
        SVFaff_net[:-1, c] = SVFaff_net[:-1, c] * 2

    SVFaff_net[:-1, -1] = SVFaff_net[:-1, -1] - np.matmul(SVFaff_net[:-1, :-1], 0.5 * (np.array([0.5, 0.5, 0.5]) - 1))

    Rlin, Raff, Rh = synthmorph_utils.compute_atlas_alignment(ref_filepath, ref_mask_filepath, Amri, Msubject)
    Flin, Faff, Fh = synthmorph_utils.compute_atlas_alignment(flo_filepath, flo_mask_filepath, Amri, Msubject)

    cnn = synthmorph_utils.VxmDenseOriginalSynthmorph.load(synthmorph_utils.path_model_registration)
    svf1 = cnn.register(Flin.detach().numpy()[np.newaxis, ..., np.newaxis],
                        Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])

    svf2 = cnn.register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                        Flin.detach().numpy()[np.newaxis, ..., np.newaxis])
    svf = 0.5 * svf1 - 0.5 * svf2

    if instance_refinement and grad_penalty > 0:
        if int_resolution != 2:
            svf = synthmorph_utils.RescaleTransform(2/int_resolution, name='diffflow_down')(svf)
        instance_model = synthmorph_utils.instance_register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                                            Flin.detach().numpy()[np.newaxis, ..., np.newaxis], svf,
                                                            inshape=A.shape, epochs=epochs, grad_penalty=grad_penalty,
                                                            int_resolution=int_resolution)

        svf = instance_model.references.flow_layer(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])
        svf = svf.numpy()

    # pdb.set_trace()
    if full_size:
        upscaler = keras.Sequential([synthmorph_utils.RescaleTransform(2)])
        svf = upscaler(tf.convert_to_tensor(svf))
        SVFaff_net = Aaff

    SVFmri_net = nib.Nifti1Image(np.squeeze(svf), Msubject @ SVFaff_net)
    nib.save(SVFmri_net, join(results_dir, filename + '.svf.nii.gz'))

    if DEBUG:
        integrator = keras.Sequential([synthmorph_utils.VecInt(method='ss', int_steps=7)])
        upscaler = keras.Sequential([synthmorph_utils.RescaleTransform(2)])

        warp_pos_small = integrator(tf.convert_to_tensor(svf))
        f2r_field = np.squeeze(upscaler(warp_pos_small))
        flow_mri = nib.Nifti1Image(f2r_field, (Msubject @ Aaff).astype('float32'))

        refproxy = nib.load(ref_filepath)
        floproxy = nib.load(flo_filepath)

        t1w_reg = def_utils.vol_resample(refproxy, floproxy, proxyflow=flow_mri)
        nib.save(t1w_reg, join(results_dir, filename + '.reg.nii.gz'))


def create_template_space(linear_image_list):

    boundaries_min = np.zeros((len(linear_image_list), 3))
    boundaries_max = np.zeros((len(linear_image_list), 3))
    margin_bb = 5
    for it_lil, lil in enumerate(linear_image_list):

        if isinstance(lil, nib.nifti1.Nifti1Image):
            proxy = lil
        else:
            proxy = nib.load(lil)
        mask = np.asarray(proxy.dataobj)
        header = proxy.affine
        idx = np.where(mask > 0)
        vox_min = np.concatenate((np.min(idx, axis=1), [1]), axis=0)
        vox_max = np.concatenate((np.max(idx, axis=1), [1]), axis=0)

        minR, minA, minS = np.inf, np.inf, np.inf
        maxR, maxA, maxS = -np.inf, -np.inf, -np.inf

        for i in [vox_min[0], vox_max[0] + 1]:
            for j in [vox_min[1], vox_max[1] + 1]:
                for k in [vox_min[2], vox_max[2] + 1]:
                    aux = np.dot(header, np.asarray([i, j, k, 1]).T)

                    minR, maxR = min(minR, aux[0]), max(maxR, aux[0])
                    minA, maxA = min(minA, aux[1]), max(maxA, aux[1])
                    minS, maxS = min(minS, aux[2]), max(maxS, aux[2])

        minR -= margin_bb
        minA -= margin_bb
        minS -= margin_bb

        maxR += margin_bb
        maxA += margin_bb
        maxS += margin_bb

        boundaries_min[it_lil] = [minR, minA, minS]
        boundaries_max[it_lil] = [maxR, maxA, maxS]
        # boundaries_min += [[minR, minA, minS]]
        # boundaries_max += [[maxR, maxA, maxS]]

    # Get the corners of cuboid in RAS space
    minR = np.mean(boundaries_min[..., 0])
    minA = np.mean(boundaries_min[..., 1])
    minS = np.mean(boundaries_min[..., 2])
    maxR = np.mean(boundaries_max[..., 0])
    maxA = np.mean(boundaries_max[..., 1])
    maxS = np.mean(boundaries_max[..., 2])

    template_size = np.asarray(
        [int(np.ceil(maxR - minR)) + 1, int(np.ceil(maxA - minA)) + 1, int(np.ceil(maxS - minS)) + 1])

    # Define header and size
    template_vox2ras0 = np.asarray([[1, 0, 0, minR],
                                    [0, 1, 0, minA],
                                    [0, 0, 1, minS],
                                    [0, 0, 0, 1]])


    # VOX Mosaic
    II, JJ, KK = np.meshgrid(np.arange(0, template_size[0]),
                             np.arange(0, template_size[1]),
                             np.arange(0, template_size[2]), indexing='ij')

    RR = II + minR
    AA = JJ + minA
    SS = KK + minS
    rasMosaic = np.concatenate((RR.reshape(-1, 1),
                                AA.reshape(-1, 1),
                                SS.reshape(-1, 1),
                                np.ones((np.prod(template_size), 1))), axis=1).T

    return rasMosaic, template_vox2ras0,  tuple(template_size)

