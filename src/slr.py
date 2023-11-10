import copy
import os.path
import pdb
from os.path import join, exists, dirname
from os import listdir, makedirs
import itertools
from datetime import date, datetime
import time
import shutil
import subprocess
from joblib import delayed, Parallel

import json
import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.optimize import linprog
import nibabel as nib
from skimage.transform import resize
from skimage.morphology import ball

from setup import *
from src.callbacks import History, ModelCheckpoint, PrinterCallback
from src.models import InstanceRigidModelLOG
from utils.io_utils import write_affine_matrix, build_bids_fileame, get_bids_fileparts, read_affine_matrix
from utils.slr_utils import create_template_space
from utils.def_utils import vol_resample
from utils.fn_utils import one_hot_encoding, get_rigid_params
from utils.labels import APARC_ARR, ASEG_ARR

# Read linear st2 graph
# Formulas extracted from: https://math.stackexchange.com/questions/3031999/proof-of-logarithm-map-formulae-from-so3-to-mathfrak-so3
def init_st2_lineal(timepoints, input_dir, eps=1e-6, extension='aff'):
    nk = 0

    N = len(timepoints)
    K = int(N * (N - 1) / 2)

    phi_log = np.zeros((6, K))

    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
        if not isinstance(tp_ref, str):
            tid_ref, tid_flo = tp_ref.id, tp_flo.id
        else:
            tid_ref, tid_flo = tp_ref, tp_flo


        filename = str(tid_ref) + '_to_' + str(tid_flo)

        if exists(join(input_dir, filename + '.aff')):
            rotation_matrix, translation_vector = read_affine_matrix(join(input_dir, filename + '.aff'))
        else:
            rigid_matrix = np.load(join(input_dir, filename + '.aff.npy'))
            rotation_matrix, translation_vector = rigid_matrix[:3, :3], rigid_matrix[:3, 3]

        # Log(R) and Log(T)
        t_norm = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1 + eps, 1 - eps)) + eps
        W = 1 / (2 * np.sin(t_norm)) * (rotation_matrix - rotation_matrix.T) * t_norm
        Vinv = np.eye(3) - 0.5 * W + ((1 - (t_norm * np.cos(t_norm / 2)) / ( 2 * np.sin(t_norm / 2))) / t_norm ** 2) * W * W  # np.matmul(W, W)

        phi_log[0, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * t_norm
        phi_log[1, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * t_norm
        phi_log[2, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * t_norm

        phi_log[3:, nk] = np.matmul(Vinv, translation_vector)

        nk += 1

    return phi_log


# Read st2 graph
def init_st2(timepoints, input_dir, image_shape, factor=1, scope='slr-lin', se=None, penalty=1):


    timepoints_dict = {
       t.id: it_t for it_t, t in enumerate(timepoints)
    } # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)

    nk = 0

    N = len(timepoints)
    K = int(N*(N-1)/2) +1

    w = np.zeros((K, N), dtype='int')
    obs_mask = np.zeros(image_shape + (K,))

    phi = np.zeros((3,) + image_shape + (K, ))
    for sl_ref, sl_flo in itertools.combinations(timepoints, 2):
        sid_ref, sid_flo = sl_ref.id, sl_flo.id

        t0 = timepoints_dict[sl_ref.id]
        t1 = timepoints_dict[sl_flo.id]
        filename = str(sid_ref) + '_to_' + str(sid_flo)

        svf_proxy = nib.load(join(input_dir, filename + '.svf.nii.gz'))
        field = np.asarray(svf_proxy.dataobj)

        if field.shape[0] != 3: field = np.transpose(field, axes=(3, 0, 1, 2))

        phi[..., nk] = factor*field

        # Masks
        if se is not None:
            cond_mask = {'suffix': 'mask', 'acquisition': '1', 'scope': scope, 'space': 'SUBJECT', 'extension': '.nii.gz', 'run': '01'}

            mask_proxy = nib.load(join(sl_ref.data_dir[scope], sl_ref.get_files(**cond_mask)[1]))
            mask_proxy = vol_resample(svf_proxy, mask_proxy)

            mask = np.asarray(mask_proxy.dataobj) > 0
            mask = binary_dilation(mask, se)

        else:
            mask = np.ones(image_shape)


        obs_mask[..., nk] = mask
        w[nk, t0] = -1
        w[nk, t1] = 1
        nk += 1

    obs_mask[..., nk] = (np.sum(obs_mask[..., :nk-1]) > 0).astype('uint8')
    w[nk, :] = penalty
    nk += 1
    return phi, obs_mask, w, nk

# Read st2 graph
def init_st2_chunks(timepoints, input_dir, image_shape, chunk, cond_mask = None, se = None):

    timepoints_dict = {
       t.id: it_t for it_t, t in enumerate(timepoints)
    } # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)

    nk = 0

    N = len(timepoints)
    K = int(N*(N-1)/2) +1
    chunk_size = (chunk[0][1] - chunk[0][0], chunk[1][1] - chunk[1][0], chunk[2][1] - chunk[2][0])

    w = np.zeros((K, N), dtype='int')
    obs_mask = np.zeros(chunk_size + (K,))

    phi = np.zeros((3,) + chunk_size + (K, ))
    for sl_ref, sl_flo in itertools.combinations(timepoints, 2):

        sid_ref, sid_flo = sl_ref.id, sl_flo.id
        t0 = timepoints_dict[sl_ref.id]
        t1 = timepoints_dict[sl_flo.id]
        filename = str(sid_ref) + '_to_' + str(sid_flo)

        if not exists(join(input_dir, filename + '.svf.nii.gz')):
            sid_ref, sid_flo = sl_flo.id, sl_ref.id
            t0 = timepoints_dict[sl_ref.id]
            t1 = timepoints_dict[sl_flo.id]
            filename = str(sid_ref) + '_to_' + str(sid_flo)

        proxy = nib.load(join(input_dir, filename + '.svf.nii.gz'))
        field = np.asarray(proxy.dataobj)
        if field.shape[:3] != image_shape:
            phi[0, ..., nk] = resize(field[..., 0], image_shape)[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1]]
            phi[1, ..., nk] = resize(field[..., 1], image_shape)[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1]]
            phi[2, ..., nk] = resize(field[..., 2], image_shape)[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1]]
        else:
            phi[0, ..., nk] = field[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1],0]
            phi[1, ..., nk] = field[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1],1]
            phi[2, ..., nk] = field[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1],2]

        # Masks
        if cond_mask is None:
            cond_mask = {'suffix': 'mask', 'acquisition': '1', 'scope': 'sreg-lin', 'space': 'SUBJECT',
                         'desc': 'resampled', 'extension': '.nii.gz', 'run': '01'}

        proxy = nib.load(join(sl_ref.data_dir['sreg-lin'], sl_ref.get_files(**cond_mask)[0]))
        mask_ref = (np.asarray(proxy.dataobj) > 0).astype('uint8')
        mask = mask_ref #*mask_mov
        mask = (resize(np.double(mask), image_shape, anti_aliasing=True) > 0).astype('uint8')
        if se is not None:
            mask = binary_dilation(mask, se)

        obs_mask[..., nk] = mask[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1]]
        w[nk, t0] = -1
        w[nk, t1] = 1
        nk += 1

    obs_mask[..., nk] = (np.sum(obs_mask[..., :nk-1]) > 0).astype('uint8')
    w[nk, :] = 1
    nk += 1
    return phi, obs_mask, w, nk

# Optimization of rigid transforms
def st_linear(subject, cost, lr, max_iter, n_epochs, force_flag=False, verbose=False):
    print('Subject: ' + str(subject.id))

    timepoints = subject.sessions
    results_dir_sbj = subject.data_dir['sreg-lin']
    # if not exists(results_dir_sbj):
    #     if len(listdir(subject.data_dir['bids'])) <= 1:
    #         shutil.rmtree(subject.data_dir['synthseg'])
    #         shutil.rmtree(subject.data_dir['bids'])
    #
    #     return

    date_start = date.today().strftime("%d/%m/%Y")
    time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    exp_dict = {
        'date': date_start,
        'time': time_start,
        'cost': cost,
        'lr': lr,
        'max_iter': max_iter,
        'n_epochs': n_epochs
    }

    fp_dict = {'sub': subject.id, 'desc': 'linTemplate'}
    filename_template = build_bids_fileame(fp_dict)
    if not exists(join(results_dir_sbj, filename_template + '_T1w.json')):
        json_object = json.dumps(exp_dict, indent=4)
        with open(join(results_dir_sbj, filename_template + '_T1w.json'), "w") as outfile:
            outfile.write(json_object)

    # if exists(join(results_dir_sbj, filename_template + '.json')): os.remove(join(results_dir_sbj, filename_template + '.json'))


    linear_template = join(subject.data_dir['sreg-lin'], filename_template + '_T1w.nii.gz')
    linear_template_mask = join(subject.data_dir['sreg-lin'], filename_template + '_mask.nii.gz')
    linear_template_seg = join(subject.data_dir['sreg-lin'], filename_template + '_dseg.nii.gz')
    conditions_image = {'space': 'orig', 'acquisition': 'orig',  'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
    conditions_image_res = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
    conditions_mask = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'mask', 'scope': 'synthseg'}
    conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'dseg', 'scope': 'synthseg'}

    if len(subject.sessions) == 0:
        print('Skipping. No sessions found.')
        return
    elif len(subject.sessions) == 1:
        filename_res = subject.sessions[0].get_files(**conditions_image_res)[0]
        filename_mask = subject.sessions[0].get_files(**conditions_mask)[0]

        if not exists(join(timepoints[0].data_dir['sreg-lin'], filename_res)):
            subprocess.call(['ln', '-s', join(timepoints[0].data_dir['synthseg'], filename_res), join(timepoints[0].data_dir['sreg-lin'], filename_res)])
        if not exists(join(timepoints[0].data_dir['sreg-lin'], filename_mask)):
            subprocess.call(['ln', '-s', join(timepoints[0].data_dir['synthseg'], filename_mask), join(timepoints[0].data_dir['sreg-lin'], filename_mask)])
        if not exists(join(timepoints[0].data_dir['sreg-lin'], linear_template)):
            subprocess.call(['ln', '-s', join(timepoints[0].data_dir['synthseg'], filename_res),  linear_template])
        print('It has only 1 timepoint. No registration is made.')
        return

    # Deformations dir
    deformations_dir = join(subject.data_dir['sreg-lin'], 'deformations')
    if not exists(join(deformations_dir, timepoints[-2].id + '_to_' + timepoints[-1].id + '.aff')):
        if verbose: print('!!! WARNING -- No observations found for subject ' + subject.id + ' and NiftyReg.')
        return

    # Check if multiple runs in this dataset.
    aff_dict = {'sub': subject.id, 'desc': 'aff'}
    if not all([len(tp.get_files(extension='npy', desc='aff', scope='sreg-lin')) > 0 for tp in timepoints]) or force_flag:

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#

        if verbose: print('[' + str(subject.id) + ' - Building the graph] Reading transforms ...')
        t_init = time.time()

        graph_structure = init_st2_lineal(timepoints, deformations_dir)
        R_log = graph_structure

        if verbose: print('[' + str(subject.id) + ' - Building the graph] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

        if verbose: print('[' + str(subject.id) + ' - SLR] Computimg the graph ...')
        t_init = time.time()

        Tres = st2_lineal_pytorch(R_log, timepoints, n_epochs, cost, lr, results_dir_sbj, max_iter=max_iter, verbose=False)

        if verbose: print('[' + str(subject.id) + ' - SLR] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#

        if verbose: print('[' + str(subject.id) + ' - Integration] Computing the latent rigid transform ... ')
        t_init = time.time()
        for it_tp, tp in enumerate(timepoints):
            # run_flag = any(['run' in f for f in tp.files['bids']])
            # if run_flag:
            #     filename = build_bids_fileame({**{'ses': tp.id}, **aff_dict, **{'run': '01'}})
            # else:
            filename = build_bids_fileame({**{'ses': tp.id}, **aff_dict})

            affine_matrix = Tres[..., it_tp]

            np.save(join(tp.data_dir['sreg-lin'], filename + '.npy'), affine_matrix)
            write_affine_matrix(join(tp.data_dir['sreg-lin'], filename + '.aff'), affine_matrix)

        if verbose: print('[' + str(subject.id) + ' - Integration] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#


    if not exists(linear_template) or force_flag:

        t_init = time.time()
        if verbose: print('[' + str(subject.id) + ' - Deforming] Updating vox2ras0  ... ')

        headers = {}
        headers_orig = {}
        linear_mask_list = {}
        for it_tp, tp in enumerate(timepoints):
            filename_res = tp.get_files(**conditions_image_res)[0]

            fp_dict = get_bids_fileparts(filename_res)
            fp_dict['space'] = 'SUBJECT'
            fp_dict['acq'] = '1'
            filename_res = build_bids_fileame(fp_dict) + '.nii.gz'

            fp_dict['suffix'] = 'mask'
            filename_mask = build_bids_fileame(fp_dict) + '.nii.gz'

            affine_matrix = np.load(join(tp.data_dir['sreg-lin'], build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'))

            # Compute image orig header
            im_orig_mri = tp.get_image(**conditions_image)
            if im_orig_mri is None:
                conditions_image.pop('acquisition')
                im_orig_mri = tp.get_image(**conditions_image)
            v2r_sbj_orig = np.linalg.inv(affine_matrix) @ im_orig_mri.affine

            # Update image res header
            im_res_mri = tp.get_image(**conditions_image_res)
            v2r_sbj_res = np.linalg.inv(affine_matrix) @ im_res_mri.affine #np.matmul(np.linalg.inv(affine_matrix), proxy.affine)
            # img = nib.Nifti1Image(np.array(im_res_mri.dataobj), v2r_sbj_res)
            # nib.save(img, join(tp.data_dir['sreg-lin'], filename_res))
            #
            #
            # # Update mask res header
            mask = np.array(tp.get_image(**conditions_mask).dataobj)
            mask_dilated = binary_dilation(mask, ball(3)).astype('uint8')
            img = nib.Nifti1Image(mask_dilated, v2r_sbj_res)
            # nib.save(img, join(tp.data_dir['sreg-lin'], filename_mask))


            linear_mask_list[tp.id] = img
            headers_orig[tp.id] = v2r_sbj_orig
            headers[tp.id] = v2r_sbj_res

        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #

        if verbose: print('[' + str(subject.id) + ' - Deforming] Creating subject space  ... ')

        rasMosaic, template_vox2ras0, template_size = create_template_space(list(linear_mask_list.values()))
        proxytemplate = nib.Nifti1Image(np.zeros(template_size), template_vox2ras0)

        if verbose: print('[' + str(subject.id) + ' - Deforming] Computing linear template ... ')
        mri_list = []
        mask_list = []
        aparc_aseg = np.concatenate((APARC_ARR, ASEG_ARR), axis=0)
        seg_list = np.zeros(template_size + (len(aparc_aseg),))
        for it_tp, tp in enumerate(timepoints):
            filename = tp.get_files(**conditions_image)[0]
            fp_dict = get_bids_fileparts(filename)
            fp_dict['space'] = 'SUBJECT'
            fp_dict['acq'] = '1'
            filename = build_bids_fileame(fp_dict) + '.nii.gz'
            fp_dict['suffix'] = 'mask'
            filename_mask = build_bids_fileame(fp_dict) + '.nii.gz'

            im_orig_mri = tp.get_image(**conditions_image)
            pixdim = np.sqrt(np.sum(im_orig_mri.affine * im_orig_mri.affine, axis=0))[:-1]
            new_vox_size = np.array([1, 1, 1])
            factor = pixdim / new_vox_size
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            volume_filt = gaussian_filter(np.array(im_orig_mri.dataobj), sigmas)
            im_orig_mri = nib.Nifti1Image(volume_filt, headers_orig[tp.id])
            im_mri = vol_resample(proxytemplate, im_orig_mri)
            nib.save(im_mri, join(tp.data_dir['sreg-lin'], filename))
            mri_list.append(np.array(im_mri.dataobj))

            mask = np.array(tp.get_image(**conditions_mask).dataobj)
            proxyflo = nib.Nifti1Image(mask, headers[tp.id])
            im_mri = vol_resample(proxytemplate, proxyflo)
            nib.save(im_mri, join(tp.data_dir['sreg-lin'], filename_mask))
            mask_list.append(np.array(im_mri.dataobj))

            seg = np.array(tp.get_image(**conditions_seg).dataobj)
            proxyflo = nib.Nifti1Image(seg, headers[tp.id])
            im_mri = vol_resample(proxytemplate, proxyflo, mode='nearest')
            seg_list += one_hot_encoding(np.array(im_mri.dataobj), categories=aparc_aseg)

        template = np.median(mri_list, axis=0)
        template = template.astype('uint8')
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, linear_template)

        template = np.sum(mask_list, axis=0)/len(mask_list) > 0.5
        template = template.astype('uint8')
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, linear_template_mask)

        seg_hard = np.argmax(seg_list, axis=-1)
        template = np.zeros(template_size, dtype='int16')
        for it_l, l in enumerate(aparc_aseg): template[seg_hard == it_l] = l
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, linear_template_seg)

        if verbose: print('[' + str(subject.id) + ' - Deforming] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

def st_linear_bids(bids_loader, subject, cost, lr, max_iter, n_epochs, slr_lin_dir, force_flag=False, verbose=False):
    print('Subject: ' + str(subject))

    timepoints = bids_loader.get_session(subject=subject)

    timepoints = list(filter(lambda x: len(bids_loader.get(extension='nii.gz', subject=subject, session=x, suffix='T1w')) > 0, timepoints))
    # timepoints = list(filter(lambda x: not exists(join(dirname(bids_loader.get(scope='synthseg',subject=subject, session=x, return_type='filename')[0]), 'excluded_file.txt')), timepoints))


    dir_results = join(slr_lin_dir, 'sub-' + subject)
    if not exists(dir_results): makedirs(dir_results)

    date_start = date.today().strftime("%d/%m/%Y")
    time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    exp_dict = {
        'date': date_start,
        'time': time_start,
        'cost': cost,
        'lr': lr,
        'max_iter': max_iter,
        'n_epochs': n_epochs
    }

    filename_template = 'sub-' + subject + '_desc-linTemplate_anat'
    if not exists(join(dir_results, filename_template + '_T1w.json')):
        json_object = json.dumps(exp_dict, indent=4)
        with open(join(dir_results, filename_template + '_T1w.json'), "w") as outfile:
            outfile.write(json_object)

    deformations_dir = join(dir_results, 'deformations')
    linear_template = join(dir_results, filename_template + '.nii.gz')

    seg_files = bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, suffix='T1wdseg')
    if len(seg_files) <= 1:
        print('  It has only 0/1 segmentation file available. No registration is made.')
        return

    if len(timepoints) == 0:
        print('Skipping. No sessions found.')
        return

    elif len(timepoints) == 1:
        ent_res = {k: str(v) for k, v in seg_files[0].entities.items() if k in ['subject', 'session', 'run', 'acquisition', 'suffix', 'extension', 'task', 'tracer']}
        im_mask = bids_loader.get(scope='synthseg', **ent_res)
        if len(im_mask) != 1:
            print('  Search mask command failed. Number of matched files: ' + str(len(im_mask)))
        else:
            im_mask = im_mask[0]

        ent_res['suffix'] = ent_res['suffix'].replace('mask', '')
        im_res = bids_loader.get(scope='synthseg', **ent_res)
        if len(im_res) != 1:
            print('  Search image command failed. Number of matched files: ' + str(len(im_res)))
        else:
            im_res = im_res[0]

        if not exists(join(dir_results, im_res.filename)):
            subprocess.call(['ln', '-s', im_res.path, join(dir_results, im_res.filename)])

        if not exists(join(dir_results, im_mask.filename)):
            subprocess.call(['ln', '-s', im_mask.path, join(dir_results, im_mask.filename)])

        if not exists(linear_template):
            subprocess.call(['ln', '-s', join(dir_results, im_res.filename), linear_template])

        print('   It has only 1 modality. No registration is made.')
        return

    # Deformations dir
    extension = 'aff'
    if not exists(join(deformations_dir, timepoints[-2] + '_to_' + timepoints[-1] + '.aff')) and \
            not exists(join(deformations_dir, timepoints[-2] + '_to_' + timepoints[-1] + '.aff.npy')):

        if verbose: print('!!! WARNING -- No observations found for subject ' + subject + ' and Procrustes Analysis.')
        return

    # Check if multiple runs in this dataset.
    aff_dict = {'subject': subject, 'desc': 'aff', 'suffix': 'T1w'}
    if not len(bids_loader.get(subject=subject,  extension='npy', desc='aff', scope='slr-lin')) == len(seg_files) or force_flag:

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#

        if verbose: print('[' + str(subject) + ' - Building the graph] Reading transforms ...')
        t_init = time.time()

        graph_structure = init_st2_lineal(timepoints, deformations_dir, extension=extension)
        R_log = graph_structure

        if verbose: print('[' + str(subject) + ' - Building the graph] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

        if verbose: print('[' + str(subject) + ' - SLR] Computimg the graph ...')
        t_init = time.time()

        Tres = st2_lineal_pytorch(R_log, timepoints, n_epochs, cost, lr, dir_results, max_iter=max_iter, verbose=False)

        if verbose: print('[' + str(subject) + ' - SLR] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#

        if verbose: print('[' + str(subject) + ' - Integration] Computing the latent rigid transform ... ')
        t_init = time.time()
        for it_tp, tp in enumerate(timepoints):

            extra_kwargs = {'session': tp}
            filename_npy = bids_loader.build_path({**extra_kwargs, **aff_dict, 'extension': 'npy'}, scope='sreg-lin',
                                                  path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                  absolute_paths=False)
            filename_aff = bids_loader.build_path({**extra_kwargs, **aff_dict, 'extension': 'txt'}, scope='sreg-lin',
                                                  path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                  absolute_paths=False)

            affine_matrix = Tres[..., it_tp]

            dir_results_sess = join(os.path.dirname(dir_results), os.path.dirname(filename_npy))
            if not exists(dir_results_sess): makedirs(dir_results_sess)
            np.save(join(dir_results_sess, os.path.basename(filename_npy)), affine_matrix)
            write_affine_matrix(join(dir_results_sess, os.path.basename(filename_aff)), affine_matrix)

        if verbose: print('[' + str(subject) + ' - Integration] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

    else:
        print('   Subject already processed.')

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#

def st_linear_multimodal(bids_loader, subject, cost, lr, max_iter, n_epochs, force_flag=False, verbose=False):

    print('Subject: ' + str(subject))
    smr_lin_dir = os.path.join(os.path.dirname(bids_loader.root), 'derivatives', 'smr-lin')

    timepoints = bids_loader.get_session(subject=subject)

    dir_results_sbj = join(smr_lin_dir, 'sub-' + subject)

    date_start = date.today().strftime("%d/%m/%Y")
    time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    exp_dict = {
        'date': date_start,
        'time': time_start,
        'cost': cost,
        'lr': lr,
        'max_iter': max_iter,
        'n_epochs': n_epochs
    }

    filename_template = 'sub-' + subject + '_desc-linTemplate_anat'
    if not exists(join(dir_results_sbj, filename_template + '_anat.json')):
        json_object = json.dumps(exp_dict, indent=4)
        with open(join(dir_results_sbj, filename_template + '_anat.json'), "w") as outfile:
            outfile.write(json_object)

    seg_dict = {'subject': subject, 'scope': 'synthseg', 'extension': 'nii.gz'}
    suffix_seg_list = [s for s in bids_loader.get(**{**seg_dict, 'return_type': 'id', 'target': 'suffix'}) if 'dseg' in s]
    for tp_id in timepoints:
        print('* Computing T=' + str(tp_id))
        # if tp_id not in ['m162']: continue

        dir_results_sess = join(dir_results_sbj, 'ses-' + tp_id)
        deformations_dir = join(dir_results_sess, 'deformations')
        linear_template = join(dir_results_sess, 'anat', filename_template + '.nii.gz')

        seg_files = bids_loader.get(**{'suffix': suffix_seg_list, 'session': tp_id, **seg_dict})
        seg_files = list(filter(lambda x: 'run' not in x.entities.keys() or 'run-01' in x.filename, seg_files))
        if len(seg_files) == 1:
            print('  It has only 1 modality. No registration is made.')
            continue

        modality_list = []
        for seg_file in seg_files:
            modality = seg_file.entities['suffix'].split('dseg')[0]
            if 'run' in seg_file.entities.keys():
                modality += '.' + str(seg_file.entities['run'])

            modality_list.append(modality)

        if len(modality_list) == 0:
            print('  Skipping. No modalities found.')
            continue

        elif len(modality_list) == 1 and seg_files[0].entities['dataype'] == 'anat':
            modality = seg_files[0].entities['suffix'].split('dseg')[0]
            ent_im_res = copy.deepcopy(seg_files[0].entities)
            ent_mask_res = copy.deepcopy(seg_files[0].entities)
            ent_im_res['suffix'] = modality
            ent_mask_res['suffix'] = modality + 'mask'
            ent_res = {k: str(v) for k, v in seg_files[0].entities.items() if k in filename_entities}
            if 'run' in modality:
                suffix, run = modality.split('.')
                ent_res['suffix'] = suffix
                ent_res['run'] = run

            else:
                ent_res['run'] = modality

            im_res = bids_loader.get(scope='synthseg', **ent_im_res)[0]
            im_mask = bids_loader.get(scope='synthseg', **ent_mask_res)[0]

            if not exists(join(dir_results_sess, 'anat', im_res.filename)):
                subprocess.call(['ln', '-s', im_res.path, join(dir_results_sess, 'anat', im_res.filename)])

            if not exists(join(dir_results_sess, 'anat', im_mask.filename)):
                subprocess.call(['ln', '-s', im_mask.path, join(dir_results_sess, 'anat', im_mask.filename)])

            if not exists(linear_template):
                subprocess.call(['ln', '-s', join(dir_results_sess, 'anat', im_res.filename), linear_template])

            print('   It has only 1 modality. No registration is made.')
            return

        elif not exists(join(deformations_dir, modality_list[-2] + '_to_' + modality_list[-1] + '.aff')):
            print('   !!! WARNING -- No observations found for subject ' + subject + '.')
            continue


        # Check if multiple runs in this dataset.
        aff_dict = {'subject': subject, 'desc': 'aff', 'session': tp_id}
        if not len(bids_loader.get(subject=subject, session=tp_id, extension='npy', desc='aff', scope='smr-lin')) == len(seg_files) or force_flag:

            # -------------------------------------------------------------------#
            # -------------------------------------------------------------------#

            if verbose: print('   - [Building the graph] Reading transforms ...')

            t_init = time.time()

            graph_structure = init_st2_lineal(modality_list, deformations_dir)
            R_log = graph_structure

            if verbose: print('   - [Building the graph] Total Elapsed time: ' + str(time.time() - t_init) + '\n')
            if verbose: print('   - [SLR] Computimg the graph ...')

            t_init = time.time()

            Tres = st2_lineal_pytorch(R_log, modality_list, n_epochs, cost, lr, dir_results_sbj, max_iter=max_iter, verbose=False)

            if verbose: print('   - [SLR] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

            # -------------------------------------------------------------------#
            # -------------------------------------------------------------------#

            if verbose: print('   - [Integration] Computing the latent rigid transform ... ')
            t_init = time.time()
            for it_mod, mod in enumerate(modality_list):

                if '.' in mod:
                    suffix, run = mod.split('.')
                    extra_kwargs = {'suffix': suffix, 'run': run}
                else:
                    extra_kwargs = {'suffix': mod}

                filename_npy = bids_loader.build_path({**extra_kwargs, **aff_dict, 'extension': 'npy'}, scope='smr-lin', path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False, absolute_paths=False)
                filename_aff = bids_loader.build_path({**extra_kwargs, **aff_dict,  'extension': 'txt'}, scope='smr-lin', path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False, absolute_paths=False)
                if not exists(join(smr_lin_dir, dirname(filename_aff))): makedirs(join(smr_lin_dir, dirname(filename_aff)))
                affine_matrix = Tres[..., it_mod]

                np.save(join(smr_lin_dir, filename_npy), affine_matrix)
                write_affine_matrix(join(smr_lin_dir, filename_aff), affine_matrix)

            if verbose: print('   - [Integration] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

            # -------------------------------------------------------------------#
            # -------------------------------------------------------------------#


        # if not exists(linear_template) or force_flag:
        #
        #     t_init = time.time()
        #     if verbose: print('   - [Deforming] Updating vox2ras0  ... ')
        #
        #     headers = {}
        #     headers_orig = {}
        #     linear_mask_list = {}
        #     for it_mod, mod in enumerate(modality_list):
        #         if '.' in mod:
        #             suffix, run = mod.split('.')
        #             extra_kwargs = {'run': run}
        #
        #         else:
        #             suffix = mod
        #             extra_kwargs = {}
        #
        #         affine_file = bids_loader.get(**{**aff_dict, **extra_kwargs, 'suffix': suffix, 'extension': 'npy', 'scope': 'sreg-lin'})
        #         if len(affine_file) != 1:
        #             print('Wrong affine file entities')
        #             pdb.set_trace()
        #             continue
        #
        #         affine_matrix = np.load(affine_file[0])
        #         # affine_matrix = np.load(join(dir_results_sess, filename_aff + '.npy'))
        #
        #         # Update image header
        #         im_file = [im for im in bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, session=tp_id, regex_search=True, suffix=suffix, **extra_kwargs) if 'acq-orig' in im.filename]# or 'acq' not in im.filename]
        #         if len(im_file) != 1:
        #             print('     !! WARNING: More than one image is found in the synthseg directory ' + str([m.filename for m in im_file]))
        #             continue
        #         else:
        #             im_file = im_file[0]
        #
        #         ent_im = {k: str(v) for k, v in im_file.entities.items() if k in ['subject', 'session', 'space', 'run', 'acquisition', 'suffix', 'extension', 'task', 'tracer']}
        #         ent_im['space'] = 'SUBJECT'
        #         if 'acquisition' not in ent_im.keys(): ent_im['acquisition'] = 'orig'
        #
        #         im_filename = bids_loader.build_path(ent_im, scope='sreg-lin', path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False, absolute_paths=False)
        #
        #         # im_filename = build_bids_fileame(ent_im)
        #
        #         proxyim = nib.load(im_file.path)
        #         v2r_sbj_orig = np.linalg.inv(affine_matrix) @ proxyim.affine
        #         headers_orig[mod] = v2r_sbj_orig
        #
        #         img = nib.Nifti1Image(np.array(proxyim.dataobj), v2r_sbj_orig)
        #         nib.save(img, join(SREG_LIN_DIR, im_filename))
        #
        #         # Update mask res header
        #         mask_file = bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, session=tp_id, suffix=suffix+'mask', regex_search=True, **extra_kwargs)
        #         if len(mask_file) != 1:
        #             print('     !! WARNING: More than one mask is found in the synthseg directory ' + str([m.filename for m in mask_file]))
        #             continue
        #         else:
        #             mask_file = mask_file[0]
        #
        #         proxymask = nib.load(mask_file.path)
        #         mask = np.array(proxymask.dataobj)
        #         mask_dilated = binary_dilation(mask, ball(3)).astype('uint8')
        #         img = nib.Nifti1Image(mask_dilated, np.linalg.inv(affine_matrix) @ proxymask.affine)
        #         linear_mask_list[mod] = img
        #         headers[mod] = np.linalg.inv(affine_matrix) @ proxymask.affine
        #
        #     # ------------------------------------------------------------------- #
        #     # ------------------------------------------------------------------- #
        #
        #     if verbose: print('   - [Deforming] Creating subject space  ... ')
        #
        #     rasMosaic, template_vox2ras0, template_size = create_template_space(list(linear_mask_list.values()))
        #     proxytemplate = nib.Nifti1Image(np.zeros(template_size), template_vox2ras0)
        #
        #     if verbose: print('   - [Deforming] Computing linear template ... ')
        #     mri_list = []
        #     mask_list = []
        #     aparc_aseg = np.concatenate((APARC_ARR, ASEG_ARR), axis=0)
        #     seg_list = np.zeros(template_size + (len(aparc_aseg),))
        #     for it_mod, mod in enumerate(modality_list):
        #
        #         if '.' in mod:
        #             suffix, run = mod.split('.')
        #             extra_kwargs = {'run': run}
        #
        #         else:
        #             suffix = mod
        #             extra_kwargs = {}
        #
        #         if suffix not in anat_modalities: continue
        #
        #         im_file = [im for im in bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, suffix=suffix, session=tp_id, regex_search=True, **extra_kwargs) if 'acq-orig' in im.filename or 'acq' not in im.filename]
        #
        #         if len(im_file) != 1:
        #             print('     !! WARNING: More than one image is found in the synthseg directory ' + str([m.filename for m in im_file]))
        #             continue
        #         else:
        #             im_file = im_file[0]
        #
        #         ent_im = {k: str(v) for k, v in im_file.entities.items() if k in ['subject', 'session', 'space', 'run', 'acquisition', 'suffix', 'extension', 'task',  'tracer']}
        #         ent_im['space'] = 'SUBJECT'
        #         ent_im['acq'] = '1'
        #         im_filename = build_bids_fileame(ent_im)
        #
        #         proxyim = nib.load(im_file.path)
        #         pixdim = np.sqrt(np.sum(proxyim.affine * proxyim.affine, axis=0))[:-1]
        #         new_vox_size = np.array([1, 1, 1])
        #         factor = pixdim / new_vox_size
        #         sigmas = 0.25 / factor
        #         sigmas[factor > 1] = 0  # don't blur if upsampling
        #
        #         im_orig_array = np.array(proxyim.dataobj)
        #         if len(im_orig_array.shape) > 3:
        #             im_orig_array = im_orig_array[..., 0]
        #         volume_filt = gaussian_filter(im_orig_array, sigmas)
        #
        #         im_orig_mri = nib.Nifti1Image(volume_filt, headers_orig[mod])
        #         im_mri = vol_resample(proxytemplate, im_orig_mri)
        #         # nib.save(im_mri, join(dir_results_sess, im_filename))
        #         mri_list.append(np.array(im_mri.dataobj))
        #
        #         mask_file = bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, session=tp_id, suffix=suffix + 'mask', regex_search=True, **extra_kwargs)
        #         if len(mask_file) != 1:
        #             pdb.set_trace()
        #             print('     !! WARNING: More than one mask is found in the synthseg directory ' + str([m.filename for m in mask_file]))
        #             continue
        #         else:
        #             mask_file = mask_file[0]
        #
        #         mask = np.array(nib.load(mask_file.path).dataobj)
        #         proxyflo = nib.Nifti1Image(mask, headers[mod])
        #         im_mri = vol_resample(proxytemplate, proxyflo)
        #         mask_list.append(np.array(im_mri.dataobj))
        #
        #         seg_file = bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, session=tp_id,
        #                                    suffix=suffix + 'dseg', regex_search=True, **extra_kwargs)
        #         if len(seg_file) != 1:
        #             pdb.set_trace()
        #             print('     !! WARNING: More than one segmentation is found in the synthseg directory ' + str([m.filename for m in seg_file]))
        #             continue
        #         else:
        #             seg_file = seg_file[0]
        #
        #         seg = np.array(nib.load(seg_file.path).dataobj)
        #         proxyflo = nib.Nifti1Image(seg, headers[mod])
        #         im_mri = vol_resample(proxytemplate, proxyflo, mode='nearest')
        #         seg_list += one_hot_encoding(np.array(im_mri.dataobj), categories=aparc_aseg)
        #
        #     template = np.median(mri_list, axis=0)
        #     template = template.astype('uint8')
        #     img = nib.Nifti1Image(template, template_vox2ras0)
        #     nib.save(img, linear_template)
        #
        #     template = np.sum(mask_list, axis=0)/len(mask_list) > 0.5
        #     template = template.astype('uint8')
        #     img = nib.Nifti1Image(template, template_vox2ras0)
        #     nib.save(img, linear_template_mask)
        #
        #     seg_hard = np.argmax(seg_list, axis=-1)
        #     template = np.zeros(template_size, dtype='int16')
        #     for it_l, l in enumerate(aparc_aseg): template[seg_hard == it_l] = l
        #     img = nib.Nifti1Image(template, template_vox2ras0)
        #     nib.save(img, linear_template_seg)
        #
        #     if verbose: print('   - [Deforming] Total Elapsed time: ' + str(time.time() - t_init) + '\n')


# Optimization of rigid transforms using pytorch
def st2_lineal_pytorch(logR, timepoints, n_epochs, cost, lr, results_dir_sbj, max_iter=5, patience=3,
                       device='cpu', verbose=True):

    if len(timepoints) > 2:
        log_keys = ['loss', 'time_duration (s)']
        logger = History(log_keys)
        model_checkpoint = ModelCheckpoint(join(results_dir_sbj, 'checkpoints'), -1)
        callbacks = [logger, model_checkpoint]
        if verbose: callbacks += [PrinterCallback()]

        model = InstanceRigidModelLOG(timepoints, cost=cost, device=device, reg_weight=0)
        optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

        min_loss = 1000
        iter_break = 0
        log_dict = {}
        logR = torch.FloatTensor(logR)
        for cb in callbacks:
            cb.on_train_init(model)

        for epoch in range(n_epochs):
            for cb in callbacks:
                cb.on_epoch_init(model, epoch)

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                loss = model(logR, timepoints)
                loss.backward()

                return loss

            optimizer.step(closure=closure)

            loss = model(logR, timepoints)

            if loss < min_loss + 1e-4:
                iter_break = 0
                min_loss = loss.item()

            else:
                iter_break += 1

            if iter_break > patience or loss.item() == 0.:
                break

            log_dict['loss'] = loss.item()

            for cb in callbacks:
                cb.on_step_fi(log_dict, model, epoch, iteration=1, N=1)

        T = model._compute_matrix().cpu().detach().numpy()

    else:
        logR = np.squeeze(logR.astype('float32'))
        model = InstanceRigidModelLOG(timepoints, cost=cost, device=device, reg_weight=0)
        model.angle = nn.Parameter(torch.tensor(np.array([[-logR[0]/2, logR[0]/2],
                                                          [-logR[1]/2, logR[1]/2],
                                                          [-logR[2]/2, logR[2]/2]])).float(), requires_grad=False)

        model.translation = nn.Parameter(torch.tensor(np.array([[-logR[3] / 2, logR[3] / 2],
                                                                [-logR[4] / 2, logR[4] / 2],
                                                                [-logR[5] / 2, logR[5] / 2]])).float(), requires_grad=False)
        T = model._compute_matrix().cpu().detach().numpy()

    return T


# Gaussian: l2-loss no masks
def st2_L2_global(phi, W, N):
    precision = 1e-6
    lambda_control = np.linalg.inv((W.T @ W) + precision * np.eye(N)) @ W.T
    Tres = lambda_control @ np.transpose(phi,[1, 2, 3, 4, 0])
    Tres = np.transpose(Tres, [4, 0, 1, 2, 3])

    return Tres

# Gaussian: l2-loss
def st2_L2(phi, obs_mask, w, N):

    image_shape = obs_mask.shape[:-1]

    #Initialize transforms
    Tres = np.zeros(phi.shape[:4] + (N,))

    print('    Computing weights and updating the transforms')
    precision = 1e-6
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0:
            print('       Row ' + str(it_control_row) + '/' + str(image_shape[0]))

        for it_control_col in range(image_shape[1]):
            # if np.mod(it_control_col, 10) == 0:
            #     print('           Col ' + str(it_control_col) + '/' + str(image_shape[1]))

            for it_control_depth in range(image_shape[2]):

                index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]
                if index_obs.shape[0] == 0:
                    Tres[:, it_control_row, it_control_col, it_control_depth] = 0
                else:
                    w_control = w[index_obs]
                    phi_control = phi[:,it_control_row, it_control_col, it_control_depth, index_obs]
                    lambda_control = np.linalg.inv(w_control.T @ (w_control + precision*np.eye(N))) @ w_control.T

                    for it_tf in range(N):
                        Tres[0, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[0].T
                        Tres[1, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[1].T
                        Tres[2, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[2].T

    return Tres


# Laplacian: l1-loss
def st2_L1(phi, obs_mask, w, N, chunk_id=None):

    if chunk_id is not None:
        print("Processing chunk " + str(chunk_id))

    image_shape = obs_mask.shape[:3]
    Tres = np.zeros(phi.shape[:4] + (N,))
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0 and chunk_id is None:
            print('    Row ' + str(it_control_row) + '/' + str(image_shape[0]))
        for it_control_col in range(image_shape[1]):
            for it_control_depth in range(image_shape[2]):
                index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]

                if index_obs.shape[0] > 0:
                    w_control = w[index_obs]
                    phi_control = phi[:, it_control_row, it_control_col, it_control_depth, index_obs]
                    n_control = len(index_obs)

                    for it_dim in range(3):
                        # Set objective
                        c_lp = np.concatenate((np.ones((n_control,)), np.zeros((N,))), axis=0)

                        # Set the inequality
                        A_lp = np.zeros((2 * n_control, n_control + N))
                        A_lp[:n_control, :n_control] = -np.eye(n_control)
                        A_lp[:n_control, n_control:] = -w_control
                        A_lp[n_control:, :n_control] = -np.eye(n_control)
                        A_lp[n_control:, n_control:] = w_control
                        # A_lp = sp.csr_matrix(A_lp)

                        reg = np.reshape(phi_control[it_dim], (n_control,))
                        b_lp = np.concatenate((-reg, reg), axis=0)

                        result = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=(None, None), method='highs-ds')
                        Tres[it_dim, it_control_row, it_control_col, it_control_depth] = result.x[n_control:]

    return Tres

def st2_L1_gurobi(phi, obs_mask, w, N, chunk_id=None):
    import gurobipy as gp
    if chunk_id is not None:
        print("Processing chunk " + str(chunk_id))

    image_shape = obs_mask.shape[:3]
    Tres = np.zeros(phi.shape[:4] + (N,))
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0 and chunk_id is None:
            print('    Row ' + str(it_control_row) + '/' + str(image_shape[0]))
        for it_control_col in range(image_shape[1]):
            for it_control_depth in range(image_shape[2]):
                index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]

                if index_obs.shape[0] > 0:
                    w_control = w[index_obs]
                    phi_control = phi[:, it_control_row, it_control_col, it_control_depth, index_obs]
                    n_control = len(index_obs)

                    for it_dim in range(3):
                        # Set objective
                        # c_lp = np.concatenate((np.ones((n_control,)), np.zeros((N,))), axis=0)

                        model = gp.Model('LP')
                        model.setParam('OutputFlag', False)
                        model.setParam('Method', 1)

                        # Set the parameters
                        params = model.addMVar(shape=n_control + N, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='x')

                        # Set objective
                        c_lp = np.concatenate((np.ones((n_control,)), np.zeros((N,))), axis=0)
                        model.setObjective(c_lp @ params, gp.GRB.MINIMIZE)

                        # Set the inequality
                        A_lp = np.zeros((2 * n_control, n_control + N))
                        A_lp[:n_control, :n_control] = -np.eye(n_control)
                        A_lp[:n_control, n_control:] = -w_control
                        A_lp[n_control:, :n_control] = -np.eye(n_control)
                        A_lp[n_control:, n_control:] = w_control
                        A_lp = sp.csr_matrix(A_lp)

                        reg = np.reshape(phi_control[it_dim], (n_control,))
                        b_lp = np.concatenate((-reg, reg), axis=0)

                        model.addConstr(A_lp @ params <= b_lp, name="c")

                        model.optimize()

                        Tres[it_dim, it_control_row, it_control_col, it_control_depth] = params.x[n_control:]

    return Tres

def st2_L1_chunks(phi, obs_mask, w, N, num_cores=4, solver='l1'):
    chunk_list = []
    nchunks = 2
    image_shape = obs_mask.shape[:3]
    chunk_size = [int(np.ceil(cs / nchunks)) for cs in image_shape]
    for x in range(nchunks):
        for y in range(nchunks):
            for z in range(nchunks):
                max_x = min((x + 1) * chunk_size[0], image_shape[0])
                max_y = min((y + 1) * chunk_size[1], image_shape[1])
                max_z = min((z + 1) * chunk_size[2], image_shape[2])
                chunk_list += [[[x * chunk_size[0], max_x],
                                [y * chunk_size[1], max_y],
                                [z * chunk_size[2], max_z]]]

    if num_cores == 1:
        Tres = st2_L1(phi, obs_mask, w, N)

    else:
        if solver == 'gurobi':
            results = Parallel(n_jobs=num_cores)(
                delayed(st2_L1_gurobi)(phi[:, chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]],
                                       obs_mask[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]],
                                       w, N, chunk_id=it_chunk) for it_chunk, chunk in enumerate(chunk_list))

        else:
            results = Parallel(n_jobs=num_cores)(
                delayed(st2_L1)(
                    phi[:, chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]],
                    obs_mask[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]],
                    w, N, chunk_id=it_chunk) for it_chunk, chunk in enumerate(chunk_list))

        Tres = np.zeros(phi.shape[:4] + (N,))
        for it_chunk, chunk in enumerate(chunk_list):
            Tres[:, chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]] = results[it_chunk]

    return Tres