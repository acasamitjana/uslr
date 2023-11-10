import os
import pdb
import shutil
from os.path import exists, join
import time
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
import torch


# project imports
from utils.io_utils import build_bids_fileame
from setup import *
from src import bids_loader, slr
from utils import labels, slr_utils, def_utils, io_utils, synthmorph_utils, fn_utils


def deform_tp(subject, tp, seg_flag=False, channel_chunk=20):

    print(tp.id)

    # conditions_image = {'suffix': 'T1w', 'acquisition': '1', 'scope': 'sreg-lin', 'space': 'SUBJECT', 'extension': '.nii.gz', 'run': '01'}
    conditions_image = {'space': 'orig', 'acquisition': 'orig', 'suffix': 'T1w', 'scope': 'synthseg', 'run': '01'}
    aff_dict = {'sub': subject.id, 'desc': 'aff'}
    svf_dict = {'sub': subject.id, 'suffix': 'svf'}
    im_dict = {'sub': subject.id, 'ses': tp.id, 'space': 'SUBJECT', 'suffix': 'T1w'}

    proxyref = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)

    A, Aaff, Ah = io_utils.load_volume(synthmorph_utils.atlas_file, im_only=False, squeeze=True, dtype=None, aff_ref=None)
    Aaff = Aaff.astype('float32')

    Msbj = np.load(join(subject.data_dir['sreg-synthmorph'], 'sub-' + subject.id + '_desc-atlas_aff.npy')).astype('float32')
    affine_matrix = np.load(join(tp.data_dir['sreg-lin'], build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'))
    proxysvf = nib.load(join(tp.data_dir[scope], build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'))
    proxyimage = tp.get_image(**conditions_image)
    if not proxyimage:
        conditions_image.pop('run')
        proxyimage = tp.get_image(**conditions_image)

    v2r_mri = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine) #
    pixdim = np.sqrt(np.sum(proxyimage.affine * proxyimage.affine, axis=0))[:-1]
    new_vox_size = np.array([1, 1, 1])
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling
    volume_filt = gaussian_filter(np.array(proxyimage.dataobj), sigmas)


    proxyflow = def_utils.integrate_svf(proxysvf)
    flow_v2r = proxyflow.affine#Msbj @ Aaff
    f2r_field = np.array(proxyflow.dataobj)
    pixdim_ref = np.sqrt(np.sum(subject.vox2ras0 * subject.vox2ras0, axis=0))[:-1]
    pixdim_flow = np.sqrt(np.sum(flow_v2r * flow_v2r, axis=0))[:-1]
    pixdim_flo = np.sqrt(np.sum(v2r_mri * v2r_mri, axis=0))[:-1]
    pixdim_mix = np.sqrt(np.abs(np.sum((np.linalg.inv(subject.vox2ras0) @ flow_v2r) ** 2, axis=0)))[:-1]
    factor = pixdim_mix#pixdim_ref/pixdim_flow#1 / pixdim_mix#
    print('[def_utils.py] Rescaling flow: ' + str(pixdim_mix) + '(' + str(pixdim_ref) + ',' + str(pixdim_flow) + ',' + str(pixdim_flo) + ').')
    for it_f, f in enumerate(factor):
        f2r_field[..., it_f] *= f

    if not exists(join(tp.data_dir[scope], build_bids_fileame(im_dict) + '.nii.gz')) or force_flag:
        mri = np.asarray(volume_filt).astype('float32', copy=False)

        II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
        II = torch.tensor(II, device='cpu')
        JJ = torch.tensor(JJ, device='cpu')
        KK = torch.tensor(KK, device='cpu')

        affine = torch.tensor(np.matmul(np.linalg.inv(Aaff), np.matmul(np.linalg.inv(Msbj), proxyref.affine)), device='cpu')
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        FIELD = synthmorph_utils.fast_3D_interp_field_torch(torch.from_numpy(f2r_field), II2, JJ2, KK2)
        II3 = II2 + FIELD[:, :, :, 0]
        JJ3 = JJ2 + FIELD[:, :, :, 1]
        KK3 = KK2 + FIELD[:, :, :, 2]

        affine = torch.tensor(np.matmul(Msbj, Aaff), device='cpu')
        RAS_X = affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
        RAS_Y = affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
        RAS_Z = affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]

        affine = torch.tensor(np.linalg.inv(v2r_mri), device='cpu')
        II4 = affine[0, 0] * RAS_X + affine[0, 1] * RAS_Y + affine[0, 2] * RAS_Z + affine[0, 3]
        JJ4 = affine[1, 0] * RAS_X + affine[1, 1] * RAS_Y + affine[1, 2] * RAS_Z + affine[1, 3]
        KK4 = affine[2, 0] * RAS_X + affine[2, 1] * RAS_Y + affine[2, 2] * RAS_Z + affine[2, 3]
        r_tensor = synthmorph_utils.fast_3D_interp_torch(torch.from_numpy(mri), II4, JJ4, KK4, 'linear')

        image = r_tensor.numpy()
        image = np.round(image).astype('uint8')
        im_mri = nib.Nifti1Image(image, proxyref.affine)
        nib.save(im_mri, join(tp.data_dir[scope], build_bids_fileame(im_dict) + '.nii.gz'))


print('\n\n\n\n\n')
print('# --------------------------------- #')
print('# Non-linear SLR: compute template  #')
print('# --------------------------------- #')
print('\n\n')

#####################
# Global parameters #
#####################


# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2', 'gurobi'], help='Likelihood cost function')
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--seg_flag', action='store_true')
arg_parser.add_argument('--force', action='store_true')

arguments = arg_parser.parse_args()
cost = arguments.cost
initial_subject_list = arguments.subjects
seg_flag = arguments.seg_flag
force_flag = arguments.force

print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = 'cpu'
labels_lut = labels.POST_AND_APARC_LUT

##############
# Processing #
##############
scope = 'sreg-synthmorph'
if cost == 'l1': scope += '-l1'
elif cost == 'gurobi': scope += '-gurobi'

atlas_slr = nib.load(synthmorph_utils.atlas_file)

data_loader = bids_loader.T1wLoader(sid_list=initial_subject_list)
data_loader.add_derivatives(SYNTHSEG_DIR)
data_loader.add_derivatives(SREG_LIN_DIR)
data_loader.add_derivatives(SREG_SYNTHMORPH_DIR)
data_loader.add_derivatives(SREG_SYNTHMORPH_L1_DIR)
data_loader.add_derivatives(SREG_SYNTHMORPH_GUROBI_DIR)
subject_list = data_loader.subject_list

for it_subject, subject in enumerate(subject_list):

    if it_subject + 1 < 0: continue

    timepoints = subject.sessions
    if len(timepoints) == 1:
        print('Skipping. It has only 1 timepoint.')
        continue

    fp_dict = {'sub': subject.id, 'desc': 'nonlinTemplate'}
    filename_template = build_bids_fileame(fp_dict)
    nonlinear_template = join(subject.data_dir[scope], filename_template + '_T1w.nii.gz')
    nonlinear_template_mask = join(subject.data_dir[scope], filename_template + '_mask.nii.gz')
    nonlinear_template_seg = join(subject.data_dir[scope], filename_template + '_dseg.nii.gz')

    svf_dict = {'sub': subject.id, 'suffix': 'svf'}
    if not all([exists(join(tp.data_dir[scope], build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz')) for tp in timepoints]):
        print('!!! WARNING: No observations found. Skipping subject ' + subject.id + '.')
        continue

    elif (exists(nonlinear_template) and not force_flag) and (seg_flag and exists(nonlinear_template_seg) and not force_flag):
        print('Skipping: it has already been processed.')
        continue

    else:
        print('Subject: ' + str(subject.id) + '  (' + str(it_subject+1) + ',' + str(len(subject_list)) + ')', end=': ', flush=True)

    ####################################################################################################
    ####################################################################################################

    t_init = time.time()
    for it_tp, tp in enumerate(timepoints): deform_tp(subject, tp)

    # run_flag = any(['run' in f for f in timepoints[0].files['bids']])
    # run_dict = {}#{'run': '01'}

    run_flag = any(['run' in f for f in timepoints[0].files['bids']])
    run_dict = {}  # {'run': '01'}
    if not exists(nonlinear_template) or force_flag:
        mri_list = []
        for it_tp, tp in enumerate(timepoints):
            im_dict = {'sub': subject.id, 'ses': tp.id, 'space': 'SUBJECT', 'suffix': 'T1w'}
            im_mri = nib.load(join(tp.data_dir[scope], build_bids_fileame({**im_dict, **run_dict}) + '.nii.gz'))
            mri_list.append(np.array(im_mri.dataobj))

        template = np.median(mri_list, axis=0)
        img = nib.Nifti1Image(template, subject.vox2ras0)
        nib.save(img, nonlinear_template)

    print('Total Elapsed time: ' + str(time.time() - t_init))

print('\n# ---------- DONE -------------- #')
