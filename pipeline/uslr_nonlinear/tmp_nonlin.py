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

    subject_shape = subject.image_shape
    conditions_image = {'suffix': 'T1w', 'acquisition': '1', 'scope': 'sreg-lin', 'space': 'SUBJECT', 'extension': '.nii.gz', 'run': '01'}
    svf_dict = {'sub': subject.id, 'suffix': 'svf'}
    im_dict = {'sub': subject.id, 'ses': tp.id, 'space': 'SUBJECT', 'suffix': 'T1w'}

    A, Aaff, Ah = io_utils.load_volume(synthmorph_utils.atlas_file, im_only=False, squeeze=True, dtype=None, aff_ref=None)
    Amri = nib.Nifti1Image(A, Aaff)
    Aaff = Aaff.astype('float32')
    Msbj = np.load(join(subject.data_dir['sreg-synthmorph'], 'sub-' + subject.id + '_desc-atlas_aff.npy')).astype('float32')
    flow_v2r = Aaff #Msbj @ Aaff

    proxysvf = nib.load(join(tp.data_dir[scope], build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'))
    proxyflow = def_utils.integrate_svf(proxysvf)
    proxyflow = nib.Nifti1Image(np.array(proxyflow.dataobj), flow_v2r)

    proxyref = nib.Nifti1Image(np.zeros(subject_shape), subject.vox2ras0)

    image_file = tp.get_files(**conditions_image)
    if not image_file:
        conditions_image.pop('acquisition')
        image_file = tp.get_files(**conditions_image)
    Rlin, Raff, Rh = synthmorph_utils.compute_atlas_alignment(join(tp.data_dir['sreg-lin'], image_file[0]),
                                                              join(tp.data_dir['sreg-lin'], image_file[0]), Amri, Msbj, normalize=False)
    if not exists(join(tp.data_dir[scope], build_bids_fileame(im_dict) + '.nii.gz')) or force_flag:
        proxyflo_im = nib.Nifti1Image(Rlin.numpy(), Aaff)

        ii = np.arange(0, Amri.shape[0], dtype='int32')
        jj = np.arange(0, Amri.shape[1], dtype='int32')
        kk = np.arange(0, Amri.shape[2], dtype='int32')

        II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')
        II = torch.tensor(II, device='cpu')
        JJ = torch.tensor(JJ, device='cpu')
        KK = torch.tensor(KK, device='cpu')
        FIELD = torch.from_numpy(np.array(proxyflow.dataobj))

        r_tensor = synthmorph_utils.fast_3D_interp_torch(Rlin, II + FIELD[..., 0], JJ + FIELD[..., 1], KK + FIELD[..., 2], 'linear')


        # affine = torch.tensor(np.linalg.inv(flow_v2r) @ ref_v2r)
        # vM_ref_svf_I = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        # vM_ref_svf_J = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        # vM_ref_svf_K = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
        #
        # im_mri = def_utils.vol_resample(Amri, proxyflo_im, proxyflow=proxyflow)#def_utils.vol_resample(proxyref, proxyflo_mask, proxysvf=proxysvf)
        image = r_tensor.numpy()
        image = np.round(image).astype('uint8')
        im_mri = nib.Nifti1Image(image, Aaff)
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
