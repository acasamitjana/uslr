import os
import pdb
import time
from os.path import exists, join
from argparse import ArgumentParser
import subprocess
import nibabel as nib
import numpy as np
import torch
from tensorflow import keras
import bids

# project imports
from src import bids_loader
from utils import synthmorph_utils, def_utils, io_utils
from setup import *
from utils.fn_utils import compute_centroids_ras
from utils.synthmorph_utils import VxmDenseOriginalSynthmorph, instance_register, path_model_registration, \
    RescaleTransform, VecInt, fast_3D_interp_torch, fast_3D_interp_field_torch


#####################
# Global parameters #
#####################

# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--scope', default='sreg-synthmorph-l1',
                        choices=['sreg-synthmorph', 'sreg-lin', 'sreg-synthmorph-l1', 'synthmorph'])
arg_parser.add_argument('--template_str', default='SYNTHMORPH', choices=['MNI', 'SYNTHMORPH'])
arg_parser.add_argument('--nepochs_refinement', default=0, type=int)
arg_parser.add_argument('--force', action='store_true')
arg_parser.add_argument('--nosym', action='store_true')

arguments = arg_parser.parse_args()
bidsdir = arguments.bids
initial_subject_list = arguments.subjects
reg_algorithm = arguments.scope
template_str = arguments.template_str
force_flag = arguments.force

print('\n\n\n\n\n')
print('# ------------------------------------------------' + '-'*len(template_str) + ' #')
print('# Register Nonlinear Template to reference space: ' + template_str + ' #')
print('# ------------------------------------------------' + '-'*len(template_str) + ' #')
print('\n\n')

if template_str == 'MNI':
    ref_template = MNI_TEMPLATE
    ref_template_mask = MNI_TEMPLATE_MASK

elif template_str == 'SYNTHMORPH':
    ref_template = synthmorph_utils.atlas_file
    ref_template_mask = synthmorph_utils.atlas_mask_file
    if not exists(ref_template_mask):
        proxy = nib.load(synthmorph_utils.atlas_seg_file)
        seg = np.array(proxy.dataobj)
        mask = seg > 0
        img = nib.Nifti1Image(mask.astype('uint8'), proxy.affine)
        nib.save(img, ref_template_mask)
else:
    raise ValueError('Please, specify a valid template name.')

proxyatlas = nib.load(synthmorph_utils.atlas_file)
atlas_aff = proxyatlas.affine
proxytemplate = nib.load(ref_template)

##################
# Data variables #
##################
if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'synthseg')
slr_lin_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'slr-lin')
slr_nonlin_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'slr-nonlin')

db_file = join(os.path.dirname(BIDS_DIR), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False, database_path=db_file)

bids_loader.add_derivatives(seg_dir)
bids_loader.add_derivatives(slr_lin_dir)
bids_loader.add_derivatives(slr_nonlin_dir)
subject_list = bids_loader.get_subjects() if initial_subject_list is None else initial_subject_list

# print('[COMPUTE MNI Transform]')
# for it_subject, subject in enumerate(subject_list):
#
#     ###########TO REMOVE -- PD only --
#     sreg_l2_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'sreg-synthmorph', 'sub-' + subject)
#     sreg_l1_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'sreg-synthmorph-l1', 'sub-' + subject)
#
#     for f in os.listdir(sreg_l2_dir):
#         if f[:3] == 'ses': continue
#         subprocess.call(['mv', join(sreg_l2_dir, f), join(slr_nonlin_dir, 'sub-' + subject)])
#
#     for f in os.listdir(sreg_l1_dir):
#         subprocess.call(['mv', join(sreg_l1_dir, f), join(slr_nonlin_dir, 'sub-' + subject)])
#     pdb.set_trace()
#     ###########TO REMOVE -- PD only --

nl_fname_suffix = '_space-' + template_str + '_desc-field_nonlinear.nii.gz'
if not force_flag:
    subject_list = list(filter(lambda s: join(slr_nonlin_dir, 'sub-' + s, 'sub-' + s + nl_fname_suffix), subject_list))

for it_subject, subject in enumerate(subject_list):

    t_init = time.time()
    print(' * Subject: ' + str(subject) + '  -  ' + str(it_subject) + '/' + str(len(subject_list)))

    timepoints = bids_loader.get_session(subject=subject)
    slr_nonlin_dir_sbj = join(slr_nonlin_dir, 'sub-' + subject)
    if len(timepoints) == 1:
        print('Skipping. Subject has only one session.')
        continue

    elif not exists(join(slr_nonlin_dir_sbj, 'sub-' + subject + '_desc-atlas_aff.npy')):
        print('Registration to SYNTHMORPH atlas not availabe in slr-nonlin pipeline. Please, compute it and re-run the script.')
        continue

    template = join(slr_nonlin_dir_sbj, 'sub-' + subject + '_desc-nonlinTemplate_T1w.nii.gz')
    template_mask = join(slr_nonlin_dir_sbj, 'sub-' + subject + '_desc-nonlinTemplate_mask.nii.gz')

    if not exists(template) or not exists(template_mask):
        print('Skipping. Template does not exist')

    nlFile = join(slr_nonlin_dir_sbj, 'sub-' + subject + nl_fname_suffix)
    outputFile = join('/tmp', subject + '_' + template_str +  '_lin_template.nonlin.nii.gz')

    if not exists(nlFile) or force_flag:
        if template_str == 'SYNTHMORPH':
            M_ref = np.eye(4)
            R_aff = np.eye(4)

            Msbj = np.load(join(slr_nonlin_dir_sbj, 'sub-' + subject + '_desc-atlas_aff.npy'))
            Rlin = torch.tensor(np.array(proxyatlas.dataobj))
            Rlin = Rlin / torch.max(Rlin)
            F = np.array(nib.load(template).dataobj)
            Flin, F_aff, F_h = synthmorph_utils.compute_atlas_alignment(template, template_mask, proxyatlas, Msbj)
            FMlin, FM_aff, FM_h = synthmorph_utils.compute_atlas_alignment(template_mask, template_mask, proxyatlas, Msbj)

        elif template_str == 'MNI':
            # Compute Msbj
            if not exists(MNI_to_ATLAS):
                centroid_sbj, ok = compute_centroids_ras(MNI_TEMPLATE_SEG, synthmorph_utils.labels_registration)
                centroid_atlas = np.load(synthmorph_utils.atlas_cog_file)
                M_ref = synthmorph_utils.getM(centroid_atlas[:, ok > 0], centroid_sbj[:, ok > 0], use_L1=False)
                np.save(MNI_to_ATLAS, M_ref)

            else:
                M_ref = np.load(MNI_to_ATLAS)

            Msbj = np.load(join(slr_nonlin_dir_sbj, 'sub-' + subject + '_desc-atlas_aff.npy'))

            F = np.array(nib.load(template).dataobj)
            Rlin, R_aff, R_h = synthmorph_utils.compute_atlas_alignment(ref_template, ref_template_mask, proxyatlas, M_ref)
            RMlin, RM_aff, RM_h = synthmorph_utils.compute_atlas_alignment(ref_template_mask, ref_template_mask, proxyatlas, M_ref)
            Flin, F_aff, F_h = synthmorph_utils.compute_atlas_alignment(template, template_mask, proxyatlas, Msbj)
            FMlin, FM_aff, FM_h = synthmorph_utils.compute_atlas_alignment(template_mask, template_mask, proxyatlas, Msbj)

        else:
            print(template_str + " template still not implemented. Skipping")
            continue

        # img = nib.Nifti1Image(Rlin.numpy(), proxyatlas.affine)
        # nib.save(img, '/tmp/R_lin.nii.gz')
        # img = nib.Nifti1Image(Flin.numpy(), proxyatlas.affine)
        # nib.save(img, '/tmp/F_lin.nii.gz')

        print('    - Nonlinear')
        cnn = VxmDenseOriginalSynthmorph.load(path_model_registration)
        svf1 = cnn.register(Flin.detach().numpy()[np.newaxis, ..., np.newaxis],
                            Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])
        if arguments.nosym:
            svf = svf1
        else:
            svf2 = cnn.register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                Flin.detach().numpy()[np.newaxis, ..., np.newaxis])
            svf = 0.5 * svf1 - 0.5 * svf2

        if arguments.nepochs_refinement > 0:
            instance_model = instance_register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                               Flin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                               svf, inshape=proxyatlas.shape, epochs=arguments.nepochs_refinement)
            svf_refined = instance_model.references.flow_layer(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])
        else:
            svf_refined = svf

        upscaler = keras.Sequential([RescaleTransform(2)])
        svf_final = upscaler(svf_refined)

        img = nib.Nifti1Image(np.squeeze(svf_final), proxyatlas.affine)
        nib.save(img, nlFile)

        if DEBUG:
            integrator = keras.Sequential([VecInt(method='ss', int_steps=7)])
            f2r_field = torch.tensor(np.squeeze(integrator(svf_final)))

            II, JJ, KK = np.meshgrid(np.arange(proxytemplate.shape[0]), np.arange(proxytemplate.shape[1]), np.arange(proxytemplate.shape[2]), indexing='ij')
            II = torch.tensor(II, device='cpu')
            JJ = torch.tensor(JJ, device='cpu')
            KK = torch.tensor(KK, device='cpu')
            #
            if template_str != 'SYNTHMORPH':
                affine = torch.tensor(np.matmul(np.linalg.inv(atlas_aff), np.matmul(np.linalg.inv(M_ref), R_aff)), device='cpu')
                II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
                JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
                KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
                FIELD = fast_3D_interp_field_torch(f2r_field, II2, JJ2, KK2)

            else:
                II2, JJ2, KK2 = II, JJ, KK
                FIELD = f2r_field
            #
            II3 = II2 + FIELD[:, :, :, 0]
            JJ3 = JJ2 + FIELD[:, :, :, 1]
            KK3 = KK2 + FIELD[:, :, :, 2]
            #
            print('  Deforming floating image')
            affine = torch.tensor(np.matmul(np.linalg.inv(F_aff), np.matmul(Msbj, atlas_aff)), device='cpu')
            II4 = affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
            JJ4 = affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
            KK4 = affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]
            #
            registered = fast_3D_interp_torch(torch.tensor(F), II4, JJ4, KK4, 'linear')
            img = nib.Nifti1Image(np.squeeze(registered.numpy()), R_aff)
            nib.save(img, outputFile)


    print('. Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)) + ' seconds.')

print('\n# ---------- DONE -------------- #')