import os
import pdb
import subprocess
from os.path import exists, join, dirname, basename
import time
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
import bids

# project imports
from utils.io_utils import build_bids_fileame
from setup import *
from src import bids_loader, slr
from utils import labels, slr_utils, def_utils, io_utils, synthmorph_utils, fn_utils


def deform_tp(svf_file, proxyref, seg_flag=False, channel_chunk=20):

    entities = svf_file.entities
    tp_dict = {'session': entities['session'], 'subject': entities['subject']}
    if 'run' in entities.keys():
        tp_dict['run'] = entities['run']

    ent_im = {**{'space': 'SUBJECT', 'suffix': 'T1w'}, **tp_dict}
    im_filename = basename(bids_loader.build_path({**ent_im, 'extension': 'nii.gz'}, scope='slr-nonlin',
                                                  path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                  absolute_paths=False))
    mask_filename = im_filename.replace('T1w', 'mask')
    seg_filename = im_filename.replace('T1w', 'dseg')

    sess_im_filepath = join(dirname(svf_file.path), im_filename)
    sess_mask_filepath = join(dirname(svf_file.path), mask_filename)
    sess_seg_filepath = join(dirname(svf_file.path), seg_filename)

    aff_dict = {'subject': subject, 'desc': 'aff', 'scope': 'slr-lin', 'extension': 'npy'}
    if 'T1w' in bids_loader.get(**{**aff_dict, 'return_type': 'id', 'target': 'suffix'}):
        aff_dict['suffix'] = 'T1w'

    affine_file = bids_loader.get(**{**aff_dict, 'session': entities['session']})
    im_file = bids_loader.get(**{**{'scope':'synthseg', 'extension':'nii.gz', 'suffix': 'T1w', 'acquisition': 'orig'}, **tp_dict})
    seg_file = bids_loader.get(**{**{'scope':'synthseg', 'extension':'nii.gz', 'suffix': 'T1wdseg', 'acquisition': '1'}, **tp_dict})

    if len(affine_file) != 1:
        print('Wrong affine file entities')
        return

    affine_matrix = np.load(affine_file[0])

    proxysvf = nib.load(svf_file)

    if len(im_file) != 1:
        im_file = bids_loader.get(**{**{'scope': 'synthseg', 'extension': 'nii.gz', 'suffix': 'T1w', 'acquisition': '1'}, **tp_dict})
        if len(im_file) != 1:
            im_file = bids_loader.get(**{**{'scope': 'synthseg', 'extension': 'nii.gz', 'suffix': 'T1w'}, **tp_dict})
            if len(im_file) != 1:
                print('Wrong im_file file entities')
                # pdb.set_trace()
                return

    proxyimage = nib.load(im_file[0].path)
    v2r_mri = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)

    if len(seg_file) != 1:
        print('Wrong seg_file file entities')
        # pdb.set_trace()
        return

    proxyseg = nib.load(seg_file[0].path)
    v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)

    if not exists(sess_im_filepath) or force_flag:
        pixdim = np.sqrt(np.sum(proxyimage.affine * proxyimage.affine, axis=0))[:-1]
        new_vox_size = np.array([1, 1, 1])
        factor = pixdim / new_vox_size
        sigmas = 0.25 / factor
        sigmas[factor > 1] = 0  # don't blur if upsampling

        mri = np.asarray(proxyimage.dataobj).astype('float32', copy=False)
        mri = gaussian_filter(mri, sigmas)

        proxyflo_im = nib.Nifti1Image(mri, v2r_mri)

        im_mri = def_utils.vol_resample(proxyref, proxyflo_im, proxysvf=proxysvf)
        image = np.array(im_mri.dataobj)
        image = np.round(image).astype('uint8')
        im_mri = nib.Nifti1Image(image, im_mri.affine)
        nib.save(im_mri, sess_im_filepath)

    if not exists(sess_mask_filepath) or force_flag:
        seg = np.asarray(proxyseg.dataobj)
        mask = seg > 0

        proxyflo_mask = nib.Nifti1Image(mask.astype('float'), v2r_seg)
        mask_mri = def_utils.vol_resample(proxyref, proxyflo_mask, proxysvf=proxysvf)
        nib.save(mask_mri, sess_mask_filepath)


    if seg_flag and (not exists(sess_seg_filepath) or force_flag):
        seg_list = []
        seg = np.asarray(proxyseg.dataobj)
        for it_c in range(0, len(labels_lut), channel_chunk):
            cat_chunk = {k: np.mod(it_d, channel_chunk) for it_d, k in enumerate(labels_lut.keys()) if it_d >= it_c and it_d < it_c + channel_chunk}
            seg_onehot = fn_utils.one_hot_encoding(seg, categories=cat_chunk).astype('float32', copy=False)
            proxyflo_seg = nib.Nifti1Image(seg_onehot, v2r_seg)
            seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxysvf=proxysvf)# ,mode='nearest')

            # In case the last iteration has only 1 channel (it squeezes due to batch dimension)
            if len(seg_mri.shape) == 3:
                seg_list += [np.array(seg_mri.dataobj)[..., np.newaxis]]
            else:
                seg_list += [np.array(seg_mri.dataobj)]
            del seg_mri

        seg_post = np.concatenate(seg_list, axis=-1)
        seg_mri = nib.Nifti1Image(seg_post, proxyref.affine)
        # seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxysvf=proxysvf)# ,mode='nearest')
        # nib.save(seg_mri, join(tp.data_dir[scope], build_bids_fileame(seg_dict) + '.nii.gz'))
        nib.save(seg_mri, sess_seg_filepath)


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
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2', 'gurobi'], help='Likelihood cost function')
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--seg_flag', action='store_true')
arg_parser.add_argument('--force', action='store_true')

arguments = arg_parser.parse_args()
bidsdir = arguments.bids
cost = arguments.cost
initial_subject_list = arguments.subjects
seg_flag = arguments.seg_flag
force_flag = arguments.force
factor = 2
labels_lut = labels.POST_AND_APARC_LUT

##############
# Processing #
##############
scope = 'slr-nonlin'
atlas_slr = nib.load(synthmorph_utils.atlas_file)
cp_shape = tuple([s//factor for s in atlas_slr.shape])

print('Reading database.')
if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'synthseg')
slr_dir = '/media/biofisica/BIG_DATA/ADNI-T1' # os.path.dirname(bidsdir)
slr_lin_dir = os.path.join(slr_dir, 'derivatives', 'slr-lin')
slr_nonlin_dir = os.path.join(slr_dir, 'derivatives', 'slr-nonlin')

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

print('\nDeforming nonlinear images.')
for it_subject, subject in enumerate(subject_list):
    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x:
                             len(bids_loader.get(extension='nii.gz', subject=subject, session=x, suffix='T1w')) > 0,
                             timepoints))
    timepoints = list(filter(lambda x: not exists(join(dirname(bids_loader.get(scope='synthseg', subject=subject, session=x, return_type='filename')[0]), 'excluded_file.txt')), timepoints))

    dir_lin_subj = join(slr_lin_dir, 'sub-' + subject)
    linear_template = join(dir_lin_subj, 'sub-' + subject + '_desc-linTemplate_T1w.nii.gz')
    if not exists(linear_template):
        print('Linear template does not exist for subject ' + subject)
        continue

    dir_nonlin_subj = join(slr_nonlin_dir, 'sub-' + subject)
    linear_template = {}
    for file in bids_loader.get(subject=subject, desc='linTemplate', extension='nii.gz'):
        if 'dseg' in file.entities['suffix']:
            linear_template['dseg'] = file
        elif 'mask' in file.entities['suffix']:
            linear_template['mask'] = file
        elif file.entities['suffix'] == 'T1w':
            linear_template['image'] = file

    nonlinear_template = join(dir_nonlin_subj, linear_template['image'].filename.replace('linTemplate', 'nonlinTemplate'))
    nonlinear_template_mask = join(dir_nonlin_subj, linear_template['mask'].filename.replace('linTemplate', 'nonlinTemplate'))
    nonlinear_template_seg = join(dir_nonlin_subj, linear_template['dseg'].filename.replace('linTemplate', 'nonlinTemplate'))
    nonlinear_template_std = nonlinear_template.replace('T1w', 'std')

    for im_type, im_file in linear_template.items():
        subprocess.call(['rm', '-rf', im_file.path])

    if len(timepoints) == 1:
        if not exists(nonlinear_template): subprocess.call(['cp', linear_template['image'].path, nonlinear_template])
        if not exists(nonlinear_template_mask): subprocess.call(['cp', linear_template['mask'].path, nonlinear_template_mask])
        if not exists(nonlinear_template_seg):  subprocess.call(['cp', linear_template['dseg'].path, nonlinear_template_seg])
        print('Skipping. It has only 1 timepoint.')
        continue

    svf_files = bids_loader.get(**{'subject': subject, 'suffix': 'svf', 'scope': scope})

    if len(svf_files) < len(timepoints):
        print('!!! WARNING: No observations found. Skipping subject ' + subject + '.')
        continue

    elif (exists(nonlinear_template) and not force_flag) and (seg_flag and exists(nonlinear_template_seg) and not force_flag):
        print('Skipping: it has already been processed.')
        continue

    else:
        print('Subject: ' + str(subject) + '  (' + str(it_subject+1) + ',' + str(len(subject_list)) + ')', end=': ', flush=True)

    ####################################################################################################
    ####################################################################################################

    t_init = time.time()
    for it_svf_f, svf_f in enumerate(svf_files): deform_tp(svf_f, nib.load(linear_template['image']), seg_flag=seg_flag)

    print('Total Elapsed time' + str(time.time() - t_init))

print('\n# ---------- DONE -------------- #')
print('\n\n')

print('Reading database.')
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
print('\nComputing nonlinear template')
for it_subject, subject in enumerate(subject_list):
    t_init = time.time()

    dir_nonlin_subj = join(slr_nonlin_dir, 'sub-' + subject)
    linear_template = {}
    for file in bids_loader.get(subject=subject, desc='linTemplate', extension='nii.gz'):
        if 'dseg' in file.entities['suffix']:
            linear_template['dseg'] = file
        elif 'mask' in file.entities['suffix']:
            linear_template['mask'] = file
        elif file.entities['suffix'] == 'T1w':
            linear_template['image'] = file

    nonlinear_template = join(dir_nonlin_subj, linear_template['image'].filename.replace('linTemplate', 'nonlinTemplate'))
    nonlinear_template_mask = join(dir_nonlin_subj, linear_template['mask'].filename.replace('linTemplate', 'nonlinTemplate'))
    nonlinear_template_seg = join(dir_nonlin_subj, linear_template['dseg'].filename.replace('linTemplate', 'nonlinTemplate'))
    nonlinear_template_std = nonlinear_template.replace('T1w', 'std')

    mri_file_list = bids_loader.get(subject=subject, scope='slr-nonlin', space='SUBJECT', suffix='T1w', extension='nii.gz')
    mri_list = [np.array(nib.load(mfl.path).dataobj) for mfl in mri_file_list]
    # for it_svf_f, svf_f in enumerate(svf_files):
    #     entities = svf_f.entities
    #     tp_dict = {'session': entities['session'], 'subject': entities['subject']}
    #     if 'run' in entities.keys():
    #         tp_dict['run'] = entities['run']
    #
    #     ent_im = {**{'space': 'SUBJECT', 'suffix': 'T1w', 'scope': 'slr-nonlin'}, **tp_dict}
    #     im_file = bids_loader.get(**ent_im)
    #     if len(im_file) == 1:
    #         im_mri = nib.load(im_file[0].path)
    #         mri_list.append(np.array(im_mri.dataobj))
    if len(mri_list) > 0:
        template = np.median(mri_list, axis=0)
        template_std = np.std(mri_list, axis=0)
        img = nib.Nifti1Image(template, nib.load(linear_template['image']).affine)
        nib.save(img, nonlinear_template)

        img = nib.Nifti1Image(template_std, nib.load(linear_template['image']).affine)
        nib.save(img, nonlinear_template_std)

    mask_filelist = bids_loader.get(subject=subject, scope='slr-nonlin', space='SUBJECT', suffix='mask', extension='nii.gz')
    mask_list = [np.array(nib.load(mfl.path).dataobj) for mfl in mask_filelist]
    # mask_list = []
    # for it_svf_f, svf_f in enumerate(svf_files):
    #     entities = svf_f.entities
    #     tp_dict = {'session': entities['session'], 'subject': entities['subject']}
    #     if 'run' in entities.keys():
    #         tp_dict['run'] = entities['run']
    #
    #     ent_im = {**{'space': 'SUBJECT', 'suffix': 'mask', 'scope': 'slr-nonlin'}, **tp_dict}
    #     im_file = bids_loader.get(**ent_im)
    #     if len(im_file) == 1:
    #         im_mask = nib.load(im_file[0].path)
    #         mask_list.append(np.array(im_mask.dataobj))
    if len(mask_list) > 0:
        template = np.sum(mask_list, axis=0)/len(mask_list)
        img = nib.Nifti1Image(template, nib.load(linear_template['image']).affine)
        nib.save(img, nonlinear_template_mask)

        nonlinear_etiv = nonlinear_template_mask.replace('mask', 'etiv')
        nonlinear_etiv = nonlinear_etiv.replace('nii.gz', 'npy')
        np.save(nonlinear_etiv, np.sum(template > 0.5))

    if seg_flag:
        seg_list = np.zeros((len(labels_lut), ) + nib.load(nonlinear_template).shape)
        seg_filelist = bids_loader.get(subject=subject, scope='slr-nonlin', space='SUBJECT', suffix='dseg', extension='nii.gz')
        if len(seg_filelist) > 0:
            for seg_file in seg_filelist:
                seg = np.array(nib.load(seg_file).dataobj)
                seg_list += np.transpose(seg, axes=(3, 0, 1, 2))
            # for it_svf_f, svf_f in enumerate(svf_files):
            #     entities = svf_f.entities
            #     tp_dict = {'session': entities['session'], 'subject': entities['subject']}
            #     if 'run' in entities.keys():
            #         tp_dict['run'] = entities['run']
            #
            #     ent_im = {**{'space': 'SUBJECT', 'suffix': 'dseg', 'scope': 'slr-nonlin'}, **tp_dict}
            #     im_file = bids_loader.get(**ent_im)
            #     if len(im_file) == 1:
            #         im_seg = nib.load(im_file[0].path)
            #         seg = np.array(im_seg.dataobj)
            #         seg_list += np.transpose(seg, axes=(3, 0, 1, 2))

            template = np.argmax(seg_list, axis=0)
            template_true = np.zeros_like(template)
            for l, it_l in labels_lut.items(): template_true[template==it_l] = l

            img = nib.Nifti1Image(template_true.astype('int16'), nib.load(linear_template['image']).affine)
            nib.save(img, nonlinear_template_seg)

    print('Total Elapsed time: ' + str(time.time() - t_init))
print('\n# ---------- DONE -------------- #')
