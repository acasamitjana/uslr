import os
from os.path import exists, join, dirname, basename
from argparse import ArgumentParser
import pdb
import copy
import subprocess
import shutil

import csv
import bids
import numpy as np
import nibabel as nib
from skimage.transform import resize

from utils.fn_utils import one_hot_encoding, rescale_voxel_size
from utils.labels import POST_LUT
from utils.bf_utils import convert_posteriors_to_unified, bias_field_corr
from src.bids_loader import T1wLoader

from setup import *

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
arg_parser.add_argument('--scope', default='synthseg', nargs='+', help="where to find the segmentations (derivative name)")

args = arg_parser.parse_args()
bidsdir = args.bids
initial_subject_list = args.subjects
if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', args.scope)
csf_labels = [24]

print('\n\n\n\n\n')
print('# --------------------------------------' + '-'.join(['']*len(bidsdir)) + ' #')
print('# BiasField correction on BIDS dataset ' + bidsdir + ' #')
print('#    - Inhomogeneity correction (bias field) ' + ' '.join(['']*(len(bidsdir)-5)) + ' #')
print('#    - Center images in RAS             ' + ' '.join(['']*len(bidsdir))  + ' #')
print('# --------------------------------------' + '-'.join(['']*len(bidsdir)) + ' #')
print('\n\n')

print('Loading dataset. \n')
db_file = join(dirname(BIDS_DIR), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False, database_path=db_file)

bids_loader.add_derivatives(seg_dir)
subject_list = bids_loader.get_subjects() if initial_subject_list is None else initial_subject_list

failed_subjects = []
for it_s, subject in enumerate(subject_list):
    print('\nSubject: ' + subject)

    timepoints = bids_loader.get_session(subject=subject)
    for tp_id in timepoints:
        print('* Session: ' + tp_id)#, end=' ', flush=True)
        # if tp_id != '01': continue

        image_dict = {'extension': 'nii.gz', 'subject': subject, 'session': tp_id, 'acquisition': '1'}

        # try:
        sess_raw_dir = join(bidsdir, 'sub-' + subject, 'ses-' + tp_id, 'anat')
        sess_seg_dir = join(seg_dir, 'sub-' + subject, 'ses-' + tp_id, 'anat')

        #to remove
        dseg_files_to_remove_rawdata = list(filter(lambda x: 'dseg' in x, os.listdir(sess_raw_dir)))
        for dseg in dseg_files_to_remove_rawdata:
            if exists(join(sess_raw_dir, dseg)):
                subprocess.call(['rm', '-rf', join(sess_raw_dir, dseg)])
        #to remove


        res_files = bids_loader.get(**{**image_dict, **{'scope': 'raw', 'suffix': 'T1w'}}, regex_search=False)
        seg_dict = {'subject': subject, 'scope': 'synthseg', 'extension': 'nii.gz', 'session': tp_id}
        if 'T1wdseg' in bids_loader.get(**{**seg_dict, 'return_type': 'id', 'target': 'suffix'}):
            seg_dict['suffix'] = 'T1wdseg'
        else:
            seg_dict['suffix'] = 'dseg'

        seg_files = bids_loader.get(**seg_dict)
        if len(bids_loader.get(**{**image_dict, **{'scope': 'synthseg', 'suffix': 'T1w'}}, regex_search=False)) == len(seg_files):
            for r in res_files:
                subprocess.call(['rm', '-rf',  r.path])
            print('   Resuming. Subject DONE.')
            continue

        if not res_files:
            res_files = []
            ifiles = bids_loader.get(scope='raw', suffix='T1w', extension='nii.gz', subject=subject, session=tp_id)
            for i in ifiles:
                entities = {k: str(v) for k, v in i.entities.items() if k in filename_entities}
                entities['acquisition'] = '1'
                p = nib.load(i.path)
                r, aff_r = rescale_voxel_size(np.array(p.dataobj).astype('float32'), p.affine, [1, 1, 1])
                rfile = basename(bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN))
                res_files.append(join(i.dirname, rfile))
                proxy = nib.Nifti1Image(r, aff_r)
                nib.save(proxy, join(i.dirname, rfile))

                # proxy = nib.load(join(i.dirname, r))
                # aff = proxy.affine
                pixdim = str(np.sqrt(np.sum(aff_r * aff_r, axis=0))[:-1])
                im_json = {
                    "Resolution": {
                        "R": str(pixdim[0]),
                        "A": str(pixdim[1]),
                        "S": str(pixdim[2])
                    },
                    "ImageShape": {
                        "X": str(proxy.shape[0]),
                        "Y": str(proxy.shape[1]),
                        "Z": str(proxy.shape[2])
                    }}

                json_object = json.dumps(im_json, indent=4)
                with open(join(i.dirname, rfile).replace('nii.gz', 'json'), 'w') as outfile:
                    outfile.write(json_object)

        if not res_files and len(seg_files) == 1:
            init_files = bids_loader.get(**{'subject': subject, 'session': tp_id, 'scope': 'raw', 'suffix': 'T1w', 'extension': 'nii.gz'}, regex_search=False)
            if len(init_files) != 1:
                print('   Resuming... Refine the raw image file filters; file(s) encountered: raw image ' + str(init_files) + ' seg: ' + str(seg_files))
                continue
            proxyinit = nib.load(init_files[0].path)
            init_vox_size = np.linalg.norm(proxyinit.affine, 2, 0)[:3]
            if all([np.abs(v-1) < 0.05 for v in init_vox_size]):
                res_filepath = bids_loader.build_path({**image_dict, **{'suffix': 'T1w'}}, scope='raw',
                                                      path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                      absolute_paths=True)
                subprocess.call(['cp', init_files[0].path, res_filepath])
                res_files = [res_filepath]

        if len(res_files) != len(seg_files):
            print('   Resuming... Refine the resampled image file filters; file(s) encountered: res image ' + str(res_files) + ' seg: ' + str(seg_files))
            continue


        for res_file, seg_file in zip(res_files, seg_files):

            # Write JSON file
            if not exists(seg_file.path.replace('nii.gz', 'json')):
                proxy = nib.load(seg_file.path)
                aff = proxy.affine
                pixdim = str(np.sqrt(np.sum(aff * aff, axis=0))[:-1])
                im_json = {"Manual": False,
                           "Resolution": {
                               "R": str(pixdim[0]),
                               "A": str(pixdim[1]),
                               "S": str(pixdim[2])
                           },
                           "ImageShape": {
                               "X": str(proxy.shape[0]),
                               "Y": str(proxy.shape[1]),
                               "Z": str(proxy.shape[2])
                           }}

                json_object = json.dumps(im_json, indent=4)
                with open(seg_file.path.replace('nii.gz', 'json'), 'w') as outfile:
                    outfile.write(json_object)

            if isinstance(res_file, str):
                res_filename = basename(res_file)
            else:
                res_filename = res_file.filename

            res_json_filename = res_filename.replace('nii.gz', 'json')
            im_raw_filename = res_filename.replace('_acq-1', '')
            if not exists(join(sess_raw_dir, im_raw_filename)):
                im_raw_filename = res_filename.replace('_acq-1', '_acq-orig')

            im_filename = res_filename.replace('_acq-1', '_acq-orig')
            im_json_filename = im_filename.replace('nii.gz', 'json')

            if exists(join(sess_seg_dir, im_filename)):
                print('   Resuming. Subject DONE.')
                continue

            proxyim = nib.load(join(sess_raw_dir, im_raw_filename))
            proxyres = nib.load(join(sess_raw_dir, res_filename))

            mask_filename = seg_file.filename.replace('dseg', 'mask')
            proxyseg = nib.load(join(sess_seg_dir, seg_file.filename))

            # ------------------------ #
            # Computing masks          #
            # ------------------------ #
            print('  - Computing masks from dseg files.')
            if not exists(join(sess_seg_dir, mask_filename)):
                seg = np.array(proxyseg.dataobj)
                mask = seg > 0
                for l in csf_labels:
                    mask[seg == l] = 0

                img = nib.Nifti1Image(mask.astype('uint8'), proxyseg.affine)
                nib.save(img, join(sess_seg_dir, mask_filename))

            # ------------------------ #
            # Bias field correction    #
            # ------------------------ #
            print('  - Correcting for inhomogeneities and normalisation (WM=110).')
            if not exists(join(sess_seg_dir, im_filename)):
                vox2ras0 = proxyres.affine
                mri_acq = np.asarray(proxyres.dataobj)
                seg = np.array(proxyseg.dataobj)

                soft_seg = one_hot_encoding(seg, categories=POST_LUT)
                soft_seg = convert_posteriors_to_unified(soft_seg, lut=POST_LUT)
                mri_acq_corr, bias_field = bias_field_corr(mri_acq, soft_seg, penalty=1, VERBOSE=False)
                if bias_field is None:
                    print('    Skipping. Bias field could not be computed.')
                    continue

                del soft_seg

                mask = seg > 0
                wm_mask = (seg == 2) | (seg == 41)

                del seg

                m = np.mean(mri_acq_corr[wm_mask])
                mri_acq_corr = 110 * mri_acq_corr / m

                img = nib.Nifti1Image(np.clip(mri_acq_corr, 0, 255).astype('uint8'), proxyres.affine)
                nib.save(img, join(sess_seg_dir, res_filename))


                vox2ras0_orig = proxyim.affine
                mri_acq_orig = np.asarray(proxyim.dataobj)

                new_vox_size = np.linalg.norm(vox2ras0_orig, 2, 0)[:3]
                vox_size = np.linalg.norm(vox2ras0, 2, 0)[:3]

                #JSON
                json_dict = {
                           "Resolution": {
                               "R": str(vox_size[0]),
                               "A": str(vox_size[1]),
                               "S": str(vox_size[2])
                           },
                           "ImageShape": {
                               "X": str(mri_acq_corr.shape[0]),
                               "Y": str(mri_acq_corr.shape[1]),
                               "Z": str(mri_acq_corr.shape[2])
                           },
                           "Description": "Bias field corrected image."
                           }


                json_object = json.dumps(json_dict, indent=4)
                with open(join(sess_seg_dir, res_json_filename), 'w') as outfile:
                    outfile.write(json_object)

                del mri_acq, mri_acq_corr

                if all([a==b for a, b in zip(vox_size, new_vox_size)]):
                    try:
                        subprocess.call(['ln', '-s', join(sess_seg_dir, res_file), join(sess_seg_dir, im_filename)])
                        subprocess.call(['ln', '-s', join(sess_seg_dir, res_json_filename), join(sess_seg_dir, im_json_filename)])
                    except:

                        subprocess.call(['cp', join(sess_seg_dir, res_file), join(sess_seg_dir, im_filename)])
                        subprocess.call(['cp', join(sess_seg_dir, res_json_filename), join(sess_seg_dir, im_json_filename)])

                else:
                    bias_field_resize, _ = rescale_voxel_size(bias_field, vox2ras0, new_vox_size)
                    if bias_field_resize.shape != mri_acq_orig.shape: bias_field_resize = resize(bias_field_resize, mri_acq_orig.shape)

                    wm_mask_resize, _ = rescale_voxel_size(wm_mask.astype('float'), vox2ras0, new_vox_size)
                    if wm_mask_resize.shape != mri_acq_orig.shape: wm_mask_resize = resize(wm_mask_resize, mri_acq_orig.shape, order=0)
                    wm_mask_resize = wm_mask_resize > 0

                    mask_resize, _ = rescale_voxel_size(mask.astype('float'), vox2ras0, new_vox_size)
                    if mask_resize.shape != mri_acq_orig.shape: mask_resize = resize(mask_resize, mri_acq_orig.shape, order=1)
                    mask_resize = mask_resize > 0

                    mri_acq_orig_corr = copy.copy(mri_acq_orig.astype('float32'))
                    mri_acq_orig_corr[mask_resize] = mri_acq_orig_corr[mask_resize] / bias_field_resize[mask_resize]

                    m = np.mean(mri_acq_orig_corr[wm_mask_resize])
                    mri_acq_orig_corr = 110 * mri_acq_orig_corr / m

                    img = nib.Nifti1Image(np.clip(mri_acq_orig_corr, 0, 255).astype('uint8'), proxyim.affine)
                    nib.save(img, join(sess_seg_dir, im_filename))

                    json_dict = {
                           "Resolution": {
                               "R": str(new_vox_size[0]),
                               "A": str(new_vox_size[1]),
                               "S": str(new_vox_size[2])
                           },
                           "ImageShape": {
                               "X": str(mri_acq_orig_corr.shape[0]),
                               "Y": str(mri_acq_orig_corr.shape[1]),
                               "Z": str(mri_acq_orig_corr.shape[2])
                           },
                           "Description": "Bias field corrected image."
                           }

                    json_object = json.dumps(json_dict, indent=4)
                    with open(join(sess_seg_dir, im_json_filename), 'w') as outfile:
                        outfile.write(json_object)

                    del bias_field, bias_field_resize, wm_mask, wm_mask_resize, mri_acq_orig, mri_acq_orig_corr

print('******************')
print('****** DONE ******')
print('******************')