from setup import *

import subprocess
from os.path import exists, join, dirname, basename
import time
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import bids

from utils import labels, def_utils, fn_utils

def process_timepoints(subject, timepoints, bids_loader, proxyref, channel_chunk=20, force_flag=False):

    svf_files = bids_loader.get(**{'subject': subject, 'suffix': 'svf', 'scope': 'uslr-nonlin'})
    if len(svf_files) < len(timepoints):
        print('[error] Not all timepoints have SVF available. Skipping.')
        return subject

    ####################################################################################################
    ####################################################################################################
    output_dict = {svf_file.entities['session']: {} for it_svf_f, svf_file in enumerate(svf_files)}
    for it_svf_f, svf_file in enumerate(svf_files):
        svf_ent = svf_file.entities
        tp_ent = {'session': svf_ent['session'], 'subject': svf_ent['subject']}
        aff_ent = {'desc': 'aff', 'scope': 'uslr-lin', 'extension': 'npy', 'suffix': 'T1w'}
        im_out_ent = {**{'space': 'SUBJECT', 'suffix': 'T1w'}, **tp_ent}

        im_filename = bids_loader.build_path({**im_out_ent, 'extension': 'nii.gz'}, scope='uslr-nonlin',
                                             path_patterns=BIDS_PATH_PATTERN, validate=False)
        im_filename = basename(im_filename)
        mask_filename = im_filename.replace('T1w', 'mask')
        seg_filename = im_filename.replace('T1w', 'dseg')

        sess_im_filepath = join(dirname(svf_file.path), im_filename)
        sess_mask_filepath = join(dirname(svf_file.path), mask_filename)
        sess_seg_filepath = join(dirname(svf_file.path), seg_filename)

        affine_file = bids_loader.get(**{**aff_ent, **tp_ent})
        im_file = bids_loader.get(scope='synthseg', extension='nii.gz', suffix='T1w', acquisition=None, **tp_ent)
        seg_file = bids_loader.get(scope='synthseg', extension='nii.gz', suffix='T1wdseg', **tp_ent)

        if len(affine_file) != 1:
            print('[error] wrong affine entities. Skipping.')
            return subject

        if len(im_file) != 1:
            im_file = bids_loader.get(scope='synthseg', extension='nii.gz', suffix='T1w', acquisition='1', **tp_ent)
            if len(im_file) != 1:
                print('[error] wrong image file entities. Skipping.')
                return subject

        if len(seg_file) != 1:
            print('[error] wrong segmentation entities. Skipping.')
            # pdb.set_trace()
            return subject

        proxysvf = nib.load(svf_file)
        proxyimage = nib.load(im_file[0].path)
        proxyseg = nib.load(seg_file[0].path)
        affine_matrix = np.load(affine_file[0])

        v2r_mri = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)
        v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)

        if not exists(sess_im_filepath) or force_flag:
            mri = fn_utils.gaussian_smoothing_voxel_size(proxyimage, [1,1,1])
            proxyflo_im = nib.Nifti1Image(mri, v2r_mri)
            im_mri = def_utils.vol_resample(proxyref, proxyflo_im, proxysvf=proxysvf)

            image = np.array(im_mri.dataobj)
            image = np.round(image).astype('uint8')
            im_mri = nib.Nifti1Image(image, im_mri.affine)
            nib.save(im_mri, sess_im_filepath)
            output_dict[svf_ent['session']]['im'] = image
        else:
            im_mri = nib.load(sess_im_filepath)
            output_dict[svf_ent['session']]['im'] = np.array(im_mri.dataobj)


        if not exists(sess_mask_filepath) or force_flag:
            seg = np.asarray(proxyseg.dataobj)
            mask = seg > 0

            proxyflo_mask = nib.Nifti1Image(mask.astype('float'), v2r_seg)
            mask_mri = def_utils.vol_resample(proxyref, proxyflo_mask, proxysvf=proxysvf)
            nib.save(mask_mri, sess_mask_filepath)
            output_dict[svf_ent['session']]['mask'] = np.array(mask_mri.dataobj)
        else:
            im_mri = nib.load(sess_im_filepath)
            output_dict[svf_ent['session']]['mask'] = np.array(im_mri.dataobj)

        if seg_flag and (not exists(sess_seg_filepath) or force_flag):
            seg_list = []
            seg = np.asarray(proxyseg.dataobj)
            for it_c in range(0, len(labels_lut), channel_chunk):
                cat_chunk = {k: np.mod(it_d, channel_chunk) for it_d, k in enumerate(labels_lut.keys()) if
                             it_d >= it_c and it_d < it_c + channel_chunk}
                seg_onehot = fn_utils.one_hot_encoding(seg, categories=cat_chunk).astype('float32', copy=False)
                proxyflo_seg = nib.Nifti1Image(seg_onehot, v2r_seg)
                seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxysvf=proxysvf)  # ,mode='nearest')

                # In case the last iteration has only 1 channel (it squeezes due to batch dimension)
                if len(seg_mri.shape) == 3:
                    seg_list += [np.array(seg_mri.dataobj)[..., np.newaxis]]
                else:
                    seg_list += [np.array(seg_mri.dataobj)]
                del seg_mri

            seg_post = np.concatenate(seg_list, axis=-1)
            seg_mri = nib.Nifti1Image(seg_post, proxyref.affine)
            nib.save(seg_mri, sess_seg_filepath)
            output_dict[svf_ent['session']]['dseg'] = seg_post

        elif seg_flag and exists(sess_seg_filepath):
            im_mri = nib.load(sess_seg_filepath)
            output_dict[svf_ent['session']]['dseg'] = np.array(im_mri.dataobj)

    return output_dict

def process_template(subject, bids_loader, force_flag=False):
    im_ent = {'scope': basename(DIR_PIPELINES['uslr-lin']), 'space': 'SUBJECT', 'acquisition': 1,
              'subject': subject, 'suffix': 'T1w', 'extension': 'nii.gz'}
    sub_str = 'sub-' + subject
    dir_nonlin_subj = join(DIR_PIPELINES['uslr-nonlin'], sub_str)

    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x: len(bids_loader.get(session=x, **im_ent)) > 0, timepoints))

    linear_template = {}
    for file in bids_loader.get(subject=subject, desc='linTemplate', extension='nii.gz'):
        if 'dseg' in file.entities['suffix']:
            linear_template['dseg'] = file
        elif 'mask' in file.entities['suffix']:
            linear_template['mask'] = file
        elif file.entities['suffix'] == 'T1w':
            linear_template['image'] = file

    if len(linear_template.keys()) == 0:
        print('[warning] Linear template does not exist for subject ' + subject + '. Skipping.')
        return subject

    fname_template = 'sub-' + subject + '_desc-nonlinTemplate_T1w'
    nonlinear_template_fpath = {
        'image': join(dir_nonlin_subj, fname_template + '.nii.gz'),
        'mask': join(dir_nonlin_subj, fname_template + 'mask.nii.gz'),
        'dseg': join(dir_nonlin_subj, fname_template + 'dseg.nii.gz'),
        'std': join(dir_nonlin_subj, fname_template + 'dseg.nii.gz'),
    }

    for im_type, im_file in linear_template.items():
        if im_type != 'image':
            subprocess.call(['rm', '-rf', im_file.path])

    if len(timepoints) == 1:
        if not exists(nonlinear_template_fpath['image']) and exists(linear_template['image'].path):
            subprocess.call(['cp', linear_template['image'].path, nonlinear_template_fpath['image']])
        if not exists(nonlinear_template_fpath['mask']) and exists(linear_template['mask'].path):
            subprocess.call(['cp', linear_template['mask'].path, nonlinear_template_fpath['mask']])
        if not exists(nonlinear_template_fpath['dseg']) and exists(linear_template['dseg'].path):
            subprocess.call(['cp', linear_template['dseg'].path, nonlinear_template_fpath['dseg']])

        print('[done] It has only 1 timepoint.')
        return

    if (exists(nonlinear_template_fpath['image']) and not force_flag) and \
            (seg_flag and exists(nonlinear_template_fpath['dseg']) and not force_flag):
        print('[done] It has already been processed. ')
        return

    proxyref = nib.load(linear_template['image'])
    output_d = process_timepoints(subject, timepoints, bids_loader, proxyref, force_flag=force_flag)
    if not isinstance(output_d, dict):
        return subject

    mri_list = [output_d[tp]['im'] for tp in timepoints]
    if len(mri_list) > 0:
        template = np.median(mri_list, axis=0)
        template_std = np.std(mri_list, axis=0)
        img = nib.Nifti1Image(template, nib.load(linear_template['image']).affine)
        nib.save(img, nonlinear_template_fpath['image'])

        img = nib.Nifti1Image(template_std, nib.load(linear_template['image']).affine)
        nib.save(img, nonlinear_template_fpath['std'])

    mask_list = [output_d[tp]['mask'] for tp in timepoints]
    if len(mask_list) > 0:
        template = np.sum(mask_list, axis=0) / len(mask_list)
        img = nib.Nifti1Image(template, nib.load(linear_template['image']).affine)
        nib.save(img, nonlinear_template_fpath['mask'])

        nonlinear_etiv = nonlinear_template_fpath['mask'].replace('mask', 'etiv')
        nonlinear_etiv = nonlinear_etiv.replace('nii.gz', 'npy')
        np.save(nonlinear_etiv, np.sum(template > 0.5))

    if seg_flag:
        seg_list = np.zeros((len(labels_lut),) + nib.load(nonlinear_template_fpath['image']).shape)
        seg_filelist = [output_d[tp]['dseg'] for tp in timepoints]
        if len(seg_filelist) > 0:
            for seg in seg_filelist:
                seg_list += np.transpose(seg, axes=(3, 0, 1, 2))

            template = np.argmax(seg_list, axis=0)
            template_true = np.zeros_like(template)
            for l, it_l in labels_lut.items(): template_true[template == it_l] = l

            img = nib.Nifti1Image(template_true.astype('int16'), nib.load(linear_template['image']).affine)
            nib.save(img, nonlinear_template_fpath['dseg'])

    return


if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# --------------------------------- #')
    print('# Non-linear SLR: compute template  #')
    print('# --------------------------------- #')
    print('\n\n')


    # Parameters
    parser = ArgumentParser(description='Computes the subject-specific non-linear template.')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2', 'gurobi'], help='Likelihood cost function')
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--factor', type=int, default=2, help="(default=2) downsample factor to run the algorithm.")
    parser.add_argument('--seg_flag', action='store_true')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()
    bids_dir = args.bids
    cost = args.cost
    initial_subject_list = args.subjects
    seg_flag = args.seg_flag
    force_flag = args.force
    factor = 2
    labels_lut = labels.SYNTHSEG_APARC_LUT

    print('Loading dataset ...\n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['uslr-nonlin'])
    bids_loader.add_derivatives(DIR_PIPELINES['uslr-lin'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if args.subjects is None else args.subjects



    for it_subject, subject in enumerate(subject_list):

        failed_subjects = []
        for it_subject, subject in enumerate(subject_list):
            t_init = time.time()
            try:
                print('* Subject: ' + subject + '. (' + str(it_subject + 1) + '/' + str(len(subject_list)) + ').')
                ms = process_template(subject, bids_loader, force_flag=force_flag)
            except:
                ms = subject

            if ms is not None:
                failed_subjects.append(ms)

            print('  Total Elapsed time: ' + str(time.time() - t_init))

        f = open(join(LOGS_DIR, 'compute_template_nonlinear.txt'), 'w')
        f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
        f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

        print('\n')
        print('Total failed subjects ' + str(len(failed_subjects)) +
              '. See ' + join(LOGS_DIR, 'compute_template_nonlinear.txt') + ' for more information.')
        print('\n')
        print('# --------- FI (USLR-NONLIN: compute latent template) --------- #')
        print('\n')


