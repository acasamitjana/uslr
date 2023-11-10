import csv
import os
import pdb
import statistics
from os.path import exists, join, basename, dirname
import time
import shutil
import copy

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.special import softmax
from scipy.stats import norm

from setup import *
from utils.labels import *
from utils import io_utils, fn_utils,  def_utils


class LabelFusion(object):

    def __init__(self, bids_loader, def_scope='slr-nonlin', seg_scope='synthseg', output_scope='slr-nonlin',
                 temp_variance=None,  time_marker='time_to_bl_days', spatial_variance=None, normalise=True,
                 smooth=False, type_map=None, fusion_method='post', all_labels_flag=None, save_seg=False):

        self.bids_loader = bids_loader
        self.seg_scope = seg_scope
        self.def_scope = def_scope
        self.output_scope = output_scope
        self.output_dir = join(dirname(self.bids_loader.root), 'derivatives', output_scope)
        self.temp_variance = temp_variance if temp_variance is not None else [np.inf]
        self.spatial_variance = spatial_variance if spatial_variance is not None else [np.inf]
        self.time_marker = time_marker
        self.smooth = smooth
        self.type_map = type_map
        self.fusion_method = fusion_method
        self.normalise = normalise
        self.save_seg = save_seg

        self.conditions_image_synthseg = {'acquisition': 'orig', 'extension': 'nii.gz', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_image_acq1_synthseg = {'acquisition': '1', 'extension': 'nii.gz', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg_synthseg = {'acquisition': '1', 'suffix': 'T1wdseg', 'scope': seg_scope, 'extension': 'nii.gz'}
        self.conditions_image = {'acquisition': 'orig', 'extension': 'nii.gz', 'suffix': 'T1w', 'scope': seg_scope}
        self.conditions_image_acq1 = {'acquisition': '1', 'extension': 'nii.gz', 'suffix': 'T1w', 'scope': seg_scope}
        self.conditions_seg = {'acquisition': '1', 'suffix': 'dseg', 'scope': seg_scope, 'extension': 'nii.gz'}


        self.all_labels_flag = all_labels_flag
        if seg_scope == 'freesurfer' and all_labels_flag is False:
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_LUT)}
        elif seg_scope == 'synthseg' and all_labels_flag is False:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_ARR)}
        elif seg_scope == 'synthseg' and all_labels_flag:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_AND_APARC_ARR)}
        else:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_ARR)}

        if type_map == 'distance_map':
            self.interpmethod = 'seg-lin-' + fusion_method
        elif type_map == 'onehot_map':
            self.interpmethod = 'onehot-lin-' + fusion_method
        elif type_map == 'gauss_map':
            self.interpmethod = 'gauss-lin-' + fusion_method
        else:
            self.interpmethod = 'post-lin-' + fusion_method

        self.chunk_size = (192, 192, 192)
        self.channel_chunk = 20

    def get_chunk_list(self, proxyimage):

        volshape = proxyimage.shape

        nchunks = [int(np.ceil(volshape[it_d] / self.chunk_size[it_d])) for it_d in range(3)]
        chunk_list = []
        for x in range(nchunks[0]):
            for y in range(nchunks[1]):
                for z in range(nchunks[2]):
                    max_x = min((x + 1) * self.chunk_size[0], volshape[0])
                    max_y = min((y + 1) * self.chunk_size[1], volshape[1])
                    max_z = min((z + 1) * self.chunk_size[2], volshape[2])
                    chunk_list += [[[x * self.chunk_size[0], max_x],
                                    [y * self.chunk_size[1], max_y],
                                    [z * self.chunk_size[2], max_z]]]

        return chunk_list

    def prepare_data(self, subject, force_flag):

        print('\n  o Reading the input files')
        timepoints = self.bids_loader.get_session(subject=subject)
        sess_tsv = self.bids_loader.get(suffix='sessions', extension='tsv', subject=subject)

        sess_df = sess_tsv[0].get_df()
        sess_df = sess_df.set_index('session_id')
        sess_df = sess_df[~sess_df.index.duplicated(keep='last')]

        if any([np.isnan(float(sess_df.loc[t][self.time_marker])) for t in timepoints]):
            timepoints_tmp = list(filter(lambda t: not np.isnan(float(sess_df.loc[t][self.time_marker])), timepoints))
            if len(timepoints_tmp) > 0:
                t_tmp = timepoints_tmp[0]
                known_age = float(sess_df.loc[t_tmp]['age'])
                known_month = 0 if t_tmp == 'bl' else int(t_tmp[1:])
                for t in timepoints:
                    if t in timepoints_tmp: continue
                    month = 0 if t == 'bl' else int(t[1:])
                    sess_df.at[t, 'time_to_bl_days'] = month*365/12
                    sess_df.at[t, 'time_to_bl_months'] = month
                    sess_df.at[t, 'time_to_bl_years'] = month/12
                    sess_df.at[t, 'age'] = known_age + (month - known_month)/12

                sess_df.to_csv(sess_tsv[0], sep='\t')

        if not (all([tv == np.inf for tv in self.temp_variance])):
            timepoints = list(filter(lambda t: not np.isnan(float(sess_df.loc[t][self.time_marker])), timepoints))



        timepoints = list(filter(lambda t: len(self.bids_loader.get(**{**{'subject': subject, 'session': t}, **self.conditions_seg}, regex_search=True)) == 1, timepoints))
        timepoints = list(filter(lambda t: len(self.bids_loader.get(**{**{'subject': subject, 'session': t}, **self.conditions_image_synthseg}, regex_search=False)) == 1, timepoints))

        if len(timepoints) == 1: return None, None, None

        if self.def_scope == 'slr-nonlin':
            for tp in timepoints:
                svf_files = self.bids_loader.get(**{'subject': subject, 'suffix': 'svf', 'scope': 'slr-nonlin', 'session': tp, 'extension': 'nii.gz'})
                if len(svf_files) != 1:
                    print('WARNING. No SVFs have been found.')
                    return None, None, None

        elif self.def_scope == 'slr-lin':
            for tp in timepoints:
                aff_files = self.bids_loader.get(**{'subject': subject, 'suffix': 'T1w', 'desc': 'aff', 'scope': 'slr-lin', 'session': tp, 'extension': 'npy'})
                if len(aff_files) != 1:
                    print('WARNING. No AFFINE has been found.')
                    return None, None, None
        vol_tsv = {}
        for tp in timepoints:
            seg_tsv = self.bids_loader.get(
                **{'subject': subject, 'suffix': 'vols', 'scope': self.output_scope, 'extension': 'tsv', 'session': tp}
            )
            if len(seg_tsv) == 1:
                seg_df = seg_tsv[0].get_df()
                seg_df = seg_df.set_index(['t_var', 's_var', 'interpmethod'])
                seg_df = seg_df[~seg_df.index.duplicated(keep='last')]
                vol_tsv[tp] = seg_df
            else:
                continue

        timepoints_to_run = []
        if force_flag:
            timepoints_to_run = timepoints
        elif DEBUG:
            timepoints_to_run = timepoints[0:1]
        else:
            for tp in timepoints:
                if tp not in vol_tsv.keys():
                    timepoints_to_run.append(tp)
                else:
                    if not self.spatial_variance and not self.temp_variance:
                        timepoints_to_run.append(tp)
                    elif all([len(vol_tsv[tp].loc[t, s, self.interpmethod]) == 1 for t in self.temp_variance for s in self.spatial_variance]):
                        timepoints_to_run.append(tp)

        if not timepoints_to_run:
            print('Subject: ' + str(subject) + '. DONE')
            return None, None, None

        if all([tv == np.inf for tv in self.temp_variance]):
            time_list = {tp: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                time_list[tp] = float(sess_df.loc[tp][self.time_marker])


        return timepoints_to_run, timepoints, time_list

    def register_timepoints_st(self, subject, tp_ref, tp_flo, proxyref, proxyimage_flo=None, proxyseg_flo=None,
                               im_mode='bilinear', seg_mode='bilinear'):
        '''
        :param subject:
        :param tp_ref:
        :param tp_flo:
        :param image_flo:
        :param seg_flo:
        :param im_mode: 'bilinear', 'nearest'.
        :param seg_mode: 'bilinear', 'nearest'.
        :return:
        '''

        # Ref parameters
        aff_dict = {'subject': subject, 'desc': 'aff', 'suffix': 'T1w', 'scope': 'slr-lin', 'extension': 'npy'}
        svf_dict = {'subject': subject, 'suffix': 'svf', 'scope': 'slr-nonlin', 'extension': 'nii.gz'}

        affine_file_ref = self.bids_loader.get(**{**aff_dict, 'session': tp_ref})
        if len(affine_file_ref) != 1:
            print('WARNING: no affine file found for REF image')
            return None, None
        affine_matrix_ref = np.load(affine_file_ref[0].path)
        svf_filename_ref = self.bids_loader.get(**{**svf_dict, 'session': tp_ref})


        # Flo parameters
        affine_file_flo = self.bids_loader.get(**{**aff_dict, 'session': tp_flo})
        if len(affine_file_flo) != 1:
            print('WARNING: no affine file found for FLO image')
            return None, None
        affine_matrix_flo = np.load(affine_file_flo[0].path)
        svf_filename_flo = self.bids_loader.get(**{**svf_dict, 'session': tp_flo})

        proxyflow = None
        if self.def_scope != 'slr-lin':
            if len(svf_filename_flo) != 1 or len(svf_filename_ref) != 1:
                print('WARNING: check SVF files and re-run again.')
                return None, None
            tp_svf = nib.load(svf_filename_ref[0].path)
            tp_flo_svf = nib.load(svf_filename_flo[0].path)

            svf = np.asarray(tp_flo_svf.dataobj) - np.asarray(tp_svf.dataobj)
            proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
            proxyflow = def_utils.integrate_svf(proxysvf)

        # Deform
        v2r_ref = np.matmul(np.linalg.inv(affine_matrix_ref), proxyref.affine)
        proxyref_aligned = nib.Nifti1Image(np.zeros(proxyref.shape), v2r_ref)
        im_resampled = None
        seg_resampled = None

        if proxyimage_flo is not None:
            v2r_target = np.matmul(np.linalg.inv(affine_matrix_flo), proxyimage_flo.affine)
            proxyflo_im = nib.Nifti1Image(np.array(proxyimage_flo.dataobj), v2r_target)
            im_mri = def_utils.vol_resample(proxyref_aligned, proxyflo_im, proxyflow=proxyflow, mode=im_mode)

            im_resampled = np.array(im_mri.dataobj)
            del im_mri

        if proxyseg_flo is not None:
            v2r_seg_target = np.matmul(np.linalg.inv(affine_matrix_flo), proxyseg_flo.affine)

            if len(proxyseg_flo.shape) > 3:
                seg_list = []
                for it_c in range(0, proxyseg_flo.shape[-1], self.channel_chunk):
                    proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj[..., it_c:it_c+self.channel_chunk]), v2r_seg_target)
                    seg_mri = def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode)

                    # In case the last iteration has only 1 channel (it squeezes due to batch dimension)
                    if len(seg_mri.shape) == 3:
                        seg_list += [np.array(seg_mri.dataobj)[..., np.newaxis]]
                    else:
                        seg_list += [np.array(seg_mri.dataobj)]

                    del seg_mri

                seg_resampled = np.concatenate(seg_list, axis=-1)
            else:
                proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj), v2r_seg_target)
                seg_resampled = np.array(def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode).dataobj)

        return im_resampled, seg_resampled

    def compute_seg_map(self, proxyimage, proxyseg, distance=False):

        v2r_ref = proxyseg.affine.copy()
        if self.type_map == 'distance_map':
            proxyimage_flo = def_utils.vol_resample(proxyseg, proxyimage)
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            if not self.all_labels_flag:
                seg[seg > 1999] = 42
                seg[seg > 999] = 3
            if distance:
                seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32', copy=False)

        elif self.type_map == 'onehot_map':
            proxyimage_flo = def_utils.vol_resample(proxyseg, proxyimage)
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            if not self.all_labels_flag:
                seg[seg > 1999] = 42
                seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)

        elif self.type_map == 'gauss_map':
            proxyimage_flo = def_utils.vol_resample(proxyseg, proxyimage)
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            if not self.all_labels_flag:
                seg[seg > 1999] = 42
                seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)
            for it_label in range(seg.shape[-1]):
                seg[..., it_label] = gaussian_filter(seg[..., it_label], sigma=1)

            seg = softmax(seg, axis=-1)
        else:
            seg = np.array(proxyseg.dataobj)
            mask = np.sum(seg[..., 1:], -1) > 0

        mask, crop_coord = fn_utils.crop_label(mask > 0, margin=10, threshold=0)
        image = fn_utils.apply_crop(np.array(proxyimage_flo.dataobj), crop_coord)
        seg = fn_utils.apply_crop(seg, crop_coord)
        v2r_ref[:3, 3] = v2r_ref[:3] @ np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0], 1])

        proxyim_ref = nib.Nifti1Image(image, v2r_ref)
        proxyseg_ref = nib.Nifti1Image(seg, v2r_ref)

        return proxyim_ref, proxyseg_ref

    def process_tp(self, subject, tp_ref, tp_flo, proxyim_ref):

        im_file = self.bids_loader.get(**{**{'subject': subject, 'session': tp_flo}, **self.conditions_image_acq1})
        im_file_orig = self.bids_loader.get(**{**{'subject': subject, 'session': tp_flo}, **self.conditions_image})
        seg_file = self.bids_loader.get(**{**{'subject': subject, 'session': tp_flo}, **self.conditions_seg})
        if len(im_file) == 0:
            im_file = self.bids_loader.get(**{**{'subject': subject, 'session': tp_flo}, **self.conditions_image_acq1_synthseg})
        if len(im_file_orig) == 0:
            im_file_orig = self.bids_loader.get(**{**{'subject': subject, 'session': tp_flo}, **self.conditions_image_synthseg})

        if len(seg_file) == 0:
            seg_file = self.bids_loader.get(**{**{'subject': subject, 'session': tp_flo}, **self.conditions_seg_synthseg})

        im_file = im_file[0]
        seg_file = seg_file[0]

        proxyimage, proxyseg = self.compute_seg_map(nib.load(im_file.path), nib.load(seg_file.path), distance=False)

        if len(im_file_orig) == 0:
            print(' - WARNING: original acquisition image is not found in ' + self.seg_scope + ' nor in the SynthSeg directory.')
        else:
            im_file_orig = im_file_orig[0]
            proxyimage = nib.load(im_file_orig)

            pixdim = np.sqrt(np.sum(proxyimage.affine * proxyimage.affine, axis=0))[:-1]
            new_vox_size = np.sqrt(np.sum(proxyseg.affine * proxyseg.affine, axis=0))[:-1]
            factor = pixdim / new_vox_size
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            if not all(sigmas == 0):
                image = gaussian_filter(np.array(proxyimage.dataobj), sigmas)
                proxyimage = nib.Nifti1Image(image, proxyimage.affine)


        if all([s == np.inf for s in self.spatial_variance]):
            proxyimage = None

        if self.type_map == 'distance_map':
            seg_mode = 'distance'
        else:
            seg_mode = 'bilinear'

        im_resampled, seg_resampled = self.register_timepoints_st(subject, tp_ref, tp_flo, proxyim_ref,
                                                                  proxyimage_flo=proxyimage, proxyseg_flo=proxyseg,
                                                                  seg_mode=seg_mode)

        return im_resampled, seg_resampled

    def compute_p_label(self, subject, timepoints, tp, chunk, proxyimage, proxyseg, mode='linear'):

        v2r_ref = proxyimage.affine

        p_data = np.zeros((chunk[0][1]-chunk[0][0], chunk[1][1]-chunk[1][0], chunk[2][1]-chunk[2][0], len(timepoints),), dtype='float32')
        p_label = np.zeros((chunk[0][1]-chunk[0][0], chunk[1][1]-chunk[1][0], chunk[2][1]-chunk[2][0], len(self.labels_lut), len(timepoints)), dtype='float32')

        im = np.array(proxyimage.dataobj)[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]]
        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp == tp_flo:
                seg_resampled = np.array(proxyseg.dataobj)[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]]
                im_resampled = im

            else:
                im_resampled, seg_resampled = self.process_tp(subject, tp, tp_flo, proxyimage)
                im_resampled = im_resampled[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]]
                seg_resampled = seg_resampled[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]]

            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            if self.fusion_method == 'seg':
                num_classes = seg_resampled.shape[-1]
                seg_resampled_max = np.argmax(seg_resampled, axis=-1)
                seg_resampled = fn_utils.one_hot_encoding(seg_resampled_max, num_classes=num_classes).astype('float32', copy=False)


            p_data[..., it_tp_flo] = im_resampled
            p_label[..., it_tp_flo] = seg_resampled
            del im_resampled, seg_resampled

        if self.smooth:
            return (gaussian_filter(im[..., np.newaxis] - p_data, sigma=[0.5, 0.5, 0.5, 0]))**2, p_label, v2r_ref
        else:
            return (im[..., np.newaxis] - p_data)**2, p_label, v2r_ref

    def label_fusion(self, subject, force_flag=False, *args, **kwargs):

        print('Subject: ' + str(subject), end=' ', flush=True)

        attach_overwrite = 'a'

        timepoints_to_run, timepoints, time_list = self.prepare_data(subject, force_flag)
        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        for tp in timepoints_to_run:
            print('        - Timepoint ' + tp, end=': ', flush=True)
            t_0 = time.time()

            im_file = self.bids_loader.get(**{**{'subject': subject, 'session': tp}, **self.conditions_image})
            seg_file = self.bids_loader.get(**{**{'subject': subject, 'session': tp}, **self.conditions_seg})
            if len(im_file) == 0:
                im_file = self.bids_loader.get(**{**{'subject': subject, 'session': tp}, **self.conditions_image_synthseg})
            if len(seg_file) == 0:
                seg_file = self.bids_loader.get(**{**{'subject': subject, 'session': tp}, **self.conditions_seg_synthseg})
            im_file = im_file[0]
            seg_file = seg_file[0]

            proxyimage, proxyseg = nib.load(im_file.path), nib.load(seg_file.path)

            pixdim = np.sqrt(np.sum(proxyimage.affine * proxyimage.affine, axis=0))[:-1]
            new_vox_size = np.sqrt(np.sum(proxyseg.affine * proxyseg.affine, axis=0))[:-1]
            factor = pixdim / new_vox_size
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            if not all(sigmas == 0):
                image = gaussian_filter(np.array(proxyimage.dataobj), sigmas)
                proxyimage = nib.Nifti1Image(image, proxyimage.affine)

            proxyimage, proxyseg = self.compute_seg_map(proxyimage, proxyseg, distance=False)

            ent_seg = {
                'subject': subject,
                'session': tp,
                'extension': 'nii.gz',
                'suffix': 'dseg'
            }

            st_vols = []
            st_vols_dict = {t_var: {s_var: {k: 0 for k in self.labels_lut.keys()} for s_var in self.spatial_variance} for t_var in self.temp_variance}
            st_vols_dict_norm = {t_var: {s_var: {k: 0 for k in self.labels_lut.keys()} for s_var in self.spatial_variance} for t_var in self.temp_variance}

            chunk_list = self.get_chunk_list(proxyimage)
            for it_chunk, chunk in enumerate(chunk_list):
                if it_chunk == len(chunk_list) - 1:
                    print(str(it_chunk) + '/' + str(len(chunk_list)), end='. ', flush=True)
                else:
                    print(str(it_chunk) + '/' + str(len(chunk_list)), end=', ', flush=True)

                mean_im_2, seg_res, aff = self.compute_p_label(subject, timepoints, tp, chunk, proxyimage, proxyseg)


                time_arr = np.array([v for v in time_list.values()])
                mean_age_2 = (time_arr - time_list[tp]) ** 2

                pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
                for t_var in self.temp_variance:
                    t_ker = 1 if t_var in [np.inf, 0] else np.exp(-0.5 / t_var * mean_age_2)
                    for s_var in self.spatial_variance:
                        s_ker = np.ones_like(mean_im_2) if s_var in [np.inf, 0] else np.exp(-0.5 / s_var * mean_im_2)
                        p_data = s_ker * t_ker
                        del s_ker

                        p_label = np.zeros(seg_res.shape[:-1])
                        if float(s_var) > 0 and float(t_var) > 0:
                            for it_t in range(seg_res.shape[-1]):
                                p_label += p_data[..., np.newaxis, it_t] * seg_res[..., it_t]#, axis=-1)
                        else:
                            it_ref = [it_t for it_t, t in enumerate(timepoints) if t==tp][0]
                            p_label = seg_res[..., it_ref]

                        del p_data

                        if self.normalise:
                            p_label = p_label / np.sum(p_label, axis=-1, keepdims=True)
                            p_label[np.isnan(p_label)] = 0


                        fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                        if self.save_seg:
                            filename_seg = self.bids_loader.build_path(
                                {**ent_seg, 'desc': 't' + str(t_var) + 's' + str(s_var) + self.interpmethod},
                                scope=self.output_scope, path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                absolute_paths=False)

                            true_vol = np.zeros_like(fake_vol)
                            for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul
                            if it_chunk == 0:
                                true_final_vol = np.zeros(proxyimage.shape)
                            else:
                                proxytmp = nib.load(join(self.output_dir, subject, tp, 'anat', filename_seg))
                                true_final_vol = np.array(proxytmp.dataobj)

                            true_final_vol[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]] = true_vol
                            img = nib.Nifti1Image(true_final_vol, aff)
                            nib.save(img, join(self.output_dir, filename_seg))

                            del true_final_vol, true_vol

                        if self.normalise:
                            vols = get_vols_post(p_label, res=pixdim)
                            st_vols_dict_norm[t_var][s_var] = {k: st_vols_dict_norm[t_var][s_var][k] + vols[v] for k, v in self.labels_lut.items()}

                        vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                        st_vols_dict[t_var][s_var] = {k: st_vols_dict[t_var][s_var][k] + vols[v] for k, v in self.labels_lut.items()}

                        del p_label, fake_vol

                del mean_im_2, seg_res

            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:

                    if self.normalise:
                        st_d = st_vols_dict_norm[t_var][s_var]
                        st_d['id'] = im_file.filename[:-7]
                        st_d['t_var'] = t_var
                        st_d['s_var'] = s_var
                        st_d['interpmethod'] = self.interpmethod
                        st_d['type'] = 'posteriors'
                        st_vols += [st_d]

                    st_d = st_vols_dict[t_var][s_var]
                    st_d['id'] = im_file.filename[:-7]
                    st_d['t_var'] = t_var
                    st_d['s_var'] = s_var
                    st_d['interpmethod'] = self.interpmethod
                    st_d['type'] = 'seg'
                    st_vols += [st_d]

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(self.output_dir, 'sub-' + subject, 'ses-' + tp, 'anat', 'sub-' + subject + '_ses-' + tp + '_vols.tsv')
            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject) + '. DONE')


class LabelFusionTemplate(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None, normalise=True,
                 interpmethod='linear', save=False, process=False, type_map=None, fusion_method='post', jlf=False,
                 all_labels_flag=None):

        self.seg_scope = seg_scop
        self.def_scope = def_scope
        self.results_scope = def_scope + '-temp' if seg_scop == 'synthseg' else def_scope + '-' + seg_scop + '-temp'
        self.temp_variance = temp_variance if temp_variance is not None else [np.inf]
        self.spatial_variance = spatial_variance if spatial_variance is not None else [np.inf]
        self.floor_variance = floor_variance if floor_variance is not None else [1]
        self.save = save
        self.process = process
        self.smooth = smooth
        self.time_marker = time_marker
        self.type_map = type_map
        self.fusion_method = fusion_method
        self.jlf = jlf
        self.normalise = normalise

        seg_suffix = 'dseg' if type_map is not None else 'post'
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': seg_suffix, 'scope': seg_scop, 'extension': 'nii.gz'}

        self.all_labels_flag = all_labels_flag
        if seg_scop == 'freesurfer' and all_labels_flag is False:
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_LUT)}
        elif seg_scop == 'synthseg' and all_labels_flag is False:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_ARR)}
        elif seg_scop == 'synthseg' and all_labels_flag:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_AND_APARC_ARR)}
        else:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_ARR)}


        if type_map == 'distance_map':
            self.interpmethod = 'seg-' + interpmethod + '-' + fusion_method
        elif type_map == 'onehot_map':
            self.interpmethod = 'onehot-' + interpmethod + '-' + fusion_method
        elif type_map == 'gauss_map':
            self.interpmethod = 'gauss-' + interpmethod + '-' + fusion_method
        else:
            self.interpmethod = 'post-' + interpmethod + '-' + fusion_method

        if jlf: self.interpmethod = self.interpmethod + '-jlf'

        self.channel_chunk = 20

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions
        if not (all([tv == np.inf for tv in self.temp_variance]) or self.save):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        if len(timepoints) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return None, None, None

        if 'sreg' in self.def_scope and 'sreg-lin' not in self.def_scope:
            svf_dict = {'sub': subject.id, 'suffix': 'svf'}
            for tp in timepoints:
                if not exists(join(tp.data_dir[self.def_scope], io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz')):
                    print('Subject: ' + str(subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                    return None, None, None

        if self.save and not force_flag:
            f = timepoints[-1].id + '_to_' + timepoints[-1].id + '_post.nii.gz'
            if (self.type_map is None and exists(join(subject.data_dir[self.def_scope], 'segmentations', f))) or \
                    (self.type_map == 'distance_map' and exists(join(subject.data_dir[self.def_scope], 'segmentations_distance', f))) \
                    or (self.type_map == 'onehot_map' and exists(join(subject.data_dir[self.def_scope], 'segmentations_onehot', f))):
                print('Subject: ' + str(subject.id) + '. DONE')
                return None, None, None
            else:
                return timepoints, timepoints, []
        else:
            vol_tsv = {}
            for tp in timepoints:
                last_dir = tp.data_dir[self.results_scope]
                if exists(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')):
                    vol_tsv[tp.id] = io_utils.read_tsv(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'), k=['id', 't_var', 's_var', 'interpmethod'])

        timepoints_to_run = []
        if force_flag:
            timepoints_to_run = timepoints
        elif DEBUG:
            timepoints_to_run = timepoints[0:1]
        else:
            for tp in timepoints:
                if tp.id not in vol_tsv.keys():
                    timepoints_to_run.append(tp)
                else:
                    filename = tp.get_files(**self.conditions_image)[0]
                    if not self.spatial_variance and not self.temp_variance:
                        timepoints_to_run.append(tp)

                    elif any([(filename[:-7] + '_' + str(t) + 'f' + str(f) + '_' + str(s) + '_' + self.interpmethod not in vol_tsv[tp.id].keys() and f != 1)
                              or (filename[:-7] + '_' + str(t) + '_' + str(s) + '_' + self.interpmethod not in vol_tsv[tp.id].keys() and f == 1) for f in self.floor_variance for t in self.temp_variance for s in self.spatial_variance]):
                        timepoints_to_run.append(tp)

        if not timepoints_to_run:
            print('Subject: ' + str(subject.id) + '. DONE')
            return None, None, None

        if all([tv == np.inf for tv in self.temp_variance]) or self.save:
            time_list = {tp.id: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                time_list[tp.id] = float(tp.sess_metadata[self.time_marker])

        return timepoints_to_run, timepoints, time_list

    def register_timepoints_st(self, subject, tp, proxyref, proxyim_flo=None, proxyseg_flo=None, im_mode='bilinear',
                               seg_mode='bilinear'):
        '''
        :param subject:
        :param tp:
        :param image_flo:
        :param seg_flo:
        :param im_mode: 'bilinear', 'nearest'.
        :param seg_mode: 'bilinear', 'nearest'.
        :return:
        '''

        # Def parameters
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        # Timepoint parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}

        affine_matrix = np.load(join(tp.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'))
        svf_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'

        if self.def_scope != 'sreg-lin':
            proxysvf = nib.load(join(tp.data_dir[self.def_scope], svf_filename))
            proxyflow = def_utils.integrate_svf(proxysvf)

        else:
            proxyflow = None

        # Deform
        im_resampled = None
        seg_resampled = None

        if proxyim_flo is not None:
            v2r_target = np.matmul(np.linalg.inv(affine_matrix), proxyim_flo.affine)
            proxyflo_im = nib.Nifti1Image(np.array(proxyim_flo.dataobj), v2r_target)
            im_mri = def_utils.vol_resample(proxyref, proxyflo_im, proxyflow=proxyflow, mode=im_mode)

            im_resampled = np.array(im_mri.dataobj)
            del im_mri

        if proxyseg_flo is not None:
            v2r_seg_target = np.matmul(np.linalg.inv(affine_matrix), proxyseg_flo.affine)
            if len(proxyseg_flo.shape) > 3:
                seg_list = []
                for it_c in range(0, proxyseg_flo.shape[-1], self.channel_chunk):
                    proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj[..., it_c:it_c+self.channel_chunk]), v2r_seg_target)
                    seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode)

                    #In case the last iteration has only 1 channel (it squeezes due to batch dimension)
                    if len(seg_mri.shape) == 3: seg_list += [np.array(seg_mri.dataobj)[..., np.newaxis]]
                    else: seg_list += [np.array(seg_mri.dataobj)]

                    del seg_mri

                seg_resampled = np.concatenate(seg_list, axis=-1)
            else:
                proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj), v2r_seg_target)
                seg_resampled = np.array(def_utils.vol_resample(proxyref, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode).dataobj)

        return im_resampled, seg_resampled

    def compute_seg_map(self, tp, distance=True):
        proxyimage = tp.get_image(**self.conditions_image)
        proxyseg = tp.get_image(**self.conditions_seg)
        v2r_ref = proxyimage.affine.copy()

        if self.type_map == 'distance_map':
            proxyseg = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            if not self.all_labels_flag:
                seg[seg > 1999] = 42
                seg[seg > 999] = 3
            if distance:
                seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32',
                                                                                                            copy=False)

        elif self.type_map == 'onehot_map':
            proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg_flo.dataobj)
            mask = seg > 0
            if not self.all_labels_flag:
                seg[seg > 1999] = 42
                seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)

        elif self.type_map == 'gauss_map':
            from scipy.ndimage import gaussian_filter
            proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg_flo.dataobj)
            mask = seg > 0
            if not self.all_labels_flag:
                seg[seg > 1999] = 42
                seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)
            for it_label in range(seg.shape[-1]):
                seg[..., it_label] = gaussian_filter(seg[..., it_label], sigma=1)

            seg = softmax(seg, axis=-1)
        else:
            proxyseg = def_utils.vol_resample(proxyimage, proxyseg)
            seg = np.array(proxyseg.dataobj)
            mask = np.sum(seg[..., 1:], -1) > 0

        mask, crop_coord = fn_utils.crop_label(mask > 0, margin=10, threshold=0)
        image = fn_utils.apply_crop(np.array(proxyimage.dataobj), crop_coord)
        seg = fn_utils.apply_crop(seg, crop_coord)
        v2r_ref[:3, 3] = v2r_ref[:3] @ np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0], 1])

        proxyim_ref = nib.Nifti1Image(image, v2r_ref)
        proxyseg_ref = nib.Nifti1Image(seg, v2r_ref)

        return proxyim_ref, proxyseg_ref

    def process_tp(self, subject, tp, proxyref):

        proxyimage, proxyseg = self.compute_seg_map(tp, distance=False)
        if all([s == np.inf for s in self.spatial_variance]): proxyimage = None

        seg_mode = 'bilinear'
        if self.type_map == 'distance_map': seg_mode = 'distance'

        im_resampled, seg_resampled = self.register_timepoints_st(subject, tp, proxyref, proxyim_flo=proxyimage, proxyseg_flo=proxyseg, seg_mode=seg_mode)

        return im_resampled, seg_resampled

    def compute_p_label(self, tp, subject, timepoints):
        proxyref = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)
        v2r_ref = subject.vox2ras0

        p_data = np.zeros(proxyref.shape + (len(timepoints),), dtype='float32')
        p_label = np.zeros(proxyref.shape + (len(self.labels_lut), len(timepoints)), dtype='float32')

        for it_tp_flo, tp_flo in enumerate(timepoints):
            im_resampled, seg_resampled = self.process_tp(subject, tp_flo, proxyref)

            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            if self.fusion_method == 'seg':
                num_classes = seg_resampled.shape[-1]
                seg_resampled_max = np.argmax(seg_resampled, axis=-1)
                seg_resampled = fn_utils.one_hot_encoding(seg_resampled_max, num_classes=num_classes).astype('float32', copy=False)

            p_data[..., it_tp_flo] = im_resampled
            p_label[..., it_tp_flo] = seg_resampled

            if tp.id == tp_flo.id:
                im = im_resampled
                # im = gaussian_filter(im, sigma=0.5)

            del im_resampled, seg_resampled

        return (im[..., np.newaxis] - p_data)**2, p_label, v2r_ref

    def label_fusion(self, subject, force_flag=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'#'w' if force_flag else 'a'
        timepoints_to_run, timepoints, time_list = self.prepare_data(subject, force_flag)

        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        for tp in timepoints_to_run:
            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)

            t_0 = time.time()
            print('        - Timepoint ' + tp.id, end=':', flush=True)
            mean_im_2, seg_res, aff = self.compute_p_label(tp, subject, timepoints)
            time_arr = np.array([v for v in time_list.values()])
            mean_age_2 = (time_arr - time_list[tp.id]) ** 2

            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            st_vols = []
            for t_var in self.temp_variance:
                t_ker = 1 if t_var in [np.inf, 0] else np.exp(-0.5 / t_var * mean_age_2)
                for s_var in self.spatial_variance:
                    s_ker = np.ones_like(mean_im_2) if s_var in [np.inf, 0] else np.exp(-0.5 / s_var * mean_im_2)
                    p_data = s_ker * t_ker
                    del s_ker

                    if self.normalise:
                        p_data = p_data/np.sum(p_data, axis=-1, keepdims=True)
                        p_data[np.isnan(p_data)] = 0

                    p_label = np.zeros(seg_res.shape[:-1])
                    if float(s_var) > 0 and float(t_var) > 0:
                        for it_t in range(seg_res.shape[-1]):
                            p_label += p_data[..., np.newaxis, it_t] * seg_res[..., it_t]#, axis=-1)
                    else:
                        it_ref = [it_t for it_t, t in enumerate(timepoints) if t.id==tp.id][0]
                        p_label = seg_res[..., it_ref]

                    del p_data

                    fp_dict['suffix'] = 'dseg'
                    fp_dict['desc'] = 't' + str(t_var) + 's' + str(s_var) + self.interpmethod
                    filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                    fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                    # pdb.set_trace()
                    # true_vol = np.zeros_like(fake_vol)
                    # for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul
                    # img = nib.Nifti1Image(true_vol, aff)
                    # nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                    if self.normalise:
                        vols = get_vols_post(p_label, res=pixdim)
                        st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                        st_vols_dict['id'] = filename[:-7]
                        st_vols_dict['t_var'] = t_var
                        st_vols_dict['s_var'] = s_var
                        st_vols_dict['interpmethod'] = self.interpmethod
                        st_vols_dict['type'] = 'posteriors'

                        st_vols += [st_vols_dict]

                    vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                    st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                    st_vols_dict['id'] = filename[:-7]
                    st_vols_dict['t_var'] = t_var
                    st_vols_dict['s_var'] = s_var
                    st_vols_dict['interpmethod'] = self.interpmethod
                    st_vols_dict['type'] = 'seg'

                    st_vols += [st_vols_dict]
                    del p_label, fake_vol#, true_vol

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del mean_im_2, seg_res
            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')

class LabelFusionDirect(LabelFusion):

    def register_timepoints_st(self, subject, tp_ref, tp_flo, proxyref, proxyimage_flo=None, proxyseg_flo=None,
                               im_mode='bilinear', seg_mode='bilinear'):
        '''
        :param subject:
        :param tp_ref:
        :param tp_flo:
        :param image_flo:
        :param seg_flo:
        :param im_mode: 'bilinear', 'nearest'.
        :param seg_mode: 'bilinear', 'nearest'.
        :return:
        '''
        # Ref parameters
        aff_dict_ref = {'sub': subject.id, 'desc': 'aff'}
        # run_flag_ref = any(['run' in f for f in tp_ref.files['bids']])
        # if run_flag_ref: aff_dict_ref['run'] = '01'
        aff_filename_ref = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict_ref}) + '.npy'
        affine_matrix_ref = np.load(join(tp_ref.data_dir['sreg-lin'], aff_filename_ref))

        # Flo parameters
        aff_dict_flo = {'sub': subject.id, 'desc': 'aff'}
        # run_flag_flo = any(['run' in f for f in tp_flo.files['bids']])
        # if run_flag_flo: aff_dict_flo['run'] = '01'
        aff_filename_flo = io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_flo}) + '.npy'
        affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], aff_filename_flo))

        # Load deformations
        if self.def_scope == 'lin':
            aff_f_r = join(subject.data_dir['sreg-' + self.def_scope], 'deformations', tp_ref.id + '_to_' + tp_flo.id + '.aff')
            aff_f_r_rev = join(subject.data_dir['sreg-' + self.def_scope], 'deformations', tp_flo.id + '_to_' + tp_ref.id + '.aff')
            proxyflow = None
            affine_matrix_flo = np.eye(4)
            if exists(aff_f_r):
                affine_matrix_ref = io_utils.read_affine_matrix(aff_f_r, full=True)
            else:
                affine_matrix_ref = io_utils.read_affine_matrix(aff_f_r_rev, full=True)
                affine_matrix_ref = np.linalg.inv(affine_matrix_ref)

        else:
            svf_filename_ref = tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz'
            svf_filename_ref_rev = tp_flo.id + '_to_' + tp_ref.id + '.svf.nii.gz'
            if exists(join(subject.data_dir['sreg-' + self.def_scope], 'deformations', svf_filename_ref)):
                proxysvf = nib.load(join(subject.data_dir['sreg-' + self.def_scope], 'deformations', svf_filename_ref))
                proxyflow = def_utils.integrate_svf(proxysvf)
            else:
                tp_svf = nib.load(join(subject.data_dir['sreg-' + self.def_scope], 'deformations', svf_filename_ref_rev))
                svf = -np.asarray(tp_svf.dataobj)
                proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
                proxyflow = def_utils.integrate_svf(proxysvf)

        # Deform
        v2r_ref = np.matmul(np.linalg.inv(affine_matrix_ref), proxyref.affine)
        proxyref = nib.Nifti1Image(np.zeros(proxyref.shape), v2r_ref)
        im_resampled = None
        seg_resampled = None

        if proxyimage_flo is not None:
            v2r_target = np.matmul(np.linalg.inv(affine_matrix_flo), proxyimage_flo.affine)
            proxyflo_im = nib.Nifti1Image(np.array(proxyimage_flo.dataobj), v2r_target)
            im_mri = def_utils.vol_resample(proxyref, proxyflo_im, proxyflow=proxyflow, mode=im_mode)

            im_resampled = np.array(im_mri.dataobj)
            del im_mri

        if proxyseg_flo is not None:
            v2r_seg_target = np.matmul(np.linalg.inv(affine_matrix_flo), proxyseg_flo.affine)

            seg_list = []
            for it_c in range(0, proxyseg_flo.shape[-1], self.channel_chunk):
                proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj[..., it_c:it_c + self.channel_chunk]),
                                               v2r_seg_target)
                seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode)

                seg_list += [np.array(seg_mri.dataobj)]

                del seg_mri

            seg_resampled = np.concatenate(seg_list, axis=-1)

        return im_resampled, seg_resampled


def write_volume_results(volume_dict, filepath, fieldnames=None, attach_overwrite='a'):
    if fieldnames is None:
        fieldnames = ['id', 't_var', 's_var'] + list(ASEG_APARC_ARR)

    write_header = True if (not exists(filepath) or attach_overwrite == 'w') else False
    with open(filepath, attach_overwrite) as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        if write_header:
            csvwriter.writeheader()
        if isinstance(volume_dict, list):
            csvwriter.writerows(volume_dict)
        else:
            csvwriter.writerow(volume_dict)

def get_vols(seg, res=1, labels=None):
    if labels is None:
        labels = np.unique(seg)

    n_dims = len(seg.shape)
    if isinstance(res, int):
        res = [res] * n_dims
    vol_vox = np.prod(res)

    vols = {}
    for l in labels:
        mask_l = seg == l
        vols[int(l)] = np.round(np.sum(mask_l) * vol_vox, 2)

    return vols

def get_vols_post(post, res=1):
    '''

    :param post: posterior probabilities
    :param res: mm^3 per voxel
    :return:
    '''

    n_labels = post.shape[-1]
    n_dims = len(post.shape[:-1])
    if isinstance(res, int):
        res = [res] * n_dims
    vol_vox = np.prod(res)

    vols = {}
    for l in range(n_labels):
        mask_l = post[..., l]
        mask_l[post[..., l] < 0.05] = 0
        vols[l] = np.round(np.sum(mask_l) * vol_vox, 2)

    return vols


