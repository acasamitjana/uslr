import csv
import os
import pdb
import statistics
from os.path import exists, join
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


class LabelSimpleBayes(object):
    # Currently working with a single gaussian for each region. Need to update the same model as SAMSEG.
    # Register all to generate a SST 'probabilistic atlas' or use synthseg
    # Compute u0 and Sigma_0 by computing the statistics in SST space
    # For each timepoint:
    #    - Compute q as the posterior likelihood*prior /
    #    - Compute model parameters with the computed 'q'
    #    - Update u0 and Sigma_t
    #    - Update ut and Sigma_t
    #

    def __init__(self, def_scope='sreg-synthmorph', seg_scope='synthseg',smooth=None,
                 time_marker='time_to_bl_days',  interpmethod='linear', type_map=None, fusion_method='post',
                 all_labels_flag=None, seg_only=False, init_mode='sst'):

        self.seg_model = 'bayes'
        self.seg_scope = seg_scope
        self.def_scope = def_scope
        self.results_scope = def_scope if seg_scope == 'synthseg' else def_scope + '-' + seg_scope

        self.smooth = smooth
        self.time_marker = time_marker
        self.type_map = type_map
        # self.seg_mode = 'distance' if self.type_map == 'distance_map' else 'bilinear'
        self.fusion_method = fusion_method
        self.seg_only = seg_only

        assert init_mode in ['sst', 'label_fusion']
        self.init_mode = init_mode

        seg_suffix = 'dseg' if type_map is not None else 'post'
        self.conditions_image_synthseg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': seg_scope}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': seg_suffix, 'scope': seg_scope, 'extension': 'nii.gz'}

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
            self.interpmethod = 'seg-' + interpmethod + '-' + fusion_method
        elif type_map == 'onehot_map':
            self.interpmethod = 'onehot-' + interpmethod + '-' + fusion_method
        elif type_map == 'gauss_map':
            self.interpmethod = 'gauss-' + interpmethod + '-' + fusion_method
        else:
            self.interpmethod = 'post-' + interpmethod + '-' + fusion_method

        self.chunk_size = (96, 96, 96)
        self.channel_chunk = 20
        self.num_contrasts = 1

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions

        timepoints = list(filter(lambda t: t.get_image(**self.conditions_seg) is not None, timepoints))
        if len(timepoints) == 1 or sum([tp.get_image(**self.conditions_image) is not None for tp in timepoints]) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return None

        if 'sreg' in self.def_scope and 'sreg-lin' not in self.def_scope:
            svf_dict = {'sub': subject.id, 'suffix': 'svf'}
            for tp in timepoints:
                if not exists(join(tp.data_dir[self.def_scope], io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz')):
                    print('Subject: ' + str(subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                    return None

        if not force_flag:
            vol_tsv = {}
            for tp in timepoints:
                last_dir = tp.data_dir[self.results_scope]
                if exists(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')):
                    vol_tsv[tp.id] = io_utils.read_tsv(
                        join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'),
                        k=['id', 't_var', 's_var', 'interpmethod'])

            if all([tp.id in vol_tsv.keys() for tp in timepoints]):
                if all([tp.get_files(**self.conditions_image)[0][:-7] + 'Bayes_fBayes_sBayes' + self.interpmethod
                        in vol_tsv[tp.id].keys()] for tp in timepoints):
                    print('Subject: ' + str(subject.id) + '. DONE')
                    return None

        return timepoints

    def register_tp_to_sst(self, subject, tp, posteriors, v2r_tp):

        # Def parameters
        svf_dict = {'sub': subject.id, 'ses': tp.id, 'suffix': 'svf'}

        # Ref parameters
        aff_dict = {'sub': subject.id, 'ses': tp.id, 'desc': 'aff'}

        affine_matrix = np.load(join(tp.data_dir['sreg-lin'], io_utils.build_bids_fileame(aff_dict) + '.npy'))
        if self.def_scope != 'sreg-lin':
            tp_svf = nib.load(join(tp.data_dir[self.def_scope], io_utils.build_bids_fileame(svf_dict) + '.nii.gz'))
            svf = np.asarray(tp_svf.dataobj)
            proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
            proxyflow = def_utils.integrate_svf(proxysvf)

        else:
            proxyflow = None

        # Deform
        proxyref = nib.Nifti1Image(np.zeros(subject.image_shape),  subject.vox2ras0)
        v2r_target = np.matmul(np.linalg.inv(affine_matrix), v2r_tp)

        seg_list = []
        for it_c in range(0, posteriors.shape[-1], self.channel_chunk):
            proxyflo_seg = nib.Nifti1Image(posteriors[..., it_c:it_c + self.channel_chunk], v2r_target)
            seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxyflow=proxyflow, mode='bilinear')

            # In case the last iteration has only 1 channel (it squeezes due to batch dimension)
            if len(seg_mri.shape) == 3:
                seg_list += [np.array(seg_mri.dataobj)[..., np.newaxis]]
            else:
                seg_list += [np.array(seg_mri.dataobj)]

            del seg_mri

        seg_resampled = np.concatenate(seg_list, axis=-1).astype('float32')

        return seg_resampled

    def register_tps_to_tp(self, subject, timepoints, tp, image, posteriors, v2r_tp):
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

        # Def parameters
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}
        aff_dict = {'sub': subject.id, 'desc': 'aff'}

        image = gaussian_filter(image, sigma=0.5)
        segtemplate = posteriors
        p_data = np.ones(image.shape)
        if self.type_map == 'distance_map': segtemplate = softmax(segtemplate, axis=-1)
        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp.id == tp_flo.id: continue

            image_flo, seg_flo, v2r_flo = self.compute_seg_map(tp_flo)
            image_flo = gaussian_filter(image_flo, sigma=0.5)

            # Ref parameters
            affine_matrix_ref = np.load(join(tp.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'))
            svf_filename_ref = io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'

            # Flo parameters
            affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict}) + '.npy'))
            svf_filename_flo = io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **svf_dict}) + '.nii.gz'

            if self.def_scope != 'sreg-lin':
                tp_svf = nib.load(join(tp.data_dir[self.def_scope], svf_filename_ref))
                tp_flo_svf = nib.load(join(tp_flo.data_dir[self.def_scope], svf_filename_flo))

                svf = np.asarray(tp_flo_svf.dataobj) - np.asarray(tp_svf.dataobj)
                proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
                proxyflow = def_utils.integrate_svf(proxysvf)

            else:
                proxyflow = None

            # Deform
            v2r_ref = np.matmul(np.linalg.inv(affine_matrix_ref), v2r_tp)
            proxyref_aligned = nib.Nifti1Image(np.zeros(image.shape), v2r_ref)

            # Image
            v2r_target = np.matmul(np.linalg.inv(affine_matrix_flo), v2r_flo)
            proxyflo_im = nib.Nifti1Image(image_flo, v2r_target)
            im_mri = def_utils.vol_resample(proxyref_aligned, proxyflo_im, proxyflow=proxyflow, mode='bilinear')

            im_resampled = np.array(im_mri.dataobj)
            del im_mri

            # Segmentation
            v2r_seg_target = np.matmul(np.linalg.inv(affine_matrix_flo), v2r_flo)

            if len(seg_flo.shape) > 3:
                seg_list = []
                for it_c in range(0, seg_flo.shape[-1], self.channel_chunk):
                    proxyflo_seg = nib.Nifti1Image(seg_flo[..., it_c:it_c + self.channel_chunk],  v2r_seg_target)
                    seg_mri = def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode='bilinear')

                    # In case the last iteration has only 1 channel (it squeezes due to batch dimension)
                    if len(seg_mri.shape) == 3:
                        seg_list += [np.array(seg_mri.dataobj)[..., np.newaxis]]
                    else:
                        seg_list += [np.array(seg_mri.dataobj)]

                    del seg_mri

                seg_resampled = np.concatenate(seg_list, axis=-1)

            else:
                proxyflo_seg = nib.Nifti1Image(seg_flo, v2r_seg_target)
                seg_resampled = np.array(def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode='nearest').dataobj)

            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            mean_im_2 = (image - im_resampled)**2
            likelihood = np.exp(-0.5 / 9 * mean_im_2)
            p_data += likelihood
            segtemplate += likelihood[..., np.newaxis] * seg_resampled

        return segtemplate / (p_data[..., np.newaxis] + 1e-5)

    def register_prior_to_tp(self, subject, tp, proxytp, priorlabel):

        # Def parameters
        svf_dict = {'sub': subject.id, 'ses': tp.id, 'suffix': 'svf'}

        # Ref parameters
        aff_dict = {'sub': subject.id, 'ses': tp.id, 'desc': 'aff'}

        affine_matrix = np.load(join(tp.data_dir['sreg-lin'], io_utils.build_bids_fileame(aff_dict) + '.npy'))
        if self.def_scope != 'sreg-lin':
            tp_svf = nib.load(join(tp.data_dir[self.def_scope], io_utils.build_bids_fileame(svf_dict) + '.nii.gz'))
            svf = -np.asarray(tp_svf.dataobj)
            proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
            proxyflow = def_utils.integrate_svf(proxysvf)

        else:
            proxyflow = None

        # Deform
        v2r_ref = np.matmul(np.linalg.inv(affine_matrix), proxytp.affine)
        proxyref_aligned = nib.Nifti1Image(np.zeros(proxytp.shape), v2r_ref)
        v2r_seg_target = subject.vox2ras0

        seg_list = []
        for it_c in range(0, priorlabel.shape[-1], self.channel_chunk):
            proxyflo_seg = nib.Nifti1Image(priorlabel[..., it_c:it_c+self.channel_chunk], v2r_seg_target)
            seg_mri = def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode='bilinear')

            # In case the last iteration has only 1 channel (it squeezes due to batch dimension)
            if len(seg_mri.shape) == 3:
                seg_list += [np.array(seg_mri.dataobj)[..., np.newaxis]]
            else:
                seg_list += [np.array(seg_mri.dataobj)]

            del seg_mri

        seg_resampled = np.concatenate(seg_list, axis=-1)

        return seg_resampled

    def compute_seg_map(self, tp, get_post=True):
        proxyimage = tp.get_image(**self.conditions_image)
        proxyseg = tp.get_image(**self.conditions_seg)
        v2r_ref = proxyimage.affine.copy()

        seg = np.array(proxyseg.dataobj)
        if not self.all_labels_flag:
            seg[seg > 1999] = 42
            seg[seg > 999] = 3

        if get_post is False:
            pass

        elif self.type_map == 'distance_map':
            seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32', copy=False)

        else:
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)

        proxyseg_flo = nib.Nifti1Image(seg, proxyseg.affine)
        proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg_flo, mode='bilinear')
        seg = np.array(proxyseg_flo.dataobj)
        mask = np.sum(seg, -1) > 0.5
        mask, crop_coord = fn_utils.crop_label(mask > 0, margin=10, threshold=0)
        image = fn_utils.apply_crop(np.array(proxyimage.dataobj), crop_coord)
        seg = fn_utils.apply_crop(seg, crop_coord)
        v2r_ref[:3, 3] = v2r_ref[:3] @ np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0], 1])

        return image, seg, v2r_ref

    def initialize_parameters(self, subject, timepoints):
        print('\n    - Initialising prior probabilities using the ' + self.init_mode + ' method: ', end='', flush=True)
        # print('    - Reading data:', end=' ', flush=True)

        proxytemplate = nib.load(join(subject.data_dir[self.def_scope],  'sub-' + subject.id + '_desc-nonlinTemplate_T1w.nii.gz'))
        image_dict = {tp.id: None for tp in timepoints}
        tp_prior = {}#{tp.id: np.zeros(proxytemplate.shape + (len(self.labels_lut),), dtype='float32') for tp in timepoints}

        if all([exists('/tmp/bayesM_SLR/' + subject.id + '_' + tp.id + '.nii.gz') for tp in timepoints]):
            print(' **Already existing, reading from disk**')
            for tp in timepoints:
                image, _, v2r_ref = self.compute_seg_map(tp, get_post=False)
                image = gaussian_filter(image, sigma=0.5)
                # image = np.log(image + 1e-5)
                image_dict[tp.id] = nib.Nifti1Image(image, v2r_ref)
                image_dict[tp.id].uncache()
                # tp_prior[tp.id] = np.array(nib.load(join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.init_mode.replace('_', '') + '_prior.nii.gz')).dataobj)

            return image_dict, tp_prior


        if self.init_mode == 'sst':
            prior_template = np.zeros(proxytemplate.shape + (len(self.labels_lut),), dtype='float32')

            for tp in timepoints:
                print(tp.id, end=' ', flush=True)
                image, posteriors, v2r_ref = self.compute_seg_map(tp)

                image = gaussian_filter(image, sigma=0.5)
                # image = np.log(image + 1e-5)
                image_dict[tp.id] = nib.Nifti1Image(image, v2r_ref)
                image_dict[tp.id].uncache()

                reg_posteriors = self.register_tp_to_sst(subject, tp, posteriors, v2r_ref)
                if self.type_map == 'distance_map': reg_posteriors = softmax(reg_posteriors, axis=-1)
                prior_template += reg_posteriors / len(timepoints)

            for tp in timepoints:
                print(tp.id, end=' ', flush=True)
                tp_prior[tp.id] = self.register_prior_to_tp(subject, tp, image_dict[tp.id], prior_template)
        else:
            for tp in timepoints:
                print(tp.id, end=' ', flush=True)
                image, posteriors, v2r_ref = self.compute_seg_map(tp)
                if not exists('/tmp/bayesM_SLR/' + subject.id + '_' + tp.id + '.nii.gz'):
                    s = self.register_tps_to_tp(subject, timepoints, tp, image, posteriors, v2r_ref)
                    img = nib.Nifti1Image(s, v2r_ref)
                    if DEBUG:
                        nib.save(img, join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.type_map + '_' + self.init_mode.replace('_', '') + 'prior.nii.gz'))
                    else:
                        nib.save(img, '/tmp/bayesM_SLR/' + subject.id + '_' + tp.id + '.nii.gz')

                    img.uncache()
                    del posteriors, s, img

                elif DEBUG and not exists(join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.type_map + '_' + self.init_mode.replace('_', '') + 'prior.nii.gz')):
                    shutil.copy('/tmp/bayesM_SLR/' + subject.id + '_' + tp.id + '.nii.gz', join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.type_map + '_' + self.init_mode.replace('_', '') + 'prior.nii.gz'))



                # else:
                #     if exists(join(join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.type_map + '_' + self.init_mode.replace('_', '') + 'prior.nii.gz'))):
                #         tp_prior[tp.id] = np.array(nib.load(join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.type_map + '_' + self.init_mode.replace('_', '') + 'prior.nii.gz')).dataobj)
                #     else:
                #         tp_prior[tp.id] = self.register_tps_to_tp(subject, tp, image, posteriors, v2r_ref)
                # #

                image = gaussian_filter(image, sigma=0.5)
                # image = np.log(image + 1e-5)
                image_dict[tp.id] = nib.Nifti1Image(image, v2r_ref)
                image_dict[tp.id].uncache()
            #
            # for tp in timepoints:
            #     if DEBUG:
            #         tp_prior[tp.id] = np.array(nib.load(join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.type_map + '_' + self.init_mode.replace('_', '') + 'prior.nii.gz')).dataobj)
            #     else:
            #         tp_prior[tp.id] = np.array(nib.load('/tmp/bayesM_SLR/' + subject.id + '_' + tp.id + '.nii.gz').dataobj)
            #         os.remove('/tmp/bayesM_SLR/' + subject.id + '_' + tp.id + '.nii.gz')

        print('. Done.')
        return image_dict, tp_prior

    def compute_template_seg(self, timepoints, u_dict, Sigma_dict, imtemplate, segtemplate, N_k):
        P0 = 0.5 * N_k

        # q_0 = np.stack([norm.pdf(imtemplate, loc=u_0[it], scale=np.sqrt(Sigma_0[it])) if not (
        #             np.isnan(u_0[it]) or np.isnan(Sigma_0[it]) or Sigma_0[it] < 1e-5) else np.zeros(imtemplate.shape)
        #                 for it in range(segtemplate.shape[-1])], -1)
        # q_0 *= segtemplate
        # norm_q = (np.sum(q_0, axis=-1, keepdims=True) + 1e-5)
        # q_0 /= norm_q

        u_0 = 1 / np.sum(np.stack([1/Sigma_dict[tp.id] for tp in timepoints], 0), 0) * np.sum(np.stack([u_dict[tp.id]/Sigma_dict[tp.id] for tp in timepoints], 0), 0)
        Sigma_0 = 1/len(timepoints) * np.sum(np.stack([1/Sigma_dict[tp.id] for tp in timepoints], 0), 0) * P0 / (P0 - self.num_contrasts - 2)
        Sigma_0 = 1 / Sigma_0

        return u_0, Sigma_0

    def compute_tp_post(self, proxyimage, u_t, Sigma_t, q_0_prior, subject, tp):
        image = np.array(proxyimage.dataobj)
        proxyimage.uncache()

        # Compute posteriors
        q_t = np.stack([norm.pdf(image, loc=u_t[it], scale=np.sqrt(Sigma_t[it]))
                        if not (np.isnan(u_t[it]) or np.isnan(Sigma_t[it]) or Sigma_t[it] < 1e-5)
                        else np.zeros(image.shape) for it in range(q_0_prior.shape[-1])], -1)
        q_t *= q_0_prior
        norm_q = (np.sum(q_t, axis=-1, keepdims=True) + 1e-5)
        q_t /= norm_q

        return q_t

    def compute_gmm_params(self, image, q_t):

        # Update GMM parameters
        norm_q_t = np.sum(q_t, axis=(0, 1, 2)) + 1e-5
        u_t = np.sum(q_t * image[..., np.newaxis], axis=(0, 1, 2)) / norm_q_t
        Sigma_t = np.sum(q_t * (image[..., np.newaxis] - u_t) ** 2, axis=(0, 1, 2)) / norm_q_t

        return u_t, Sigma_t

    def initialize_GMM(self, timepoints, image_dict, prior_dict):
        u_dict = {}
        Sigma_dict = {}
        for tp in timepoints:
            image = np.array(image_dict[tp.id].dataobj)
            image_dict[tp.id].uncache()

            u_dict[tp.id], Sigma_dict[tp.id] = self.compute_gmm_params(image, prior_dict[tp.id])

        return u_dict, Sigma_dict

    def compute_p_label(self, subject, timepoints):

        tmpdir = '/tmp/' + subject.id + '_SLR_bayes'
        if not exists(tmpdir): os.makedirs(tmpdir)

        image_dict, segtemplate = self.initialize_parameters(subject, timepoints)
        u_dict, Sigma_dict = self.initialize_GMM(timepoints, image_dict, segtemplate)
        num_iter = 5
        for it in range(num_iter):
            print('Iteration: ' + str(it))

            for tp in timepoints:
                print('   Timepoint: ' + str(tp.id))

                q_t = self.compute_tp_post(image_dict[tp.id], u_dict[tp.id], Sigma_dict[tp.id], copy.copy(segtemplate[tp.id]), subject, tp)

                image = np.array(image_dict[tp.id].dataobj)
                v2r_ref = image_dict[tp.id].affine
                image_dict[tp.id].uncache()

                u_dict[tp.id], Sigma_dict[tp.id] = self.compute_gmm_params(image, q_t)

                if tp.id == '01':
                    proxyflo = nib.Nifti1Image(np.argmax(q_t, -1).astype('uint8'), v2r_ref)
                    nib.save(proxyflo, join(tp.data_dir[self.def_scope], str(it) + '.nii.gz'))

                if it == num_iter - 1:
                    proxyflo = nib.Nifti1Image(q_t, v2r_ref)
                    nib.save(proxyflo, join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.seg_model + '_posteriors.nii.gz'))

    def label_fusion(self, subject, force_flag=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'#'w' if force_flag else 'a'
        timepoints = self.prepare_data(subject, force_flag)
        if timepoints is None:
            return

        print('  o Computing the segmentation')
        t_0 = time.time()
        self.compute_p_label(subject, timepoints)
        t_1 = time.time()
        print('  Elapsed time: ' + str(t_1 - t_0) + ' seconds.\n')
        print('  o Writing the results')
        for tp in timepoints:
            print('    - Timepoint ' + tp.id)

            proxylabel = nib.load(join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.seg_model + '_posteriors.nii.gz'))
            aff = proxylabel.affine
            p_label = np.array(proxylabel.dataobj)
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]

            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)
            fp_dict['suffix'] = 'dseg'
            fp_dict['desc'] = 't' + self.seg_model + 's' + self.seg_model + self.interpmethod
            filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

            fake_vol = np.argmax(p_label, axis=-1).astype('int16')
            if self.seg_only:
                true_vol = np.zeros_like(fake_vol)
                for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul
                img = nib.Nifti1Image(true_vol, aff)
                nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                del true_vol

            vols = get_vols_post(p_label, res=pixdim)
            st_vols_dict_norm = {k: vols[v] for k, v in self.labels_lut.items()}

            vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
            st_vols_dict = {k:  vols[v] for k, v in self.labels_lut.items()}

            del p_label, fake_vol

            st_vols = []
            st_vols_dict_norm['id'] = filename[:-7]
            st_vols_dict_norm['t_var'] = self.seg_model
            st_vols_dict_norm['s_var'] = self.seg_model
            st_vols_dict_norm['interpmethod'] = self.interpmethod
            st_vols_dict_norm['type'] = 'posteriors'
            st_vols += [st_vols_dict_norm]

            st_vols_dict['id'] = filename[:-7]
            st_vols_dict['t_var'] = self.seg_model
            st_vols_dict['s_var'] = self.seg_model
            st_vols_dict['interpmethod'] = self.interpmethod
            st_vols_dict['type'] = 'seg'
            st_vols += [st_vols_dict]

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)



        # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')

class LabelSimpleBayesModel(LabelSimpleBayes):
    # Currently working with a single gaussian for each region. Need to update the same model as SAMSEG.
    # Register all to generate a SST 'probabilistic atlas' or use synthseg
    # Compute u0 and Sigma_0 by computing the statistics in SST space
    # For each timepoint:
    #    - Compute q as the posterior likelihood*prior /
    #    - Compute model parameters with the computed 'q'
    #    - Update u0 and Sigma_t
    #    - Update ut and Sigma_t
    #

    def __init__(self, def_scope='sreg-synthmorph', seg_scope='synthseg', smooth=None,
                 time_marker='time_to_bl_days', interpmethod='linear', type_map=None, fusion_method='post',
                 all_labels_flag=None, seg_only=False, init_mode='sst'):


        super().__init__(def_scope=def_scope, seg_scope=seg_scope, smooth=smooth, time_marker=time_marker,
                         interpmethod=interpmethod, type_map=type_map, fusion_method=fusion_method,
                         all_labels_flag=all_labels_flag, seg_only=seg_only, init_mode=init_mode)
        self.seg_model = 'bayesM'

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions

        timepoints = list(filter(lambda t: t.get_image(**self.conditions_seg) is not None, timepoints))
        if len(timepoints) == 1 or sum([tp.get_image(**self.conditions_image_synthseg) is not None for tp in timepoints]) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return None

        if 'sreg' in self.def_scope and 'sreg-lin' not in self.def_scope:
            svf_dict = {'sub': subject.id, 'suffix': 'svf'}
            for tp in timepoints:
                if not exists(join(tp.data_dir[self.def_scope], io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz')):
                    print('Subject: ' + str(subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                    return None

        if not force_flag:
            vol_tsv = {}
            for tp in timepoints:
                last_dir = tp.data_dir[self.results_scope]
                if exists(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')):
                    vol_tsv[tp.id] = io_utils.read_tsv(
                        join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'),
                        k=['id', 't_var', 's_var', 'interpmethod'])

            if all([tp.id in vol_tsv.keys() for tp in timepoints]):
                if all([tp.get_files(**self.conditions_image)[0][:-7] + 'Bayes_fBayes_sBayes' + self.interpmethod
                        in vol_tsv[tp.id].keys()] for tp in timepoints):
                    print('Subject: ' + str(subject.id) + '. DONE')
                    return None

        return timepoints

    def get_gmm(self):

        simple_model = {
            'PA': {'labels': [52, 13], 'nc': 2, 'u': [100, 100], 'sigma': [10, 10], 'g': [0.5, 0.5]},
            'PU': {'labels': [51, 12], 'nc': 2, 'u': [100, 100], 'sigma': [10, 10], 'g': [0.5, 0.5]},
            'TH': {'labels': [49, 10], 'nc': 2, 'u': [100, 100], 'sigma': [10, 10], 'g': [0.5, 0.5]},
            'CSF': {'labels': [4, 5, 14, 15, 24, 43, 44, 30, 31, 62, 63], 'nc': 3, 'u': [30, 30, 30], 'sigma': [10, 10, 10], 'g': [1/3, 1/3, 1/3]},
            'GM': {'labels': [17, 18, 53, 54, 26, 58, 11, 50, 3, 8, 42, 47], 'nc': 3, 'u': [90, 90, 90], 'sigma': [10, 10, 10], 'g': [1/3, 1/3, 1/3]},
            'WM': {'labels': [2, 7, 16, 41, 46, 28, 60, 85], 'nc': 2, 'u': [110, 110], 'sigma': [10, 10], 'g': [0.5, 0.5]},
            'Other': {'labels': [77, 80, 0, 251, 252, 253, 254, 255], 'nc': 3, 'u': [0, 50, 120], 'sigma': [10, 10, 10], 'g': [1/3, 1/3, 1/3]},
        }

        complex_model = {
            'PA': {'labels': [52, 13], 'nc': 2, 'g': [0.5, 0.5]},
            'PU': {'labels': [51, 12], 'nc': 2, 'g': [0.5, 0.5]},
            'TH': {'labels': [49, 10], 'nc': 2, 'g': [0.5, 0.5]},
            # 'HP': {'labels': [17, 53], 'nc': 2, 'g': [0.5, 0.5]},
            # 'InfVent': {'labels': [5, 44], 'nc': 2},
            'CSF': {'labels': [24], 'nc': 2},
            'OtherCSF': {'labels': [4, 5, 14, 15, 30, 31, 43, 44, 62, 63], 'nc': 2},
            'GM': {'labels': [3, 8, 11, 17, 18, 26, 42, 47, 50, 53, 54, 58], 'nc': 2},
            'WM': {'labels': [2, 7, 16, 41, 46, 28, 60, 85], 'nc': 2},
            'Other': {'labels': [77, 80, 0, 251, 252, 253, 254, 255], 'nc': 3},
        }

        model = simple_model
        for k, v in model.items():
            model[k]['g'] = 1/model[k]['nc']
        return model

    def compute_tp_post(self, image, u_t, Sigma_t, w_t, q_0_prior, subject, tp):

        gmm_model = self.get_gmm()
        q_t = {}
        ll_tgk = {}
        for group_label, group_dict in gmm_model.items():
            # Compute posteriors
            u_tg = u_t[group_label]
            Sigma_tg = Sigma_t[group_label]
            w_tg = w_t[group_label]
            imarray = np.array(image.dataobj)
            ll_tg = np.stack([w_tg[it] * norm.pdf(imarray, loc=u_tg[it], scale=np.sqrt(Sigma_tg[it]))
                              if not (np.isnan(u_tg[it]) or np.isnan(Sigma_tg[it]) or Sigma_tg[it] < 1e-5)
                              else np.zeros(image.shape) for it in range(group_dict['nc'])], -1)

            ll_tgk[group_label] = ll_tg

        norm_q = np.zeros(image.shape)
        for group_label, group_dict in gmm_model.items():
            pdata = np.sum(ll_tgk[group_label], -1)
            idx_group_labels = [self.labels_lut[l] for l in group_dict['labels'] if l in self.labels_lut.keys()]
            for idx_l in idx_group_labels:
                norm_q += (pdata * q_0_prior[..., idx_l])

        for group_label, group_dict in gmm_model.items():
            q_tg = np.zeros(image.shape + (group_dict['nc'],))
            idx_group_labels = [self.labels_lut[l] for l in group_dict['labels'] if l in self.labels_lut.keys()]
            for idx_l in idx_group_labels:
                q_tg += ll_tgk[group_label] * q_0_prior[..., idx_l, np.newaxis]
            q_tg /= (norm_q[..., np.newaxis] + 1e-5)

            q_t[group_label] = q_tg

        return q_t

    def compute_final_tp_post(self, image, u_t, Sigma_t, w_t, q_0_prior, subject, tp):

        gmm_model = self.get_gmm()
        ll_tgk = {}
        for group_label, group_dict in gmm_model.items():
            # Compute posteriors
            u_tg = u_t[group_label]
            Sigma_tg = Sigma_t[group_label]
            w_tg = w_t[group_label]

            ll_tg = np.stack([w_tg[it] * norm.pdf(np.array(image.dataobj), loc=u_tg[it], scale=np.sqrt(Sigma_tg[it]))
                              if not (np.isnan(u_tg[it]) or np.isnan(Sigma_tg[it]) or Sigma_tg[it] < 1e-5)
                              else np.zeros(image.shape) for it in range(group_dict['nc'])], -1)

            ll_tgk[group_label] = ll_tg

        norm_q = np.zeros(image.shape)
        for group_label, group_dict in gmm_model.items():
            pdata = np.sum(ll_tgk[group_label], -1)
            idx_group_labels = [self.labels_lut[l] for l in group_dict['labels'] if l in self.labels_lut.keys()]
            for idx_l in idx_group_labels:
                norm_q += (pdata * q_0_prior[..., idx_l])

        q_t = np.zeros_like(q_0_prior)
        for group_label, group_dict in gmm_model.items():
            idx_group_labels = [self.labels_lut[l] for l in group_dict['labels'] if l in self.labels_lut.keys()]
            for idx_l in idx_group_labels:
                q_t[..., idx_l] = np.sum(ll_tgk[group_label] * q_0_prior[..., idx_l, np.newaxis], axis=-1)/norm_q

        return q_t

    def compute_gmm_params(self, image, q_t):
        #q_t is a dictionary of {'group_label_i': proxyimage.shape + (num_gaussians,)}}
        # u_t is a dictionary of {'group_label_i':  (num_gaussians,)}}
        gmm_model = self.get_gmm()

        # Update GMM parameters
        u_t = {}
        Sigma_t = {}
        w_t = {}
        for group_label, group_dict in gmm_model.items():
            q_tg = q_t[group_label]

            norm_q_tg = np.sum(q_tg, axis=(0, 1, 2))
            w_tg = norm_q_tg / np.sum(norm_q_tg)
            u_tg = np.sum(q_tg * image[..., np.newaxis], axis=(0, 1, 2)) / (norm_q_tg + 1e-5)
            Sigma_tg = np.sum(q_tg * (image[..., np.newaxis] - u_tg) ** 2, axis=(0, 1, 2)) / (norm_q_tg + 1e-5)

            u_t[group_label] = u_tg
            Sigma_t[group_label] = Sigma_tg
            w_t[group_label] = w_tg

        return u_t, Sigma_t, w_t

    def initialize_GMM(self, timepoints, image_dict, segtemplate):
        gmm_model = self.get_gmm()

        u_dict = {}
        Sigma_dict = {}
        w_dict = {}
        for tp in timepoints:
            image = np.array(image_dict[tp.id].dataobj)
            prior = segtemplate[tp.id]

            u_t = {}
            Sigma_t = {}
            w_t = {}
            for group_label, group_dict in gmm_model.items():
                idx_group_labels = [self.labels_lut[l] for l in group_dict['labels'] if l in self.labels_lut.keys()]
                # Calculate the global weighted mean and variance of this class, where the weights are given by the prior
                classprior = np.sum(np.stack([prior[..., idx] for idx in idx_group_labels], 0), 0)
                mean = np.sum(image * classprior) / np.sum(classprior)
                variance = np.sum((image - mean) * ((image - mean) * classprior)) / np.sum(classprior)

                # Based on this, initialize the mean and variance of the individual Gaussian components in this class'
                # mixture model: variances are simply copied from the global class variance, whereas the means are
                # determined by splitting the [ mean-sqrt( variance ) mean+sqrt( variance ) ] domain into equal intervals,
                # the middle of which are taken to be the means of the Gaussians. Mixture weights are initialized to be
                # all equal.

                # This actually creates a mixture model that mimics the single Gaussian quite OK-ish
                numberOfComponents = group_dict['nc']

                Sigma_t[group_label] = np.zeros((group_dict['nc'],))
                u_t[group_label] = np.zeros((group_dict['nc'],))
                w_t[group_label] = np.zeros((group_dict['nc'],))
                for nc in range(group_dict['nc']):
                    intervalSize = 2 * np.sqrt(variance) / group_dict['nc']
                    Sigma_t[group_label][..., nc] = variance
                    u_t[group_label][..., nc] = (mean - np.sqrt(variance) + intervalSize / 2 + nc * intervalSize).T
                    w_t[group_label][..., nc] = 1 / numberOfComponents

            u_dict[tp.id] = u_t
            Sigma_dict[tp.id] = Sigma_t
            w_dict[tp.id] = w_t
        return u_dict, Sigma_dict, w_dict

    def compute_p_label(self, subject, timepoints):
        # Compute u0 and Sigma_0 by computing the statistics in SST space
        # For each timepoint:
        #    - Compute q as the posterior likelihood*prior /
        #    - Compute model parameters with the computed 'q'
        #    - Update u0 and Sigma_t
        #    - Update ut and Sigma_t
        tmpdir = '/tmp/' + subject.id + '_SLR_bayes'
        if not exists(tmpdir): os.makedirs(tmpdir)

        num_iter = 5
        # Initialise STT segmentation. Need to account for several options:
        # (1) Registration;
        # (2) SynthSEG;
        # (3) Registration with label fusion at each timepoint, i.e., a different seg. of the SST for each timepoint.
        # image_dict, segtemplate = self.initialize_parameters(subject, timepoints)
        # u_dict, Sigma_dict, w_dict = self.initialize_GMM(timepoints, image_dict, segtemplate)
        for tp in timepoints:
            print('   Timepoint: ' + str(tp.id))
            image, posteriors, v2r_ref = self.compute_seg_map(tp)
            segtemplate = self.register_tps_to_tp(subject, timepoints, tp, image, posteriors, v2r_ref)
            image = gaussian_filter(image, sigma=0.5)
            image_dict = {tp.id: nib.Nifti1Image(image, v2r_ref)}

            u_dict, Sigma_dict, w_dict = self.initialize_GMM([tp], image_dict, {tp.id: segtemplate})
            for it in range(num_iter):
                print('Iteration: ' + str(it))
                q_t = self.compute_tp_post(image_dict[tp.id], u_dict[tp.id], Sigma_dict[tp.id], w_dict[tp.id], segtemplate, subject, tp)
                image = np.array(image_dict[tp.id].dataobj)

                u_dict[tp.id], Sigma_dict[tp.id], w_dict[tp.id] = self.compute_gmm_params(image, q_t)

                if it + 1 == num_iter:
                    q_t = self.compute_final_tp_post(image_dict[tp.id], u_dict[tp.id], Sigma_dict[tp.id], w_dict[tp.id], segtemplate, subject, tp)
                    proxyflo = nib.Nifti1Image(q_t, v2r_ref)
                    nib.save(proxyflo, join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-bayesM_posteriors.nii.gz'))

        # for tp in timepoints:
        #     v2r_ref = image_dict[tp.id].affine
        #     proxyflo = nib.Nifti1Image(segtemplate[tp.id], v2r_ref)
        #     nib.save(proxyflo, join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-bayesM_priorlast.nii.gz'))
        #     q_t = self.compute_final_tp_post(image_dict[tp.id], u_dict[tp.id], Sigma_dict[tp.id], w_dict[tp.id], segtemplate[tp.id], subject, tp)
        #
        #     proxyflo = nib.Nifti1Image(q_t, v2r_ref)
        #     nib.save(proxyflo, join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-bayesM_posteriors.nii.gz'))




class LabelGIF(LabelSimpleBayes):
    # Currently working with a single gaussian for each region. Need to update the same model as SAMSEG.
    # Register all to generate a SST 'probabilistic atlas' or use synthseg
    # Compute u0 and Sigma_0 by computing the statistics in SST space
    # For each timepoint:
    #    - Compute q as the posterior likelihood*prior /
    #    - Compute model parameters with the computed 'q'
    #    - Update u0 and Sigma_t
    #    - Update ut and Sigma_t

    def __init__(self, def_scope='sreg-synthmorph', seg_scope='synthseg', smooth=None,
                 time_marker='time_to_bl_days', interpmethod='linear', type_map=None, fusion_method='post',
                 all_labels_flag=None, seg_only=False, init_mode='sst'):


        assert init_mode == 'label_fusion'
        super().__init__(def_scope=def_scope, seg_scope=seg_scope, smooth=smooth, time_marker=time_marker,
                         interpmethod=interpmethod, type_map=type_map, fusion_method=fusion_method,
                         all_labels_flag=all_labels_flag, seg_only=seg_only, init_mode=init_mode)
        self.seg_model = 'bayesGIF'

    def initialize_MRF(self, timepoints):
        return {tp.id: np.eye(len(self.labels_lut.values())) for tp in timepoints}

    def compute_mrf_params(self, q_t):
        return None

    def compute_mrf_params_GIF(self, q_t):

        # mrf_t has K**2 parameters, so its a long vector, for synthses needs 33*33=1089 weights or parameters
        # from the paper one inferes that there are
        #
        # One needs to define the v_zh vector, being z de vector probability of label K and h the (weighted) vector probability of neighbours
        l_t = np.argmax(q_t, -1)
        l_t = fn_utils.one_hot_encoding(l_t).astype('float32', copy=False)

        g_i = np.zeros_like(l_t)
        g_i[:-1] += l_t[:-1]
        g_i[1:] += l_t[1:]
        g_i[:, -1] += l_t[:, -1]
        g_i[:, 1:] += l_t[:, 1:]
        g_i[:, :, -1] += l_t[:, -1]
        g_i[:, :, 1:] += l_t[:, 1:]

        f_i = np.zeros_like(q_t)
        f_i[:-1] += q_t[:-1]
        f_i[1:] += q_t[1:]
        f_i[:, -1] += q_t[:, -1]
        f_i[:, 1:] += q_t[:, 1:]
        f_i[:, :, -1] += q_t[:, -1]
        f_i[:, :, 1:] += q_t[:, 1:]

        # v_xyzk = l_t[x, y, z, k] * g_i[x, y, z]
        # v_xyzk2 = l_t[x, y, z, k2] * g_i[x, y, z]
        # f_kh = np.sum(q_t[..., k:k+1] * f_i)
        # f_k2h = np.sum(q_t[..., k2:k2+1] * f_i)
        # (v_xykz - v_xyk2z )*theta = np.log(f_kh) - np.log(f_k2h)
        #
        # v_k = l_t[..., k:k+1] * g_i

    def compute_tp_prior(self, q_t, q_0, mrf_t):
        l_t = np.argmax(q_t, -1)
        l_t = fn_utils.one_hot_encoding(l_t).astype('float32', copy=False)

        u_mrf = np.zeros_like(l_t)
        u_mrf[:-1] += l_t[:-1]
        u_mrf[1:] += l_t[1:]
        u_mrf[:, -1] += l_t[:, -1]
        u_mrf[:, 1:] += l_t[:, 1:]
        u_mrf[:, :, -1] += l_t[:, -1]
        u_mrf[:, :, 1:] += l_t[:, 1:]

        u_mrf = q_0 * np.exp(-u_mrf) / np.sum(q_0 * np.exp(-u_mrf), axis=-1, keepdims=True)
        return u_mrf

    def compute_tp_prior_VanLeemput(self, q_0, mrf_t, beta=1):
        K = q_0.shape[-1]
        g_i = np.zeros_like(q_0)
        g_i[:-1] += q_0[:-1]
        g_i[1:] += q_0[1:]
        g_i[:, -1] += q_0[:, -1]
        g_i[:, 1:] += q_0[:, 1:]
        h_i = np.zeros_like(q_0)
        h_i[:, :, -1] += h_i[:, -1]
        h_i[:, :, 1:] += h_i[:, 1:]

        u_mrf = q_0.reshape(-1, K) @ mrf_t['G'] @ np.transpose(g_i, axes=(3, 0, 1, 2)).reshape((K, -1)) + \
                q_0.reshape(-1, K) @ mrf_t['H'] @ np.transpose(h_i, axes=(3, 0, 1, 2)).reshape((K, -1))

        u_mrf = u_mrf.reshape(q_0.shape)

        return np.exp(-u_mrf) / np.sum(np.exp(-u_mrf), axis=-1, keepdims=True)

    def compute_tp_prior_GIF(self, q_t, q_0, mrf_t):
        K = q_t.shape[-1]
        g_i = np.zeros_like(q_t)
        g_i[:-1] += q_t[:-1]
        g_i[1:] += q_t[1:]
        g_i[:, -1] += q_t[:, -1]
        g_i[:, 1:] += q_t[:, 1:]
        g_i[:, :, -1] += q_t[:, -1]
        g_i[:, :, 1:] += q_t[:, 1:]

        u_mrf = mrf_t @ np.transpose(g_i, axes=(3, 0, 1, 2)).reshape((K, -1))
        u_mrf = np.transpose(u_mrf, axis=(1, 0)).reshape(q_0.shape)
        u_mrf = q_0 * np.exp(u_mrf) / np.sum(q_0 * np.exp(u_mrf), axis=-1, keepdims=True)
        return u_mrf


    def compute_p_label(self, subject, timepoints):
        # Compute u0 and Sigma_0 by computing the statistics in SST space
        # For each timepoint:
        #    - Compute q as the posterior likelihood*prior /
        #    - Compute model parameters with the computed 'q'
        #    - Update u0 and Sigma_t
        #    - Update ut and Sigma_t
        tmpdir = '/tmp/' + subject.id + '_SLR_bayes'
        if not exists(tmpdir): os.makedirs(tmpdir)

        # Initialization
        image_dict, segtemplate = self.initialize_parameters(subject, timepoints)
        u_dict, Sigma_dict = self.initialize_GMM(timepoints, image_dict, segtemplate)
        mrf_dict = self.initialize_MRF(timepoints)

        # Optimization
        for tp in timepoints:
            print('   Timepoint: ' + str(tp.id))
            q_t = segtemplate[tp.id]
            image = np.array(image_dict[tp.id].dataobj)
            v2r_ref = image_dict[tp.id].affine
            image_dict[tp.id].uncache()

            for it in range(5):
                print('Iteration: ' + str(it))

                q_t_prior = self.compute_tp_prior(q_t, segtemplate[tp.id], mrf_dict[tp.id])
                q_t = self.compute_tp_post(image, u_dict[tp.id], Sigma_dict[tp.id], mrf_dict[tp.id], q_t_prior, subject, tp)

                u_dict[tp.id], Sigma_dict[tp.id] = self.compute_gmm_params(image, q_t)
                mrf_dict[tp.id] = self.compute_mrf_params(q_t)

            proxyflo = nib.Nifti1Image(q_t, v2r_ref)
            nib.save(proxyflo, join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-' + self.seg_model + '_posteriors.nii.gz'))

class LabelModelGIF(LabelSimpleBayesModel):
    # Currently working with a single gaussian for each region. Need to update the same model as SAMSEG.
    # Register all to generate a SST 'probabilistic atlas' or use synthseg
    # Compute u0 and Sigma_0 by computing the statistics in SST space
    # For each timepoint:
    #    - Compute q as the posterior likelihood*prior /
    #    - Compute model parameters with the computed 'q'
    #    - Update u0 and Sigma_t
    #    - Update ut and Sigma_t

    def __init__(self, def_scope='sreg-synthmorph', seg_scope='synthseg', smooth=None,
                 time_marker='time_to_bl_days', interpmethod='linear', type_map=None, fusion_method='post',
                 all_labels_flag=None, seg_only=False, init_mode='sst'):


        assert init_mode == 'label_fusion'
        super().__init__(def_scope=def_scope, seg_scope=seg_scope, smooth=smooth, time_marker=time_marker,
                         interpmethod=interpmethod, type_map=type_map, fusion_method=fusion_method,
                         all_labels_flag=all_labels_flag, seg_only=seg_only, init_mode=init_mode)
        self.seg_model = 'bayesMGIF'

    def initialize_MRF(self, timepoints):
        return {tp.id: np.eye(len(self.labels_lut.values())) for tp in timepoints}

    def compute_mrf_params(self, q_t):
        return None

    def compute_tp_prior(self, q_t, q_0, mrf_t):
        l_t = np.argmax(q_t, -1)
        l_t = fn_utils.one_hot_encoding(l_t).astype('float32', copy=False)

        # l_t = q_t
        u_mrf = l_t
        u_mrf[:-1] += l_t[:-1]
        u_mrf[1:] += l_t[1:]
        u_mrf[:, :-1] += l_t[:, :-1]
        u_mrf[:, 1:] += l_t[:, 1:]
        u_mrf[:, :, :-1] += l_t[:, :, :-1]
        u_mrf[:, :, 1:] += l_t[:, :, 1:]

        u_mrf = q_0 * np.exp(-u_mrf*q_t) / np.sum(q_0 * np.exp(-u_mrf*q_t), axis=-1, keepdims=True)
        return u_mrf

    def compute_p_label(self, subject, timepoints):
        # Compute u0 and Sigma_0 by computing the statistics in SST space
        # For each timepoint:
        #    - Compute q as the posterior likelihood*prior /
        #    - Compute model parameters with the computed 'q'
        #    - Update u0 and Sigma_t
        #    - Update ut and Sigma_t
        tmpdir = '/tmp/' + subject.id + '_SLR_bayes'
        if not exists(tmpdir): os.makedirs(tmpdir)

        # Initialization
        image_dict, segtemplate = self.initialize_parameters(subject, timepoints)
        u_dict, Sigma_dict, w_dict = self.initialize_GMM(timepoints, image_dict, segtemplate)
        mrf_dict = self.initialize_MRF(timepoints)

        # Optimization
        num_iter = 5
        for tp in timepoints:
            print('   Timepoint: ' + str(tp.id))
            q_t = segtemplate[tp.id]
            image = np.array(image_dict[tp.id].dataobj)
            v2r_ref = image_dict[tp.id].affine
            image_dict[tp.id].uncache()

            for it in range(num_iter):
                print('Iteration: ' + str(it))

                q_t_prior = self.compute_tp_prior(q_t, segtemplate[tp.id], mrf_dict[tp.id])
                q_t_group = self.compute_tp_post(image_dict[tp.id], u_dict[tp.id], Sigma_dict[tp.id], w_dict[tp.id], q_t_prior, subject, tp)
                q_t = self.compute_final_tp_post(image_dict[tp.id], u_dict[tp.id], Sigma_dict[tp.id], w_dict[tp.id], q_t_prior, subject, tp)

                u_dict[tp.id], Sigma_dict[tp.id], w_dict[tp.id] = self.compute_gmm_params(image, q_t_group)
                mrf_dict[tp.id] = self.compute_mrf_params(q_t)

                if it == num_iter - 1:
                    q_t_prior = self.compute_tp_prior(q_t, segtemplate[tp.id], mrf_dict[tp.id])
                    q_t = self.compute_final_tp_post(image_dict[tp.id], u_dict[tp.id], Sigma_dict[tp.id], w_dict[tp.id], q_t_prior, subject, tp)

                    proxyflo = nib.Nifti1Image(q_t, v2r_ref)
                    nib.save(proxyflo, join(tp.data_dir[self.def_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_desc-bayesMGIF_posteriors.nii.gz'))





class LabelFusion(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None, normalise=True,
                 interpmethod='linear', save=False, process=False, type_map=None, fusion_method='post', jlf=False,
                 all_labels_flag=None, seg_only=False):

        self.seg_scope = seg_scop
        self.def_scope = def_scope
        self.results_scope = def_scope if seg_scop == 'synthseg' else def_scope + '-' + seg_scop
        self.temp_variance = temp_variance if temp_variance is not None else ['inf']
        self.spatial_variance = spatial_variance if spatial_variance is not None else ['inf']
        self.floor_variance = floor_variance if floor_variance is not None else [1]
        self.save = save
        self.process = process
        self.smooth = smooth
        self.time_marker = time_marker
        self.type_map = type_map
        self.fusion_method = fusion_method
        self.jlf = jlf
        self.normalise = normalise
        self.seg_only = seg_only

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

        self.channel_chunk = 10000

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions
        if not (all([tv == 'inf' for tv in self.temp_variance]) or self.save):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        if len(timepoints) == 1 or sum([tp.get_image(**self.conditions_image) is not None for tp in timepoints]) == 1:
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

        if all([tv == 'inf' for tv in self.temp_variance]) or self.save:
            time_list = {tp.id: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                time_list[tp.id] = float(tp.sess_metadata[self.time_marker])

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

        # Def parameters
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        # Ref parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}

        # ## To remove
        # if exists(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id, 'run': '01'}, **aff_dict}) + '.npy')):
        #     shutil.move(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id, 'run': '01'}, **aff_dict}) + '.npy'),
        #                 join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict}) + '.npy'))
        # ##
        affine_matrix_ref = np.load(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict}) + '.npy'))
        svf_filename_ref = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'


        # Flo parameters

        # ## To remove
        # if exists(join(tp_flo.data_dir['sreg-lin'],
        #                io_utils.build_bids_fileame({**{'ses': tp_flo.id, 'run': '01'}, **aff_dict_ref}) + '.npy')):
        #     shutil.move(join(tp_flo.data_dir['sreg-lin'],
        #                      io_utils.build_bids_fileame({**{'ses': tp_flo.id, 'run': '01'}, **aff_dict_ref}) + '.npy'),
        #                 join(tp_flo.data_dir['sreg-lin'],
        #                      io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_ref}) + '.npy'))
        # ##
        affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict}) + '.npy'))
        svf_filename_flo = io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **svf_dict}) + '.nii.gz'

        if self.def_scope != 'sreg-lin':
            tp_svf = nib.load(join(tp_ref.data_dir[self.def_scope], svf_filename_ref))
            tp_flo_svf = nib.load(join(tp_flo.data_dir[self.def_scope], svf_filename_flo))

            svf = np.asarray(tp_flo_svf.dataobj) - np.asarray(tp_svf.dataobj)
            proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
            proxyflow = def_utils.integrate_svf(proxysvf)

        else:
            proxyflow = None

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

    def compute_seg_map(self, tp, labels_lut, distance=False):
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
                seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=labels_lut).astype('float32', copy=False)

        elif self.type_map == 'onehot_map':
            proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg_flo.dataobj)
            mask = seg > 0
            if not self.all_labels_flag:
                seg[seg > 1999] = 42
                seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(labels_lut.keys())).astype('float32', copy=False)

        elif self.type_map == 'gauss_map':
            proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg_flo.dataobj)
            mask = seg > 0
            if not self.all_labels_flag:
                seg[seg > 1999] = 42
                seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(labels_lut.keys())).astype('float32', copy=False)
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

    def process_tp(self, subject, tp_ref, tp_flo, proxyim_ref, labels_lut):

        proxyimage, proxyseg = self.compute_seg_map(tp_flo, labels_lut, distance=False)
        if (not self.save and all([s == 'inf' for s in self.spatial_variance])) or \
                (self.save and exists(join(subject.data_dir[self.def_scope], 'images', tp_ref.id + '_to_' + tp_flo.id + '.nii.gz'))):
            proxyimage = None

        if self.type_map == 'distance_map':
            seg_mode = 'distance'
        else:
            seg_mode = 'bilinear'

        im_resampled, seg_resampled = self.register_timepoints_st(subject, tp_ref, tp_flo, proxyim_ref,
                                                                  proxyimage_flo=proxyimage,
                                                                  proxyseg_flo=proxyseg,
                                                                  seg_mode=seg_mode)

        return im_resampled, seg_resampled

    def compute_p_label(self, tp, subject, timepoints, labels_lut_process=None, mode='linear'):

        # proxyimage = tp.get_image(**self.conditions_image)
        if labels_lut_process is None:
            labels_lut_process = self.labels_lut

        proxyimage, proxyseg = self.compute_seg_map(tp, labels_lut_process, distance=False)
        v2r_ref = proxyimage.affine
        p_data = np.zeros(proxyimage.shape + (len(timepoints),), dtype='float32')
        p_label = np.zeros(proxyimage.shape + (len(labels_lut_process), len(timepoints)), dtype='float32')
        im = np.array(proxyimage.dataobj)
        im = gaussian_filter(im, sigma=0.5)
        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp.id == tp_flo.id:

                seg_resampled = np.array(proxyseg.dataobj)
                im_resampled = im

            else:
                im_resampled, seg_resampled = self.process_tp(subject, tp, tp_flo, proxyimage, labels_lut_process)

            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            if self.fusion_method == 'seg':
                num_classes = seg_resampled.shape[-1]
                seg_resampled_max = np.argmax(seg_resampled, axis=-1)
                seg_resampled = fn_utils.one_hot_encoding(seg_resampled_max, num_classes=num_classes).astype('float32', copy=False)

            p_data[..., it_tp_flo] = im_resampled
            p_label[..., it_tp_flo] = seg_resampled
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
            st_vols = []
            st_vols_dict = {t_var: {s_var: {} for s_var in self.spatial_variance} for t_var in self.temp_variance}
            st_vols_dict_norm = {t_var: {s_var: {} for s_var in self.spatial_variance} for t_var in self.temp_variance}
            labels_lut_list = []
            for it_c in range(0, len(self.labels_lut), self.channel_chunk):
                labels_lut_list.append({k: v-it_c for it_k, (k, v) in enumerate(self.labels_lut.items())
                                        if it_k >= it_c and it_k < it_c + self.channel_chunk })

            for labels_lut_process in labels_lut_list:
                mean_im_2, seg_res, aff = self.compute_p_label(tp, subject, timepoints, labels_lut_process=labels_lut_process)
                time_arr = np.array([v for v in time_list.values()])
                mean_age_2 = (time_arr - time_list[tp.id]) ** 2

                pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
                for t_var in self.temp_variance:
                    t_ker = 1 if t_var in ['inf', 0] else np.exp(-0.5 / t_var * mean_age_2)
                    for s_var in self.spatial_variance:
                        s_ker = np.ones_like(mean_im_2) if s_var in ['inf', 0] else np.exp(-0.5 / s_var * mean_im_2)
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
                            st_vols_dict_norm[t_var][s_var] = {**st_vols_dict_norm[t_var][s_var], **{k: vols[v] for k, v in labels_lut_process.tems()}}

                        vols = get_vols(fake_vol, res=pixdim, labels=list(labels_lut_process.values()))
                        st_vols_dict[t_var][s_var] = {**st_vols_dict[t_var][s_var], **{k: vols[v] for k, v in labels_lut_process.items()}}

                        del p_label, fake_vol#, true_vol

            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    if self.normalise:
                        st_d = st_vols_dict_norm[t_var][s_var]
                        st_d['id'] = filename[:-7]
                        st_d['t_var'] = t_var
                        st_d['s_var'] = s_var
                        st_d['interpmethod'] = self.interpmethod
                        st_d['type'] = 'posteriors'
                        st_vols += [st_vols_dict_norm]

                    st_d = st_vols_dict[t_var][s_var]
                    st_d['id'] = filename[:-7]
                    st_d['t_var'] = t_var
                    st_d['s_var'] = s_var
                    st_d['interpmethod'] = self.interpmethod
                    st_d['type'] = 'seg'
                    st_vols += [st_d]

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del mean_im_2, seg_res
            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')

class LabelFusionChunks(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None, normalise=True,
                 interpmethod='linear', save=False, process=False, type_map=None, fusion_method='post', jlf=False,
                 all_labels_flag=None, seg_only=False, kernel='gauss'):

        self.seg_scope = seg_scop
        self.def_scope = def_scope
        self.results_scope = def_scope if seg_scop == 'synthseg' else def_scope + '-' + seg_scop
        self.temp_variance = temp_variance if temp_variance is not None else ['inf']
        self.spatial_variance = spatial_variance if spatial_variance is not None else ['inf']
        self.floor_variance = floor_variance if floor_variance is not None else [1]
        self.save = save
        self.process = process
        self.smooth = smooth
        self.time_marker = time_marker
        self.type_map = type_map
        self.fusion_method = fusion_method
        self.jlf = jlf
        self.normalise = normalise
        self.seg_only = seg_only
        self.kernel = kernel

        seg_suffix = 'dseg' if type_map is not None else 'post'
        self.conditions_image_synthseg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': seg_scop}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': seg_suffix, 'scope': seg_scop, 'extension': 'nii.gz'}

        self.all_labels_flag = all_labels_flag
        if seg_scop == 'freesurfer' and all_labels_flag is False:
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_ARR)}
        elif seg_scop == 'freesurfer' and all_labels_flag:
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_APARC_ARR)}
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

        self.chunk_size = (192, 192, 192)#(96,96,96)#
        self.channel_chunk = 20

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions
        if not (all([tv == 'inf' for tv in self.temp_variance]) or self.save):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        timepoints = list(filter(lambda t: t.get_image(**self.conditions_seg) is not None, timepoints))
        if len(timepoints) == 1 or sum([tp.get_image(**self.conditions_image_synthseg) is not None for tp in timepoints]) == 1:
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
                    filename = tp.get_files(**self.conditions_image_synthseg)[0]
                    if not self.spatial_variance and not self.temp_variance:
                        timepoints_to_run.append(tp)

                    elif any([(filename[:-7] + '_' + str(t) + 'f' + str(f) + '_' + str(s) + '_' + self.interpmethod not in vol_tsv[tp.id].keys() and f != 1)
                              or (filename[:-7] + '_' + str(t) + '_' + str(s) + '_' + self.interpmethod not in vol_tsv[tp.id].keys() and f == 1) for f in self.floor_variance for t in self.temp_variance for s in self.spatial_variance]):
                        timepoints_to_run.append(tp)

        if not timepoints_to_run:
            print('Subject: ' + str(subject.id) + '. DONE')
            return None, None, None

        if all([tv == 'inf' for tv in self.temp_variance]) or self.save:
            time_list = {tp.id: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                time_list[tp.id] = float(tp.sess_metadata[self.time_marker])

        if subject.id == 'UMA0168': timepoints_to_run = timepoints_to_run[4:]
        if subject.id == 'UMA0001': timepoints_to_run = timepoints_to_run[2:]
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

        # Def parameters
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        # Ref parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}

        # ## To remove
        # if exists(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id, 'run': '01'}, **aff_dict}) + '.npy')):
        #     shutil.move(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id, 'run': '01'}, **aff_dict}) + '.npy'),
        #                 join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict}) + '.npy'))
        # ##
        affine_matrix_ref = np.load(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict}) + '.npy'))
        svf_filename_ref = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'


        # Flo parameters

        # ## To remove
        # if exists(join(tp_flo.data_dir['sreg-lin'],
        #                io_utils.build_bids_fileame({**{'ses': tp_flo.id, 'run': '01'}, **aff_dict_ref}) + '.npy')):
        #     shutil.move(join(tp_flo.data_dir['sreg-lin'],
        #                      io_utils.build_bids_fileame({**{'ses': tp_flo.id, 'run': '01'}, **aff_dict_ref}) + '.npy'),
        #                 join(tp_flo.data_dir['sreg-lin'],
        #                      io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_ref}) + '.npy'))
        # ##
        affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict}) + '.npy'))
        svf_filename_flo = io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **svf_dict}) + '.nii.gz'

        if self.def_scope != 'sreg-lin':
            tp_svf = nib.load(join(tp_ref.data_dir[self.def_scope], svf_filename_ref))
            tp_flo_svf = nib.load(join(tp_flo.data_dir[self.def_scope], svf_filename_flo))

            svf = np.asarray(tp_flo_svf.dataobj) - np.asarray(tp_svf.dataobj)
            proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
            proxyflow = def_utils.integrate_svf(proxysvf)

        else:
            proxyflow = None

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

    def compute_seg_map(self, tp, distance=False):
        proxyimage = tp.get_image(**self.conditions_image)
        if proxyimage is None:
            pdb.set_trace()
            proxyimage = tp.get_image(**self.conditions_image_synthseg)

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
                seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32', copy=False)

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

    def process_tp(self, subject, tp_ref, tp_flo, proxyim_ref):

        proxyimage, proxyseg = self.compute_seg_map(tp_flo, distance=False)
        if (not self.save and all([s == 'inf' for s in self.spatial_variance])) or \
                (self.save and exists(join(subject.data_dir[self.def_scope], 'images', tp_ref.id + '_to_' + tp_flo.id + '.nii.gz'))):
            proxyimage = None

        if self.type_map == 'distance_map':
            seg_mode = 'distance'
        else:
            seg_mode = 'bilinear'

        im_resampled, seg_resampled = self.register_timepoints_st(subject, tp_ref, tp_flo, proxyim_ref,
                                                                  proxyimage_flo=proxyimage,
                                                                  proxyseg_flo=proxyseg,
                                                                  seg_mode=seg_mode)

        return im_resampled, seg_resampled

    def compute_p_label(self, tp, subject, timepoints, chunk, mode='linear'):

        proxyimage, proxyseg = self.compute_seg_map(tp, distance=False)
        v2r_ref = proxyimage.affine

        p_data = np.zeros((chunk[0][1]-chunk[0][0], chunk[1][1]-chunk[1][0], chunk[2][1]-chunk[2][0], len(timepoints),), dtype='float32')
        p_label = np.zeros((chunk[0][1]-chunk[0][0], chunk[1][1]-chunk[1][0], chunk[2][1]-chunk[2][0], len(self.labels_lut), len(timepoints)), dtype='float32')


        im = np.array(proxyimage.dataobj)[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]]
        im = gaussian_filter(im, sigma=0.5)
        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp.id == tp_flo.id:
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

        return (im[..., np.newaxis] - p_data)**2, p_label, v2r_ref

    def label_fusion(self, subject, force_flag=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'#'w' if force_flag else 'a'
        timepoints_to_run, timepoints, time_list = self.prepare_data(subject, force_flag)

        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        for tp in timepoints_to_run:
            filename = tp.get_files(**self.conditions_image)
            if not filename:
                filename = tp.get_image(**self.conditions_image_synthseg)

            filename = filename[0]
            fp_dict = io_utils.get_bids_fileparts(filename)

            t_0 = time.time()
            print('        - Timepoint ' + tp.id, end=': ', flush=True)
            st_vols = []
            st_vols_dict = {t_var: {s_var: {k: 0 for k in self.labels_lut.keys()} for s_var in self.spatial_variance} for t_var in self.temp_variance}
            st_vols_dict_norm = {t_var: {s_var: {k: 0 for k in self.labels_lut.keys()} for s_var in self.spatial_variance} for t_var in self.temp_variance}

            proxyimage, _ = self.compute_seg_map(tp, distance=False)
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


            for it_chunk, chunk in enumerate(chunk_list):
                if it_chunk == len(chunk_list) - 1:
                    print(str(it_chunk) + '/' + str(len(chunk_list)), end='. ', flush=True)
                else:
                    print(str(it_chunk) + '/' + str(len(chunk_list)), end=', ', flush=True)

                mean_im_2, seg_res, aff = self.compute_p_label(tp, subject, timepoints, chunk)
                time_arr = np.array([v for v in time_list.values()])
                mean_age_2 = (time_arr - time_list[tp.id]) ** 2

                pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
                for t_var in self.temp_variance:
                    t_ker = 1 if t_var in ['inf', 0] else np.exp(-0.5 / t_var * mean_age_2)
                    for s_var in self.spatial_variance:
                        if self.kernel=='laplacian':
                            s_ker = np.ones_like(mean_im_2) if s_var in ['inf', 0] else np.exp(-np.sqrt(mean_im_2) / s_var)
                        else:
                            s_ker = np.ones_like(mean_im_2) if s_var in ['inf', 0] else np.exp(-0.5 / s_var * mean_im_2)

                        p_data = s_ker * t_ker
                        del s_ker

                        p_label = np.zeros(seg_res.shape[:-1])
                        if float(s_var) > 0 and float(t_var) > 0:
                            for it_t in range(seg_res.shape[-1]):
                                p_label += p_data[..., np.newaxis, it_t] * seg_res[..., it_t]#, axis=-1)
                        else:
                            it_ref = [it_t for it_t, t in enumerate(timepoints) if t.id==tp.id][0]
                            p_label = seg_res[..., it_ref]

                        del p_data

                        if self.normalise:
                            p_label = p_label / np.sum(p_label, axis=-1, keepdims=True)
                            p_label[np.isnan(p_label)] = 0

                        fp_dict['suffix'] = 'dseg'
                        fp_dict['desc'] = 't' + str(t_var) + 's' + str(s_var) + self.interpmethod
                        filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                        fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                        if self.seg_only:
                            true_vol = np.zeros_like(fake_vol)
                            for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul
                            if it_chunk == 0:
                                true_final_vol = np.zeros(volshape)
                            else:
                                proxytmp = nib.load(join(tp.data_dir[self.results_scope], filename_seg))
                                true_final_vol = np.array(proxytmp.dataobj)
                            true_final_vol[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]] = true_vol
                            img = nib.Nifti1Image(true_final_vol, aff)
                            nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                            del true_final_vol, true_vol

                        if self.normalise:
                            vols = get_vols_post(p_label, res=pixdim)
                            st_vols_dict_norm[t_var][s_var] = {k: st_vols_dict_norm[t_var][s_var][k] + vols[v] for k, v in self.labels_lut.items()}

                        vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                        st_vols_dict[t_var][s_var] = {k: st_vols_dict[t_var][s_var][k] + vols[v] for k, v in self.labels_lut.items()}

                        del p_label, fake_vol

            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:

                    if self.normalise:
                        st_d = st_vols_dict_norm[t_var][s_var]
                        st_d['id'] = filename[:-7]
                        st_d['t_var'] = t_var
                        st_d['s_var'] = s_var
                        st_d['interpmethod'] = self.interpmethod
                        st_d['type'] = 'posteriors'
                        st_vols += [st_d]

                    st_d = st_vols_dict[t_var][s_var]
                    st_d['id'] = filename[:-7]
                    st_d['t_var'] = t_var
                    st_d['s_var'] = s_var
                    st_d['interpmethod'] = self.interpmethod
                    st_d['type'] = 'seg'
                    st_vols += [st_d]

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')
            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del mean_im_2, seg_res
            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')





class LabelFusionTemplate(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None, normalise=True,
                 interpmethod='linear', save=False, process=False, type_map=None, fusion_method='post', jlf=False,
                 all_labels_flag=None):

        self.seg_scope = seg_scop
        self.def_scope = def_scope
        self.results_scope = def_scope + '-temp' if seg_scop == 'synthseg' else def_scope + '-' + seg_scop + '-temp'
        self.temp_variance = temp_variance if temp_variance is not None else ['inf']
        self.spatial_variance = spatial_variance if spatial_variance is not None else ['inf']
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
        if not (all([tv == 'inf' for tv in self.temp_variance]) or self.save):
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

        if all([tv == 'inf' for tv in self.temp_variance]) or self.save:
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
        if all([s == 'inf' for s in self.spatial_variance]): proxyimage = None

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
                t_ker = 1 if t_var in ['inf', 0] else np.exp(-0.5 / t_var * mean_age_2)
                for s_var in self.spatial_variance:
                    s_ker = np.ones_like(mean_im_2) if s_var in ['inf', 0] else np.exp(-0.5 / s_var * mean_im_2)
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

class LabelFusionSubfields(object):

    def __init__(self, def_scope='sreg-synthmorph', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, normalise=True,
                 interpmethod='linear', type_map=None, fusion_method='post', hemi=None):

        self.seg_scope = 'freesurfer-subfields'
        self.def_scope = def_scope
        self.results_scope = def_scope + '-freesurfer-subfields'
        self.channel_chunk = 20

        self.temp_variance = temp_variance if temp_variance is not None else ['inf']
        self.spatial_variance = spatial_variance if spatial_variance is not None else ['inf']

        self.smooth = smooth
        self.time_marker = time_marker
        self.type_map = type_map
        self.fusion_method = fusion_method
        self.normalise = normalise

        if hemi == None: hemi = ['rh', 'lh']
        self.hemi_list = hemi if isinstance(hemi, list) else [hemi]

        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_cog = {'desc': 'cog', 'scope': 'sreg-lin', 'extension': 'npy', 'run': '01'}
        self.conditions_seg = {'lh': {'space': 'lh', 'desc': 'subfields', 'suffix': 'dseg', 'scope': 'freesurfer-subfields'},
                               'rh':{'space': 'rh', 'desc': 'subfields', 'suffix': 'dseg', 'scope': 'freesurfer-subfields'} }

        self.labels_lut = {k: it_k for it_k, k in enumerate(SUBFIELDS_LABELS_ARR)}

        if type_map == 'distance_map':
            self.interpmethod = 'seg-' + interpmethod + '-' + fusion_method
        elif type_map == 'onehot_map':
            self.interpmethod = 'onehot-' + interpmethod + '-' + fusion_method
        elif type_map == 'gauss_map':
            self.interpmethod = 'gauss-' + interpmethod + '-' + fusion_method
        else:
            self.interpmethod = 'post-' + interpmethod + '-' + fusion_method

        self.fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type', 'hemi'] + list(ASEG_DICT.values()) + \
                                ['Left-' + i for i in list(EXTENDED_SUBFIELDS_LABELS.values())] + \
                                ['Right-' + i for i in list(EXTENDED_SUBFIELDS_LABELS.values())]

    def prepare_data(self, subject, force_flag, *args, **kwargs):
        timepoints = subject.sessions
        if len(timepoints) == 1:
            print('   Subject: ' + subject.id + ' has only 1 timepoint. No segmentation is computed.')
            return None, None

        if 'sreg' in self.def_scope and 'sreg-lin' not in self.def_scope:
            svf_dict = {'sub': subject.id, 'suffix': 'svf'}
            for tp in timepoints:
                # run_flag = any(['run' in f for f in tp.files['bids']])
                run_dict = {} #{'run': '01'} if run_flag and self.reg_algorithm == 'regnet' else {}
                if not exists(join(tp.data_dir[self.def_scope], io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict, **run_dict}) + '.nii.gz')):
                    print('Subject: ' + str(subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                    return None, None

        elif 'lin' not in self.def_scope:
            if not exists(join(subject.data_dir[self.def_scope], 'deformations')):
                print('Subject: ' + str(subject.id) + ' has not SVF maps available. Please, run the initialize_graph before')
                return None, None

        vol_tsv = {}
        for tp in timepoints:
            last_dir = tp.data_dir[self.results_scope]
            if exists(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')):
                vol_tsv[tp.id] = io_utils.read_tsv(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'), k=['id', 't_var', 's_var', 'hemi', 'interpmethod'])

        timepoints_to_run = []
        if force_flag:
            timepoints_to_run = timepoints
        elif DEBUG:
            timepoints_to_run = timepoints[0:1]
        else:
            to_compute = []
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    for h in self.hemi_list:
                        to_compute += [str(t_var) +'_' + str(s_var) + '_' + h + '_' + self.interpmethod]

            for tp in timepoints:
                if tp.id not in vol_tsv.keys():
                    timepoints_to_run.append(tp)
                else:
                    computed = ['_'.join(k.split('_')[-4:]) for k in vol_tsv[tp.id].keys()]
                    if not all([t in computed for t in to_compute]):
                        timepoints_to_run.append(tp)

        if not timepoints_to_run:
            print('Subject: ' + str(subject.id) + '. DONE')
            return None, None

        # timepoints_to_run = timepoints
        print('  o Reading the input files')
        if all([tv == 'inf' for tv in self.temp_variance]):
            time_list = {tp.id: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                # Age
                if 'time_to_bl_days' in tp.sess_metadata.keys():
                    time_list[tp.id] = float(tp.sess_metadata['time_to_bl_days'])
                else:
                    time_list[tp.id] = float(tp.sess_metadata['age'])

        return timepoints_to_run, time_list

    def compute_seg_map(self, tp, hemi, distance=True):
        proxyseg = tp.get_image(**self.conditions_seg[hemi])
        proxyimage = tp.get_image(**self.conditions_image)
        proxyimage = def_utils.vol_resample(proxyseg, proxyimage, mode='bilinear')
        v2r_ref = proxyseg.affine.copy()

        if self.type_map == 'distance_map':
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            if distance:
                seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32', copy=False)

        elif self.type_map == 'onehot_map':
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)

        elif self.type_map == 'gauss_map':
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)
            for it_label in range(seg.shape[-1]):
                seg[..., it_label] = gaussian_filter(seg[..., it_label], sigma=1)

            seg = softmax(seg, axis=-1)
        else:
            seg = np.array(proxyseg.dataobj)
            mask = np.sum(seg[..., 1:], -1) > 0

        mask, crop_coord = fn_utils.crop_label(mask > 0, margin=10, threshold=0)
        image = fn_utils.apply_crop(np.array(proxyimage.dataobj), crop_coord)
        seg = fn_utils.apply_crop(seg, crop_coord)
        v2r_ref[:3, 3] = v2r_ref[:3] @ np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0], 1])

        proxyim_ref = nib.Nifti1Image(image, v2r_ref)
        proxyseg_ref = nib.Nifti1Image(seg, v2r_ref)

        return proxyim_ref, proxyseg_ref

    def get_svf(self, subject, tp_ref, tp_flo, svf_dict):
        if 'sreg' in self.def_scope:
            # SVFs
            tp_ref_svf = nib.load(join(tp_ref.data_dir[self.def_scope], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'))
            tp_flo_svf = nib.load(join(tp_flo.data_dir[self.def_scope], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **svf_dict}) + '.nii.gz'))

            svf = np.asarray(tp_flo_svf.dataobj) - np.asarray(tp_ref_svf.dataobj)
            svf = svf.astype('float32')

        else:
            filename_svf = tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz'
            if exists(join(subject.data_dir['sreg-synthmorph'], 'deformations', filename_svf)):
                tp_ref_svf = nib.load(join(subject.data_dir['sreg-synthmorph'], 'deformations', filename_svf))
                svf = np.asarray(tp_ref_svf.dataobj)

            else:
                filename_svf = tp_flo.id + '_to_' + tp_ref.id + '.svf.nii.gz'
                tp_ref_svf = nib.load(join(subject.data_dir['sreg-synthmorph'], 'deformations', filename_svf))
                svf = -np.asarray(tp_ref_svf.dataobj)

            svf = svf.astype('float32')

        return svf, tp_ref_svf.affine


    def compute_1x1x1_inputs(self, subject, tp_ref, tp_flo, proxyref, proxyimage_flo=None, proxyseg_flo=None,
                             im_mode='bilinear', seg_mode='bilinear'):


        # Def parameters
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}
        aff_dict = {'sub': subject.id, 'desc': 'aff'}

        # Reference
        affine_matrix_ref = np.load(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict}) + '.npy'))
        # run_flag_ref = any(['run' in f for f in tp_ref.files['bids']])
        # aff_dict_ref = {'sub': subject.id, 'desc': 'aff'}
        # if run_flag_ref: aff_dict_ref['run'] = '01'

        # cog_ref = np.load(join(tp_ref.data_dir['sreg-lin'], tp_ref.get_files(**self.conditions_cog)[0]))
        # affine_matrix_ref = np.load(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict_ref}) + '.npy'))
        # affine_matrix_ref[:3, 3] += cog_ref[:3]
        # v2r_ref_subject = np.matmul(np.linalg.inv(affine_matrix_ref), v2r_ref)

        # Floating image
        affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict}) + '.npy'))
        # aff_dict_flo = {'sub': subject.id, 'desc': 'aff'}
        # run_flag_flo = any(['run' in f for f in tp_flo.files['bids']])
        # if run_flag_flo: aff_dict_flo['run'] = '01'
        #
        # cog_flo = np.load(join(tp_flo.data_dir['sreg-lin'], tp_flo.get_files(**self.conditions_cog)[0]))
        # affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_flo}) + '.npy'))
        # affine_matrix_flo[:3, 3] += cog_flo[:3]


        if self.def_scope != 'sreg-lin':
            svf, v2r_svf = self.get_svf(subject, tp_ref, tp_flo, svf_dict)
            proxysvf = nib.Nifti1Image(svf.astype('float32'), v2r_svf)
            proxyflow = def_utils.integrate_svf(proxysvf)
        else:
            proxyflow = None

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

                    seg_list += [np.array(seg_mri.dataobj)]

                    del seg_mri

                seg_resampled = np.concatenate(seg_list, axis=-1)
            else:
                proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj), v2r_seg_target)
                seg_resampled = np.array(def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode).dataobj)

        return im_resampled, seg_resampled

    # def create_ref_space(self, proxyseg_ref):
    #
    #     ii = np.arange(0, proxyseg_ref.shape[0], dtype='int32')
    #     jj = np.arange(0, proxyseg_ref.shape[1], dtype='int32')
    #     kk = np.arange(0, proxyseg_ref.shape[2], dtype='int32')
    #
    #     II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')
    #
    #     I = II.reshape(-1, 1)
    #     J = JJ.reshape(-1, 1)
    #     K = KK.reshape(-1, 1)
    #     C = np.ones((int(np.prod(proxyseg_ref.shape[:3])), 1), dtype='int32')
    #     voxMosaic_orig = np.concatenate((I, J, K, C), axis=1).T
    #     rasMosaic_ref = np.dot(proxyseg_ref.affine, voxMosaic_orig).astype('float32')
    #
    #     rasMosaic_ref = rasMosaic_ref.reshape((4,) + proxyseg_ref.shape)
    #
    #     return rasMosaic_ref
    #
    # def resample_ras_space(self, image, v2r, rasMosaic_ref, mode='bilinear'):
    #
    #     ref_shape = rasMosaic_ref.shape[1:]
    #     voxMosaic_targ = np.matmul(np.linalg.inv(v2r), rasMosaic_ref.reshape(4, -1)).astype('float32')
    #     voxMosaic_targ = voxMosaic_targ[:3]
    #
    #     voxMosaic_targ = voxMosaic_targ.reshape((3,) + ref_shape)
    #     voxMosaic_targ_torch = voxMosaic_targ.copy()
    #     voxMosaic_targ_torch = torch.from_numpy(voxMosaic_targ_torch[np.newaxis]).float().to(self.device)
    #     im = torch.from_numpy(image[np.newaxis]).float().to(self.device)
    #     im_resampled = self.interp_func(im, voxMosaic_targ_torch, mode=mode)
    #
    #     return im_resampled[0]

    def compute_p_label(self, tp, subject):

        timepoints = subject.sessions
        p_label, p_data, v2r_ref = {}, {}, {}
        for hemi in self.hemi_list:
            proxyimage, proxyseg = self.compute_seg_map(tp, hemi)
            ref_shape = proxyimage.shape
            v2r_ref[hemi] = proxyimage.affine
            im = np.array(proxyimage.dataobj)
            im = gaussian_filter(im, [0.25] * 3)

            # rasMosaic_ref = self.create_ref_space(proxyseg)

            p_data[hemi] = np.zeros(proxyimage.shape + (len(timepoints),), dtype='float32')
            p_label[hemi] = np.zeros(proxyimage.shape + (len(self.labels_lut), len(timepoints)), dtype='float32')

            for it_tp_flo, tp_flo in enumerate(timepoints):

                if tp_flo.id == tp.id:
                    p_label[hemi][..., it_tp_flo] = np.array(proxyseg.dataobj)
                    # pdb.set_trace()
                    # img = nib.Nifti1Image(im, proxyseg.affine)
                    # nib.save(img, tp_flo.id + '.im.nii.gz')
                    # img = nib.Nifti1Image(p_label[hemi], proxyseg.affine)
                    # nib.save(img, tp_flo.id + '.seg.nii.gz')

                else:
                    ################################
                    proxyimage_flo, proxyseg_flo = self.compute_seg_map(tp_flo, hemi)
                    im_res, seg_res = self.compute_1x1x1_inputs(subject, tp, tp_flo, proxyseg, proxyimage_flo, proxyseg_flo)
                    ################################

                    p_data[hemi][..., it_tp_flo] = (im - im_res)**2
                    p_label[hemi][..., it_tp_flo] = seg_res

                    # pdb.set_trace()
                    # img = nib.Nifti1Image(im_res, proxyseg.affine)
                    # nib.save(img, tp_flo.id + '.im.nii.gz')
                    # img = nib.Nifti1Image(seg_res, proxyseg.affine)
                    # nib.save(img, tp_flo.id + '.seg.nii.gz')

        return p_data, p_label, v2r_ref

    def label_fusion(self, subject, force_flag=False, **kwargs):

        print('Subject: ' + str(subject.id))
        attach_overwrite = 'a'
        timepoints_to_run, time_list = self.prepare_data(subject, force_flag, **kwargs)
        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        for tp in timepoints_to_run:
            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)

            t_0 = time.time()
            print('        - Timepoint ' + tp.id, end=':', flush=True)
            mean_im_2, seg_res, aff = self.compute_p_label(tp, subject)
            time_arr = np.array([v for v in time_list.values()])
            mean_age_2 = (time_arr - time_list[tp.id]) ** 2

            st_vols = []
            for hemi in self.hemi_list:
                prefix = 'Right-' if hemi == 'rh' else 'Left-'
                pixdim = np.sqrt(np.sum(aff[hemi] * aff[hemi], axis=0))[:-1]
                for t_var in self.temp_variance:
                    t_ker = 1 if t_var in ['inf', 0] else np.exp(-0.5 / t_var * mean_age_2)
                    for s_var in self.spatial_variance:
                        s_ker = np.ones_like(mean_im_2[hemi]) if s_var in ['inf', 0] else np.exp(-0.5 / s_var * mean_im_2[hemi])
                        p_data = s_ker * t_ker
                        del s_ker

                        if self.normalise:
                            p_data = p_data / np.sum(p_data, axis=-1, keepdims=True)
                            p_data[np.isnan(p_data)] = 0

                        p_label = np.zeros(seg_res[hemi].shape[:-1])
                        if float(s_var) > 0 and float(t_var) > 0:
                            for it_t in range(seg_res[hemi].shape[-1]):
                                p_label += p_data[..., np.newaxis, it_t] * seg_res[hemi][..., it_t]  # , axis=-1)
                        else:
                            it_ref = [it_t for it_t, t in enumerate(subject.sessions) if t.id == tp.id][0]
                            p_label = seg_res[hemi][..., it_ref]

                        del p_data

                        fp_dict['suffix'] = 'dseg'
                        fp_dict['desc'] = 't' + str(t_var) + 's' + str(s_var) + self.interpmethod
                        filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                        fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                        # true_vol = np.zeros_like(fake_vol)
                        # for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul
                        # img = nib.Nifti1Image(true_vol, aff[hemi])
                        # nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                        if self.normalise:
                            vols = get_vols_post(p_label, res=pixdim)
                            st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                            st_vols_dict['id'] = filename[:-7]
                            st_vols_dict['t_var'] = t_var
                            st_vols_dict['s_var'] = s_var
                            st_vols_dict['hemi'] = hemi
                            st_vols_dict['interpmethod'] = self.interpmethod
                            st_vols_dict['type'] = 'posteriors'

                            st_vols += [st_vols_dict]

                        vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                        st_vols_dict = {prefix + EXTENDED_SUBFIELDS_LABELS[k]: vols[v] for k, v in self.labels_lut.items()}
                        st_vols_dict[prefix + 'TotalAM'] = sum([vols[v] for k, v in self.labels_lut.items() if k in AM_LABELS_REV.values()])#for k in AM_LABELS_REV.values()])
                        st_vols_dict[prefix + 'TotalHP'] = sum([vols[v] for k, v in self.labels_lut.items() if k in HP_LABELS_REV.values()])
                        st_vols_dict['id'] = filename[:-7]
                        st_vols_dict['t_var'] = t_var
                        st_vols_dict['s_var'] = s_var
                        st_vols_dict['hemi'] = hemi
                        st_vols_dict['interpmethod'] = self.interpmethod
                        st_vols_dict['type'] = 'seg'

                        st_vols += [st_vols_dict]
                        del p_label, fake_vol  # , true_vol

            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')
            write_volume_results(st_vols, vols_dir, fieldnames=self.fieldnames, attach_overwrite=attach_overwrite)

            del mean_im_2, seg_res
            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')






# class FSSubfields_Nonlin(LabelFusionSubfields):
#
#     def get_svf(self, subject, tp_ref, tp_flo, svf_dict):
#         if 'sreg' in self.pipeline:
#             # SVFs
#             tp_ref_svf = nib.load(join(tp_ref.data_dir[self.pipeline + self.reg_algorithm], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'))
#             tp_flo_svf = nib.load(join(tp_flo.data_dir[self.pipeline + self.reg_algorithm], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **svf_dict}) + '.nii.gz'))
#
#             svf = np.asarray(tp_flo_svf.dataobj) - np.asarray(tp_ref_svf.dataobj)
#             svf = svf.astype('float32')
#
#         else:
#             filename_svf = tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz'
#             if exists(join(subject.data_dir[self.pipeline + self.reg_algorithm], 'deformations', filename_svf)):
#                 tp_ref_svf = nib.load(join(subject.data_dir[self.pipeline + self.reg_algorithm], 'deformations', filename_svf))
#                 svf = np.asarray(tp_ref_svf.dataobj)
#
#             else:
#                 filename_svf = tp_flo.id + '_to_' + tp_ref.id + '.svf.nii.gz'
#                 tp_ref_svf = nib.load(join(subject.data_dir[self.pipeline + self.reg_algorithm], 'deformations', filename_svf))
#                 svf = -np.asarray(tp_ref_svf.dataobj)
#
#             svf = svf.astype('float32')
#
#         return svf, tp_ref_svf.affine
#
#     def compute_1x1x1_inputs(self, subject, tp_ref, tp_flo, proxyref, proxyimage_flo=None, proxyseg_flo=None,
#                                im_mode='bilinear', seg_mode='bilinear'):
#         svf_dict = {'sub': subject.id, 'suffix': 'svf'}
#
#         # Reference
#         run_flag_ref = any(['run' in f for f in tp_ref.files['bids']])
#         aff_dict_ref = {'sub': subject.id, 'desc': 'aff'}
#         if run_flag_ref: aff_dict_ref['run'] = '01'
#
#         cog_ref = np.load(join(tp_ref.data_dir['sreg-lin'], tp_ref.get_files(**self.conditions_cog)[0]))
#         affine_matrix_ref = np.load(join(tp_ref.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict_ref}) + '.npy'))
#         affine_matrix_ref[:3, 3] += cog_ref[:3]
#         # v2r_ref_subject = np.matmul(np.linalg.inv(affine_matrix_ref), v2r_ref)
#
#         # Floating image
#         aff_dict_flo = {'sub': subject.id, 'desc': 'aff'}
#         run_flag_flo = any(['run' in f for f in tp_flo.files['bids']])
#         if run_flag_flo: aff_dict_flo['run'] = '01'
#
#         cog_flo = np.load(join(tp_flo.data_dir['sreg-lin'], tp_flo.get_files(**self.conditions_cog)[0]))
#         affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_flo}) + '.npy'))
#         affine_matrix_flo[:3, 3] += cog_flo[:3]
#
#
#         if self.def_scope != 'sreg-lin':
#             svf, v2r_svf = self.get_svf(subject, tp_ref, tp_flo, svf_dict)
#             proxysvf = nib.Nifti1Image(svf.astype('float32'), v2r_svf.affine)
#             proxyflow = def_utils.integrate_svf(proxysvf)
#         else:
#             proxyflow = None
#
#         # Deform
#         v2r_ref = np.matmul(np.linalg.inv(affine_matrix_ref), proxyref.affine)
#         proxyref_aligned = nib.Nifti1Image(np.zeros(proxyref.shape), v2r_ref)
#         im_resampled = None
#         seg_resampled = None
#
#         if proxyimage_flo is not None:
#             v2r_target = np.matmul(np.linalg.inv(affine_matrix_flo), proxyimage_flo.affine)
#             proxyflo_im = nib.Nifti1Image(np.array(proxyimage_flo.dataobj), v2r_target)
#             im_mri = def_utils.vol_resample(proxyref_aligned, proxyflo_im, proxyflow=proxyflow, mode=im_mode)
#
#             im_resampled = np.array(im_mri.dataobj)
#             del im_mri
#
#         if proxyseg_flo is not None:
#             v2r_seg_target = np.matmul(np.linalg.inv(affine_matrix_flo), proxyseg_flo.affine)
#
#             if len(proxyseg_flo.shape) > 3:
#                 seg_list = []
#                 for it_c in range(0, proxyseg_flo.shape[-1], self.channel_chunk):
#                     proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj[..., it_c:it_c+self.channel_chunk]), v2r_seg_target)
#                     seg_mri = def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode)
#
#                     seg_list += [np.array(seg_mri.dataobj)]
#
#                     del seg_mri
#
#                 seg_resampled = np.concatenate(seg_list, axis=-1)
#             else:
#                 proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj), v2r_seg_target)
#                 seg_resampled = np.array(def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode).dataobj)
#
#         return im_resampled, seg_resampled
#
# class FSSubfields_Lin(LabelFusionSubfields):
#
#     def compute_1x1x1_inputs(self, subject, tp, tp_flo, ref_shape, **kwargs):
#         # proxytemplate = nib.load(join(subject.data_dir['sreg-lin'], 'sub-' + subject.id + '_desc-linTemplate_T1w.nii.gz'))
#         proxyimage = tp.get_image(**self.conditions_image)
#
#         # Reference
#         run_flag_ref = any(['run' in f for f in tp.files['bids']])
#         aff_dict_ref = {'sub': subject.id, 'desc': 'aff'}
#         if run_flag_ref: aff_dict_ref['run'] = '01'
#
#         cog_ref = np.load(join(tp.data_dir['sreg-lin'], tp.get_files(**self.conditions_cog)[0]))
#         affine_matrix_ref = np.load(join(tp.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict_ref}) + '.npy'))
#         affine_matrix_ref[:3, 3] += cog_ref[:3]
#         v2r_ref = proxyimage.affine
#         v2r_ref_subject = np.matmul(np.linalg.inv(affine_matrix_ref), v2r_ref)
#
#         # Floating image
#         aff_dict_flo = {'sub': subject.id, 'desc': 'aff'}
#         run_flag_flo = any(['run' in f for f in tp_flo.files['bids']])
#         if run_flag_flo: aff_dict_flo['run'] = '01'
#
#         cog_flo = np.load(join(tp_flo.data_dir['sreg-lin'], tp_flo.get_files(**self.conditions_cog)[0]))
#         affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_flo}) + '.npy'))
#         affine_matrix_flo[:3, 3] += cog_flo[:3]
#
#         # Computing the rasMosaic
#         ii = np.arange(0, ref_shape[0], dtype='int32')
#         jj = np.arange(0, ref_shape[1], dtype='int32')
#         kk = np.arange(0, ref_shape[2], dtype='int32')
#
#         II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')
#
#         I = II.reshape(-1, 1)
#         J = JJ.reshape(-1, 1)
#         K = KK.reshape(-1, 1)
#         C = np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')
#         voxMosaic_ref = np.concatenate((I, J, K, C), axis=1).T
#         rasMosaic_ref = np.dot(v2r_ref_subject, voxMosaic_ref).astype('float32')
#
#         del voxMosaic_ref, ii, jj, kk, II, JJ, KK, I, J, K, C
#
#         rasMosaic_targ = np.matmul(affine_matrix_flo, rasMosaic_ref).astype('float32')
#
#         del rasMosaic_ref
#
#         rasMosaic_targ = rasMosaic_targ.reshape((4,) + ref_shape)
#
#         return rasMosaic_targ#, None
#
#     # def compute_1x1x1_inputs_old(self, subject, tp, tp_flo, ref_shape, **kwargs):
#     #     proxyimage = tp.get_image(**self.conditions_image)
#     #
#     #     run_flag_ref = any(['run' in f for f in tp.files['bids']])
#     #     aff_dict_ref = {'sub': subject.id, 'desc': 'aff'}
#     #     if run_flag_ref: aff_dict_ref['run'] = '01'
#     #
#     #     cog_ref = np.load(join(tp.data_dir['sreg-lin'], tp.get_files(**self.conditions_cog)[0]))
#     #     affine_matrix_ref = np.load(join(tp.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict_ref}) + '.npy'))
#     #     affine_matrix_ref[:3, 3] += cog_ref[:3]
#     #     v2r_ref = proxyimage.affine
#     #
#     #     ii = np.arange(0, ref_shape[0], dtype='int32')
#     #     jj = np.arange(0, ref_shape[1], dtype='int32')
#     #     kk = np.arange(0, ref_shape[2], dtype='int32')
#     #
#     #     II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')
#     #
#     #     I = II.reshape(-1, 1)
#     #     J = JJ.reshape(-1, 1)
#     #     K = KK.reshape(-1, 1)
#     #     C = np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')
#     #     voxMosaic_ref = np.concatenate((I, J, K, C), axis=1).T
#     #     rasMosaic_ref = np.dot(v2r_ref, voxMosaic_ref).astype('float32')
#     #     rasMosaic_sbj = np.dot(np.linalg.inv(affine_matrix_ref), rasMosaic_ref)
#     #
#     #     del voxMosaic_ref, ii, jj, kk, II, JJ, KK, I, J, K, C
#     #
#     #     aff_dict_flo = {'sub': subject.id, 'desc': 'aff'}
#     #     run_flag_flo = any(['run' in f for f in tp_flo.files['bids']])
#     #     if run_flag_flo: aff_dict_flo['run'] = '01'
#     #
#     #     cog_flo = np.load(join(tp_flo.data_dir['sreg-lin'], tp_flo.get_files(**self.conditions_cog)[0]))
#     #     affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'], io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_flo}) + '.npy'))
#     #     affine_matrix_flo[:3, 3] += cog_flo[:3]
#     #
#     #     rasMosaic_flo = np.matmul(affine_matrix_flo, rasMosaic_sbj).astype('float32')
#     #     rasMosaic_flo = rasMosaic_flo.reshape((4,) + ref_shape)
#     #
#     #
#     #     return rasMosaic_flo, None
#     #

class DiceComputation(object):

    def __init__(self, results_dir, def_scope='sreg-synthmorph', seg_scope='synthseg',  time_marker='time_to_bl_days',
                 template=False):

        self.results_dir = results_dir
        self.seg_scope = seg_scope
        self.def_scope = def_scope
        self.time_marker = time_marker
        self.template = template
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'dseg',
                               'scope': seg_scope, 'extension': 'nii.gz'}

        self.channel_chunk = 20
        self.labels_lut = APARC_TO_ASEG_LUT #{k: it_k for it_k, k in enumerate(ASEG_APARC_ARR)}

    def process_flag(self, subject):
        print('\n  o Reading the input files')
        timepoints = subject.sessions

        if len(timepoints) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return False

        if exists(join(self.results_dir, 'sub-' + subject.id + '_dice.tsv')):
            vol_tsv = io_utils.read_tsv(join(self.results_dir, 'sub-' + subject.id + '_dice.tsv'),
                                        k=['scope', 'num_tp'])
        else:
            return True

        if self.template:
            if any([self.def_scope + '_' + str(it_tp) not in vol_tsv.keys() for it_tp in range(1, len(timepoints))]):
                return True
        else:
            if all([self.def_scope + '_T0' + str(it_tp) not in vol_tsv.keys() for it_tp in range(1, len(timepoints))]) and  \
                    all([self.def_scope + '_TN-1' + str(it_tp) not in vol_tsv.keys() for it_tp in range(1, len(timepoints))]):
                return True

        return False

    def deform_tp(self, subject, ref, tp):

        rdict = {}#{'run': '01'} if any(['run' in f for f in subject.sessions[0].files['bids']]) else {}
        aff_f_ref = io_utils.build_bids_fileame({**{'ses': ref.id, 'sub': subject.id, 'desc': 'aff'}, **rdict}) + '.npy'
        aff_f_flo = io_utils.build_bids_fileame({**{'ses': tp.id, 'sub': subject.id, 'desc': 'aff'}, **rdict}) + '.npy'
        svf_f_ref = io_utils.build_bids_fileame({**{'ses': ref.id, 'sub': subject.id, 'suffix': 'svf'}} ) + '.nii.gz'
        svf_f_flo = io_utils.build_bids_fileame({**{'ses': tp.id, 'sub': subject.id, 'suffix': 'svf'}} ) + '.nii.gz'

        affine_matrix_ref = np.load(join(ref.data_dir['sreg-lin'], aff_f_ref))
        affine_matrix_flo = np.load(join(tp.data_dir['sreg-lin'], aff_f_flo))

        # Load deformations
        if self.def_scope != 'sreg-lin':
            def_scope = self.def_scope if  'sreg' in self.def_scope else 'sreg-' + self.def_scope
            proxysvf_ref = nib.load(join(ref.data_dir[def_scope], svf_f_ref))
            proxysvf_flo = nib.load(join(tp.data_dir[def_scope], svf_f_flo))
            proxysvf = nib.Nifti1Image(np.array(proxysvf_flo.dataobj) - np.array(proxysvf_flo.dataobj), proxysvf_ref.affine)
            proxyflow = def_utils.integrate_svf(proxysvf)

        else:
            proxyflow=None

        proxyref = ref.get_image(**self.conditions_seg)
        v2r_ref = np.matmul(np.linalg.inv(affine_matrix_ref), proxyref.affine)
        seg_ref = np.asarray(proxyref.dataobj)
        proxyref = nib.Nifti1Image(seg_ref, v2r_ref)

        proxyseg = tp.get_image(**self.conditions_seg)
        v2r_seg = np.matmul(np.linalg.inv(affine_matrix_flo), proxyseg.affine)
        seg = np.asarray(proxyseg.dataobj)
        proxyflo_seg = nib.Nifti1Image(seg, v2r_seg)

        seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxyflow=proxyflow, mode='nearest')

        return np.array(seg_mri.dataobj).astype('int32')

    def deform_t_template(self, subject, tp):

        proxyref = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)
        rdict = {}#{'run': '01'} if any(['run' in f for f in subject.sessions[0].files['bids']]) else {}

        aff_f = io_utils.build_bids_fileame({**{'ses': tp.id, 'sub': subject.id, 'desc': 'aff'}, **rdict}) + '.npy'
        svf_f = io_utils.build_bids_fileame({**{'ses': tp.id, 'sub': subject.id, 'suffix': 'svf'}} ) + '.nii.gz'
        affine_matrix = np.load(join(tp.data_dir['sreg-lin'], aff_f))
        if self.def_scope != 'sreg-lin':
            proxysvf = nib.load(join(tp.data_dir[self.def_scope], svf_f))
            proxyflow = def_utils.integrate_svf(proxysvf, factor=2)
        else:
            proxyflow = None

        proxyseg = tp.get_image(**self.conditions_seg)
        v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)

        seg = np.asarray(proxyseg.dataobj)

        proxyflo_seg = nib.Nifti1Image(seg, v2r_seg)
        seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxyflow=proxyflow, mode='nearest')

        return np.array(seg_mri.dataobj).astype('int32')

    def compute_p_label(self, subject):

        timepoints = subject.sessions
        timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))
        timepoints.sort(key=lambda t: float(t.sess_metadata[self.time_marker]))

        p_label_full = []
        for it_ref, ref in enumerate([timepoints[0], timepoints[-1]]):
            proxyref = ref.get_image(**self.conditions_seg)

            p_label = np.zeros(proxyref.shape + (len(np.unique(list(self.labels_lut.values()))),), dtype='float32')

            for it_tp, tp in enumerate(timepoints):
                if ref.id == tp.id:
                    seg_resampled = np.asarray(proxyref.dataobj)

                else:
                    seg_resampled = self.deform_tp(subject, ref, tp)

                seg_resampled[np.isnan(seg_resampled)] = 0
                p_tp = fn_utils.one_hot_encoding(seg_resampled, categories=self.labels_lut)
                p_label += p_tp

            p_label_full += [p_label]

        return p_label_full

    def compute_p_label_template(self, subject):

        p_label = np.zeros(subject.image_shape + (len(np.unique(list(self.labels_lut.values()))),), dtype='float32')

        timepoints = subject.sessions
        #timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        for it_tp, tp in enumerate(timepoints):

            seg_resampled = self.deform_t_template(subject, tp)
            seg_resampled[np.isnan(seg_resampled)] = 0

            p_tp = fn_utils.one_hot_encoding(seg_resampled, categories=self.labels_lut)
            p_label += p_tp

        return p_label

    def compute_dice(self, subject, force_flag=False):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'
        if not self.process_flag(subject) and not force_flag:
            return

        print('  o Computing the ovelap on template.')
        t_0 = time.time()
        st_vols = []
        if self.template:
            p_label = self.compute_p_label_template(subject)
            p_label_union = p_label.copy() > 0
            p_label_union = p_label_union.reshape(-1, p_label.shape[-1])

            for it_tp in range(1, len(subject.sessions)):
                true_vol = p_label.copy() > it_tp - 1
                jacc_idx = np.sum(true_vol.reshape(-1, p_label.shape[-1]), axis=0) / np.sum(p_label_union, axis=0)
                jacc_idx[np.isnan(jacc_idx)] = 0

                st_vols_dict = {k: jacc_idx[..., v] for k, v in self.labels_lut.items()}
                st_vols_dict['id'] = subject.id
                st_vols_dict['scope'] = self.def_scope
                st_vols_dict['num_tp'] = it_tp
                st_vols += [st_vols_dict]
        else:
            p_label_full = self.compute_p_label(subject)

            for it_ref, tp_str in enumerate(['T0', 'TN-1']):
                p_label = p_label_full[it_ref]
                p_label_union = p_label.copy() > 0
                p_label_union = p_label_union.reshape(-1, p_label.shape[-1])

                for it_tp in range(1, len(subject.sessions)):
                    true_vol = p_label.copy() > it_tp - 1
                    jacc_idx = np.sum(true_vol.reshape(-1, p_label.shape[-1]), axis=0) / np.sum(p_label_union, axis=0)
                    jacc_idx[np.isnan(jacc_idx)] = 0

                    st_vols_dict = {k: jacc_idx[..., v] for k, v in self.labels_lut.items()}
                    st_vols_dict['id'] = subject.id
                    st_vols_dict['scope'] = self.def_scope + '-' + tp_str
                    st_vols_dict['num_tp'] = it_tp
                    st_vols += [st_vols_dict]


        fieldnames = ['id', 'scope', 'num_tp'] + [k for k in ASEG_APARC_ARR]
        vols_dir = join(self.results_dir, 'sub-' + subject.id + '_dice.tsv')

        write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

        t_1 = time.time()

        print('Subject: ' + str(subject.id) + ' DONE. ' + str(t_1 - t_0) + ' seconds.')
        print('')

class DiceComputationDirect(object):

    def __init__(self, results_dir, def_scope='sreg-synthmorph', seg_scope='synthseg',  time_marker='time_to_bl_days'):

        self.results_dir = results_dir
        self.seg_scope = seg_scope
        self.def_scope = def_scope
        self.time_marker = time_marker
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'dseg',
                               'scope': seg_scope, 'extension': 'nii.gz'}

        self.channel_chunk = 20
        self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_APARC_ARR)}

    def process_flag(self, subject):
        print('\n  o Reading the input files')
        timepoints = subject.sessions

        if len(timepoints) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return False

        if exists(join(self.results_dir, 'sub-' + subject.id + '_dice.tsv')):
            vol_tsv = io_utils.read_tsv(join(self.results_dir, 'sub-' + subject.id + '_dice.tsv'),
                                        k=['scope', 'num_tp'])
        else:
            return True

        if any([self.def_scope + '-T0' '_' + str(it_tp) not in vol_tsv.keys() or
                self.def_scope + '-TN-1' '_' + str(it_tp) not in vol_tsv.keys() for it_tp in range(1, len(timepoints))]):
            return True

        return False


    def deform_tp(self, subject, ref, tp):

        rdict = {'run': '01'} if any(['run' in f for f in subject.sessions[0].files['bids']]) else {}
        aff_f_ref = io_utils.build_bids_fileame({**{'ses': ref.id, 'sub': subject.id, 'desc': 'aff'}, **rdict}) + '.npy'
        aff_f_flo = io_utils.build_bids_fileame({**{'ses': tp.id, 'sub': subject.id, 'desc': 'aff'}, **rdict}) + '.npy'
        affine_matrix_ref = np.load(join(ref.data_dir['sreg-lin'], aff_f_ref))
        affine_matrix_flo = np.load(join(tp.data_dir['sreg-lin'], aff_f_flo))

        # Load deformations
        if self.def_scope == 'lin':
            aff_f_r = join(subject.data_dir['sreg-' + self.def_scope], 'deformations', ref.id + '_to_' + tp.id + '.aff')
            aff_f_r_rev = join(subject.data_dir['sreg-' + self.def_scope], 'deformations', tp.id + '_to_' + ref.id + '.aff')
            proxyflow = None
            affine_matrix_flo = np.eye(4)
            if exists(aff_f_r):
                affine_matrix_ref = io_utils.read_affine_matrix(aff_f_r, full=True)
            else:
                affine_matrix_ref = io_utils.read_affine_matrix(aff_f_r_rev, full=True)

        else:
            svf_filename_ref = ref.id + '_to_' + tp.id + '.svf.nii.gz'
            svf_filename_ref_rev = tp.id + '_to_' + ref.id + '.svf.nii.gz'
            if exists(join(subject.data_dir['sreg-' + self.def_scope], 'deformations', svf_filename_ref)):
                proxysvf = nib.load(join(subject.data_dir['sreg-' + self.def_scope], 'deformations', svf_filename_ref))
                proxyflow = def_utils.integrate_svf(proxysvf)
            else:
                tp_svf = nib.load(join(subject.data_dir['sreg-' + self.def_scope], 'deformations', svf_filename_ref_rev))
                svf = -np.asarray(tp_svf.dataobj)
                proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
                proxyflow = def_utils.integrate_svf(proxysvf)

        proxyref = ref.get_image(**self.conditions_seg)
        v2r_ref = np.matmul(np.linalg.inv(affine_matrix_ref), proxyref.affine)
        seg_ref = np.asarray(proxyref.dataobj)
        proxyref = nib.Nifti1Image(seg_ref, v2r_ref)

        proxyseg = tp.get_image(**self.conditions_seg)
        v2r_seg = np.matmul(np.linalg.inv(affine_matrix_flo), proxyseg.affine)
        seg = np.asarray(proxyseg.dataobj)
        proxyflo_seg = nib.Nifti1Image(seg, v2r_seg)

        seg_mri = def_utils.vol_resample(proxyref, proxyflo_seg, proxyflow=proxyflow, mode='nearest')

        return np.array(seg_mri.dataobj).astype('int32')

    def compute_p_label(self, subject):


        timepoints = subject.sessions
        timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))
        timepoints.sort(key=lambda t: float(t.sess_metadata[self.time_marker]))

        p_label_full = []
        for it_ref, ref in enumerate([timepoints[0], timepoints[-1]]):
            proxyref = ref.get_image(**self.conditions_seg)

            p_label = np.zeros(proxyref.shape + (len(self.labels_lut), ), dtype='float32')

            for it_tp, tp in enumerate(timepoints):
                if ref.id == tp.id:
                    seg_resampled = np.asarray(proxyref.dataobj)

                else:
                    seg_resampled = self.deform_tp(subject, ref, tp)

                seg_resampled[np.isnan(seg_resampled)] = 0
                p_tp = fn_utils.one_hot_encoding(seg_resampled, categories=list(self.labels_lut.keys()))
                p_label += p_tp

            p_label_full += [p_label]
        return p_label_full

    def compute_dice(self, subject, force_flag=False):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'
        if not self.process_flag(subject) and not force_flag:
            return

        print('  o Computing the ovelap on template T(0) and T(N-1)')
        t_0 = time.time()
        p_label_full = self.compute_p_label(subject)

        st_vols = []
        for it_ref, tp_str in enumerate(['T0', 'TN-1']):
            p_label = p_label_full[it_ref]
            p_label_union = p_label.copy() > 0
            p_label_union = p_label_union.reshape(-1, p_label.shape[-1])

            for it_tp in range(1, len(subject.sessions)):
                true_vol = p_label.copy() > it_tp - 1
                jacc_idx = np.sum(true_vol.reshape(-1, p_label.shape[-1]), axis=0) / np.sum(p_label_union, axis=0)
                jacc_idx[np.isnan(jacc_idx)] = 0

                st_vols_dict = {k: jacc_idx[..., v] for k, v in self.labels_lut.items()}
                st_vols_dict['id'] = subject.id
                st_vols_dict['scope'] = self.def_scope + '-' + tp_str
                st_vols_dict['num_tp'] = it_tp
                st_vols += [st_vols_dict]


        fieldnames = ['id', 'scope', 'num_tp'] + [k for k in ASEG_APARC_ARR]
        vols_dir = join(self.results_dir, 'sub-' + subject.id + '_dice.tsv')

        write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

        t_1 = time.time()

        print('Subject: ' + str(subject.id) + '. DONE. ' + str(t_1 - t_0) + ' seconds.')
        print('')

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


