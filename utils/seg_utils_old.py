import csv
import pdb
import statistics
from os.path import exists, join
import time
import shutil

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.special import softmax

from setup import *
from utils.labels import *
from utils import io_utils, fn_utils,  def_utils

class LabelFusion(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None, normalise=True,
                 interpmethod='linear', save=False, process=False, type_map=None, fusion_method='post', jlf=False):

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

        seg_suffix = 'dseg' if type_map is not None else 'post'
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': seg_suffix, 'scope': seg_scop, 'extension': 'nii.gz'}

        if seg_scop == 'freesurfer':
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_LUT)}
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

                    seg_list += [np.array(seg_mri.dataobj)]

                    del seg_mri

                seg_resampled = np.concatenate(seg_list, axis=-1)
            else:
                proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj), v2r_seg_target)
                seg_resampled = np.array(def_utils.vol_resample(proxyref_aligned, proxyflo_seg, proxyflow=proxyflow, mode=seg_mode).dataobj)

        return im_resampled, seg_resampled

    def compute_seg_map(self, tp, distance=True):
        proxyimage = tp.get_image(**self.conditions_image)
        proxyseg = tp.get_image(**self.conditions_seg)
        v2r_ref = proxyimage.affine.copy()

        if self.type_map == 'distance_map':
            proxyseg = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            if distance:
                seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32',
                                                                                                            copy=False)

        elif self.type_map == 'onehot_map':
            proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg_flo.dataobj)
            mask = seg > 0
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)

        elif self.type_map == 'gauss_map':
            from scipy.ndimage import gaussian_filter
            proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg_flo.dataobj)
            mask = seg > 0
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
        im_resampled, seg_resampled = self.register_timepoints_st(subject, tp_ref, tp_flo, proxyim_ref, proxyimage_flo=proxyimage, proxyseg_flo=proxyseg, seg_mode=seg_mode)

        return im_resampled, seg_resampled

    def compute_p_label(self, tp, subject, timepoints, time_list, mode='linear'):

        # proxyimage = tp.get_image(**self.conditions_image)

        proxyimage, proxyseg = self.compute_seg_map(tp)
        v2r_ref = proxyimage.affine
        p_data = np.zeros(proxyimage.shape + (len(timepoints),), dtype='float32')
        p_label = np.zeros(proxyimage.shape + (len(self.labels_lut),len(timepoints)), dtype='float32')
        # p_data = {t_var: {sp_var: np.zeros(proxyimage.shape + (len(timepoints),), dtype='float32') for sp_var in self.spatial_variance} for t_var in self.temp_variance}
        # p_label = {t_var: {sp_var: np.zeros(proxyimage.shape + (len(self.labels_lut),), dtype='float32') for sp_var in self.spatial_variance} for t_var in self.temp_variance}
        im = np.array(proxyimage.dataobj)
        im = gaussian_filter(im, sigma=0.5)
        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp.id == tp_flo.id:
                # proxyseg_ref = tp.get_image(**self.conditions_seg)
                #
                # # pdb.set_trace()
                # # _, crop_coord = fn_utils.crop_label(np.array(proxyseg_ref)>0, margin=10)
                # if self.type_map == 'distance_map':
                #     seg = np.array(proxyseg_ref.dataobj)
                #     seg[seg > 1999] = 42
                #     seg[seg > 999] = 3
                #     seg_resampled = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype( 'float32', copy=False)
                #     proxyseg = nib.Nifti1Image(seg_resampled, proxyseg_ref.affine)
                #     proxyseg = def_utils.vol_resample(proxyimage, proxyseg)#, mode='nearest')
                #     seg_resampled = np.array(proxyseg.dataobj)
                #
                # elif self.type_map == 'onehot_map':
                #     seg = np.array(proxyseg_ref.dataobj)
                #     seg[seg > 1999] = 42
                #     seg[seg > 999] = 3
                #     seg_resampled = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)
                #     proxyseg = nib.Nifti1Image(seg_resampled, proxyseg_ref.affine)
                #     proxyseg = def_utils.vol_resample(proxyimage, proxyseg)#, mode='nearest')
                #     seg_resampled = np.array(proxyseg.dataobj)
                #
                # else:
                #     if np.sum(proxyseg_ref.affine - proxyimage.affine) > 1e-4:
                #         proxyseg = def_utils.vol_resample(proxyimage, proxyseg_ref, mode='linear')
                #         seg_resampled = np.array(proxyseg.dataobj)
                #     else:
                #         seg_resampled = np.array(proxyseg_ref.dataobj)

                # _, proxyseg = self.compute_seg_map(tp)
                seg_resampled = np.array(proxyseg.dataobj)
                im_resampled = im#np.array(proxyimage.dataobj)
                # im_resampled = gaussian_filter(im_resampled, sigma=0.5)


            else:
                im_resampled, seg_resampled = self.process_tp(subject, tp, tp_flo, proxyimage)

                # mean_age_2 = (time_list[tp_flo.id] - time_list[tp.id]) ** 2
                # if im_resampled is not None:
                #     mean_im_2 = (im_resampled - im) ** 2
                #     ###
                #     # img = nib.Nifti1Image(mean_im_2, proxyimage.affine)
                #     # nib.save(img, 'mean_im_2.nii.gz')
                #     # mean_im_2_mask = (mean_im_2 < 10) & (np.sum(seg_resampled[..., 1:]) > 0.5)
                #     # mean_im_2_filt = mean_im_2[mean_im_2_mask]
                #     # plt.figure()
                #     # ybins, xbins, _ = plt.hist(mean_im_2_filt, bins=100)#, density=True)
                #     # plt.ylim(0, np.percentile(ybins, 95)*1.1) # plt.ylim(0, 0.5)
                #     # plt.grid(True)
                #     # plt.savefig(tp_flo.id + '_mean_im_2_hist.png')
                #     # plt.close()
                #     ##
                #     if self.smooth is not None:
                #         kern = np.ones((self.smooth, self.smooth, self.smooth)) / (self.smooth*self.smooth*self.smooth)
                #         mean_im_2 = convolve(mean_im_2, kern, mode='constant')
                #         ###
                #         # img = nib.Nifti1Image(mean_im_2, np.eye(4))
                #         # nib.save(img, 'mean_im_2_gauss.nii.gz')
                #         # mean_im_2_filt = mean_im_2[mean_im_2_mask &  (mean_im_2 < 10)]
                #         # plt.figure()
                #         # ybins, xbins, _ = plt.hist(mean_im_2_filt, bins=100)#, density=True)
                #         # plt.ylim(0, np.percentile(ybins, 95)*1.1)
                #         # plt.xlim(0, 10)
                #         # plt.grid()
                #         # plt.savefig(tp_flo.id + '_mean_im_2_hist_gauss.png')
                #         # plt.close()
                #         ###
                # else:
                #     mean_im_2 = None


            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            if self.fusion_method == 'seg':
                num_classes = seg_resampled.shape[-1]
                seg_resampled_max = np.argmax(seg_resampled, axis=-1)
                seg_resampled = fn_utils.one_hot_encoding(seg_resampled_max, num_classes=num_classes).astype('float32', copy=False)

            p_data[..., it_tp_flo] = im_resampled
            p_label[..., it_tp_flo] = seg_resampled
            del im_resampled, seg_resampled
            # for t_var in self.temp_variance:
            #     t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
            #     for s_var in self.spatial_variance:
            #         s_ker = 1 if s_var == 'inf' else np.exp(-0.5 / s_var * mean_im_2)
            #         p_data[t_var][s_var][..., it_tp_flo] = s_ker * t_ker
            #         p_label[t_var][s_var] = np.add(p_label[t_var][s_var],
            #                                        p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled,
            #                                        out=p_label[t_var][s_var])
        # img = nib.Nifti1Image(p_label[..., self.labels_lut[17], :], proxyimage.affine)
        # nib.save(img, 'p_label.nii.gz')
        # img = nib.Nifti1Image(p_data, proxyimage.affine)
        # nib.save(img, 'p_data.nii.gz')
        # pdb.set_trace()
        # if self.normalise:
        #     for t_var in self.temp_variance:
        #         for s_var in self.spatial_variance:
        #             p_label[t_var][s_var] = np.divide(p_label[t_var][s_var], np.sum(p_data[t_var][s_var], axis=-1, keepdims=True))
        #             p_label[t_var][s_var][np.isnan(p_label[t_var][s_var])] = 0


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
            mean_im_2, seg_res, aff = self.compute_p_label(tp, subject, timepoints, time_list)
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
                    if np.float(s_var) > 0 and np.float(t_var) > 0:
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

class LabelFusionTemplate(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None, normalise=True,
                 interpmethod='linear', save=False, process=False, type_map=None, fusion_method='post', jlf=False):

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

        if seg_scop == 'freesurfer':
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_LUT)}
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

                    seg_list += [np.array(seg_mri.dataobj)]

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
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            if distance:
                seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32',
                                                                                                            copy=False)

        elif self.type_map == 'onehot_map':
            proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg_flo.dataobj)
            mask = seg > 0
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)

        elif self.type_map == 'gauss_map':
            from scipy.ndimage import gaussian_filter
            proxyseg_flo = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
            seg = np.array(proxyseg_flo.dataobj)
            mask = seg > 0
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
        p_label = np.zeros(proxyref.shape + (len(self.labels_lut),len(timepoints)), dtype='float32')

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
                im = gaussian_filter(im, sigma=0.5)

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
                    if np.float(s_var) > 0 and np.float(t_var) > 0:
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

class LabelFusionSubject(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None,
                 interpmethod='linear', save=False, process=False, type_map=None, jlf=False):

        self.seg_scope = seg_scop
        self.def_scope = def_scope
        self.results_scope = def_scope + '-temp' if seg_scop == 'synthseg' else def_scope + '-' + seg_scop + '-temp'
        self.results_lin_scope = 'sreg-lin' + '-temp' if seg_scop == 'synthseg' else 'sreg-lin-' + seg_scop + '-temp'
        self.temp_variance = temp_variance if temp_variance is not None else ['inf']
        self.spatial_variance = spatial_variance if spatial_variance is not None else ['inf']
        self.floor_variance = floor_variance if floor_variance is not None else [1]
        self.save = save
        self.process = process
        self.smooth = smooth
        self.time_marker = time_marker
        self.type_map = type_map
        self.jlf = jlf

        seg_suffix = 'dseg' if type_map is not None else 'post'
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': seg_suffix,
                               'scope': seg_scop, 'extension': 'nii.gz'}

        if seg_scop == 'freesurfer':
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_LUT)}
        else:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_ARR)}

        if type_map == 'distance_map':
            self.interpmethod = 'seg-' + interpmethod + '-post'
        elif type_map == 'onehot_map':
            self.interpmethod = 'onehot-' + interpmethod + '-post'
        else:
            self.interpmethod = 'post-' + interpmethod + '-post'

        if jlf: self.interpmethod = self.interpmethod + '-jlf'

        self.channel_chunk = -1

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions
        if not (all([tv == 'inf' for tv in self.temp_variance]) or self.save):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        if len(timepoints) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return None, None

        if 'sreg' in self.def_scope and 'sreg-lin' not in self.def_scope:
            svf_dict = {'sub': subject.id, 'suffix': 'svf'}
            for tp in timepoints:
                if not exists(join(tp.data_dir[self.def_scope],
                                   io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz')):
                    print('Subject: ' + str(
                        subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                    return None, None

        if self.save and not force_flag:
            f = timepoints[-1].id + '_to_' + timepoints[-1].id + '_post.nii.gz'
            if (self.type_map is None and exists(join(subject.data_dir[self.def_scope], 'segmentations', f))) or \
                    (self.type_map == 'distance_map' and exists(
                        join(subject.data_dir[self.def_scope], 'segmentations_distance', f))) \
                    or (self.type_map == 'onehot_map' and exists(
                join(subject.data_dir[self.def_scope], 'segmentations_onehot', f))):
                print('Subject: ' + str(subject.id) + '. DONE')
                return None, None
            else:
                return timepoints, []
        else:
            vol_tsv = {}
            for tp in timepoints:
                last_dir = tp.data_dir[self.results_scope]
                if exists(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')):
                    vol_tsv[tp.id] = io_utils.read_tsv(
                        join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'),
                        k=['id', 't_var', 's_var', 'interpmethod'])

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

                    elif any([(filename[:-7] + '_' + str(t) + 'f' + str(f) + '_' + str(
                            s) + '_' + self.interpmethod not in vol_tsv[tp.id].keys() and f != 1)
                              or (filename[:-7] + '_' + str(t) + '_' + str(s) + '_' + self.interpmethod not in vol_tsv[
                        tp.id].keys() and f == 1) for f in self.floor_variance for t in self.temp_variance for s in
                              self.spatial_variance]):
                        timepoints_to_run.append(tp)

        if not timepoints_to_run:
            print('Subject: ' + str(subject.id) + '. DONE')
            return None, None

        if all([tv == 'inf' for tv in self.temp_variance]) or self.save:
            time_list = {tp.id: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                time_list[tp.id] = float(tp.sess_metadata[self.time_marker])

        return timepoints_to_run, time_list

    def reg2temp(self, proxyref, proxyim_flo, proxyseg_flo):

        if self.type_map == 'distance_map':
            seg = np.array(proxyseg_flo.dataobj)
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32', copy=False)
            proxyseg_flo = nib.Nifti1Image(seg_flo, proxyseg_flo.affine)

        elif self.type_map == 'onehot_map':
            seg = np.array(proxyseg_flo.dataobj)
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)
            proxyseg_flo = nib.Nifti1Image(seg_flo, proxyseg_flo.affine)

        else:
            pass

        proxyseg_res = def_utils.vol_resample(proxyref, proxyseg_flo)
        proxyim_res = def_utils.vol_resample(proxyref, proxyim_flo)

        return proxyim_res, proxyseg_res

    def resample_lin(self, subject):
        # Def parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}

        # Ref parameters
        proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)

        for tp in subject.sessions:
            # if exists(join(tp.data_dir[self.results_lin_scope], tp.id + '.seg.nii.gz')): continue
            aff_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'

            proxyimage = tp.get_image(**self.conditions_image)
            proxyseg = tp.get_image(**self.conditions_seg)

            affine_matrix = np.load(join(tp.data_dir['sreg-lin'], aff_filename))

            v2r_im = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)
            v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)
            proxyimage = nib.Nifti1Image(np.array(proxyimage.dataobj), v2r_im)
            proxyseg = nib.Nifti1Image(np.array(proxyseg.dataobj), v2r_seg)

            proxyimage_res, proxyseg_res = self.reg2temp(proxytemplate, proxyimage, proxyseg)

            nib.save(proxyimage_res, join(tp.data_dir[self.results_lin_scope], tp.id + '.im.nii.gz'))
            nib.save(proxyseg_res, join(tp.data_dir[self.results_lin_scope], tp.id + '.seg.nii.gz'))

    def subject2temp(self, subject, tp):
        aff_dict = {'sub': subject.id, 'desc': 'aff'}
        aff_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'
        affine_matrix = np.load(join(tp.data_dir['sreg-lin'], aff_filename))

        proxyimage = tp.get_image(**self.conditions_image)#nib.load(join(tp_ref.data_dir[self.results_lin_scope], tp_ref.id + '.im.nii.gz'))
        proxyseg = tp.get_image(**self.conditions_seg)
        v2r_im = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)
        v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)

        if self.type_map == 'distance_map':
            seg = np.array(proxyseg.dataobj)
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32',
                                                                                                            copy=False)
            proxyseg = nib.Nifti1Image(seg_flo, proxyseg.affine)

        elif self.type_map == 'onehot_map':
            seg = np.array(proxyseg.dataobj)
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)
            proxyseg = nib.Nifti1Image(seg, proxyseg.affine)

        else:
            pass

        proxyimage = nib.Nifti1Image(np.array(proxyimage.dataobj), v2r_im)
        proxyseg = nib.Nifti1Image(np.array(proxyseg.dataobj), v2r_seg)

        # proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)
        # proxyimage_ref, proxyseg_ref = self.reg2temp(proxytemplate, proxyimage_ref, proxyseg_ref)
        # im = np.array(proxyimage_ref.dataobj)
        return proxyimage, proxyseg

    def compute_p_label(self, subject, tp_ref, time_list):
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        p_data = {
            t_var: {sp_var: np.zeros(subject.image_shape + (len(subject.sessions),), dtype='float32') for sp_var in
                    self.spatial_variance} for t_var in self.temp_variance}
        p_label = {
            t_var: {sp_var: np.zeros(subject.image_shape + (len(self.labels_lut),), dtype='float32') for sp_var in
                    self.spatial_variance} for t_var in self.temp_variance}

        # Reference images
        proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)
        proxyimage_ref, proxyseg_ref = self.subject2temp(subject, tp_ref)

        proxyimage_ref = def_utils.vol_resample(proxytemplate, proxyimage_ref)
        proxyseg_ref = def_utils.vol_resample(proxytemplate, proxyseg_ref)

        im = np.array(proxyimage_ref.dataobj)

        proxysvf_ref = None
        if 'lin' not in self.def_scope:
            svf_filename_ref = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'
            proxysvf_ref = nib.load(join(tp_ref.data_dir[self.def_scope], svf_filename_ref))

        for it_tp_flo, tp_flo in enumerate(subject.sessions):

            if tp_ref.id == tp_flo.id:
                seg_resampled = np.array(proxyseg_ref.dataobj)
                mean_age_2 = 0
                mean_im_2 = np.zeros(proxyimage_ref.shape, dtype='float32')

            else:
                proxyimage_flo, proxyseg_flo = self.subject2temp(subject, tp_flo)

                if 'lin' in self.def_scope:
                    proxyimage_flo = def_utils.vol_resample(proxytemplate, proxyimage_flo)
                    proxyseg_flo = def_utils.vol_resample(proxytemplate, proxyseg_flo)
                    im_resampled = np.array(proxyimage_flo.dataobj)
                    seg_resampled = np.array(proxyseg_flo.dataobj)

                else:
                    svf_filename_flo = io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **svf_dict}) + '.nii.gz'
                    proxysvf_flo = nib.load(join(tp_flo.data_dir[self.def_scope], svf_filename_flo))
                    svf = np.asarray(proxysvf_flo.dataobj) - np.asarray(proxysvf_ref.dataobj)
                    proxysvf = nib.Nifti1Image(svf.astype('float32'), proxysvf_ref.affine)
                    proxyflow = def_utils.integrate_svf(proxysvf)

                    # Deform
                    im_mri = def_utils.vol_resample(proxytemplate, proxyimage_flo, proxyflow=proxyflow)
                    im_resampled = np.array(im_mri.dataobj)

                    if self.channel_chunk == -1:
                        seg_mri = def_utils.vol_resample(proxytemplate, proxyseg_flo, proxyflow=proxyflow)
                        seg_resampled = np.array(seg_mri.dataobj)
                    else:
                        seg_list = []
                        for it_c in range(0, proxyseg_flo.shape[-1], self.channel_chunk):
                            seg_chunk = np.array(proxyseg_flo.dataobj[..., it_c:it_c + self.channel_chunk])
                            proxyflo_seg = nib.Nifti1Image(seg_chunk, proxyseg_flo.affine)
                            seg_mri = def_utils.vol_resample(proxytemplate, proxyflo_seg, proxyflow=proxyflow)

                            seg_list += [np.array(seg_mri.dataobj)]

                            del seg_mri

                        seg_resampled = np.concatenate(seg_list, axis=-1)

                mean_age_2 = (time_list[tp_flo.id] - time_list[tp_ref.id]) ** 2
                if im_resampled is not None:
                    mean_im_2 = (im_resampled - im) ** 2
                    if self.smooth is not None:
                        kern = np.ones((self.smooth, self.smooth, self.smooth)) / (
                                    self.smooth * self.smooth * self.smooth)
                        mean_im_2 = convolve(mean_im_2, kern, mode='constant')
                else:
                    mean_im_2 = None

                del im_resampled

            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            for t_var in self.temp_variance:
                t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
                for s_var in self.spatial_variance:
                    s_ker = 1 if s_var == 'inf' else np.exp(-0.5 / s_var * mean_im_2)
                    p_data[t_var][s_var][..., it_tp_flo] = s_ker * t_ker
                    p_label[t_var][s_var] = np.add(p_label[t_var][s_var],
                                                   p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled,
                                                   out=p_label[t_var][s_var])

        for t_var in self.temp_variance:
            for s_var in self.spatial_variance:
                p_label[t_var][s_var] = np.divide(p_label[t_var][s_var],
                                                  np.sum(p_data[t_var][s_var], axis=-1, keepdims=True))
                p_label[t_var][s_var][np.isnan(p_label[t_var][s_var])] = 0

        return {self.floor_variance[0]: p_label}

    def compute_p_label_resample_lin(self, subject, tp_ref, time_list):
        # Def parameters
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        p_data = {
            t_var: {sp_var: np.zeros(subject.image_shape + (len(subject.sessions),), dtype='float32') for sp_var in
                    self.spatial_variance} for t_var in self.temp_variance}
        p_label = {
            t_var: {sp_var: np.zeros(subject.image_shape + (len(self.labels_lut),), dtype='float32') for sp_var in
                    self.spatial_variance} for t_var in self.temp_variance}

        proxyimage_ref = nib.load(join(tp_ref.data_dir[self.results_lin_scope], tp_ref.id + '.im.nii.gz'))
        proxyseg_ref = nib.load(join(tp_ref.data_dir[self.results_lin_scope], tp_ref.id + '.seg.nii.gz'))
        im = np.array(proxyimage_ref.dataobj)

        proxysvf_ref = None
        if 'lin' not in self.def_scope:
            svf_filename_ref = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'
            proxysvf_ref = nib.load(join(tp_ref.data_dir[self.def_scope], svf_filename_ref))

        for it_tp_flo, tp_flo in enumerate(subject.sessions):

            if tp_ref.id == tp_flo.id:
                seg_resampled = np.array(proxyseg_ref.dataobj)
                mean_age_2 = 0
                mean_im_2 = np.zeros(proxyimage_ref.shape, dtype='float32')

            else:
                proxyimage_flo = nib.load(join(tp_flo.data_dir[self.results_lin_scope], tp_flo.id + '.im.nii.gz'))
                proxyseg_flo = nib.load(join(tp_flo.data_dir[self.results_lin_scope], tp_flo.id + '.seg.nii.gz'))

                if 'lin' in self.def_scope:
                    im_resampled = np.array(proxyimage_flo.dataobj)
                    seg_resampled = np.array(proxyseg_flo.dataobj)

                else:
                    svf_filename_flo = io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **svf_dict}) + '.nii.gz'
                    proxysvf_flo = nib.load(join(tp_flo.data_dir[self.def_scope], svf_filename_flo))
                    svf = np.asarray(proxysvf_flo.dataobj) - np.asarray(proxysvf_ref.dataobj)
                    proxysvf = nib.Nifti1Image(svf.astype('float32'), proxysvf_ref.affine)
                    proxyflow = def_utils.integrate_svf(proxysvf)

                    # Deform
                    im_mri = def_utils.vol_resample(proxyimage_ref, proxyimage_flo, proxyflow=proxyflow)
                    im_resampled = np.array(im_mri.dataobj)

                    if self.channel_chunk == -1:
                        seg_mri = def_utils.vol_resample(proxyimage_ref, proxyseg_flo, proxyflow=proxyflow)
                        seg_resampled = np.array(seg_mri.dataobj)
                    else:
                        seg_list = []
                        for it_c in range(0, proxyseg_flo.shape[-1], self.channel_chunk):
                            seg_chunk = np.array(proxyseg_flo.dataobj[..., it_c:it_c + self.channel_chunk])
                            proxyflo_seg = nib.Nifti1Image(seg_chunk, proxyseg_flo.affine)
                            seg_mri = def_utils.vol_resample(proxyimage_ref, proxyflo_seg, proxyflow=proxyflow)

                            seg_list += [np.array(seg_mri.dataobj)]

                            del seg_mri

                        seg_resampled = np.concatenate(seg_list, axis=-1)

                mean_age_2 = (time_list[tp_flo.id] - time_list[tp_ref.id]) ** 2
                if im_resampled is not None:
                    mean_im_2 = (im_resampled - im) ** 2
                    if self.smooth is not None:
                        kern = np.ones((self.smooth, self.smooth, self.smooth)) / (
                                    self.smooth * self.smooth * self.smooth)
                        mean_im_2 = convolve(mean_im_2, kern, mode='constant')
                else:
                    mean_im_2 = None

                del im_resampled

            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            for t_var in self.temp_variance:
                t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
                for s_var in self.spatial_variance:
                    s_ker = 1 if s_var == 'inf' else np.exp(-0.5 / s_var * mean_im_2)
                    p_data[t_var][s_var][..., it_tp_flo] = s_ker * t_ker
                    p_label[t_var][s_var] = np.add(p_label[t_var][s_var],
                                                   p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled,
                                                   out=p_label[t_var][s_var])

        for t_var in self.temp_variance:
            for s_var in self.spatial_variance:
                p_label[t_var][s_var] = np.divide(p_label[t_var][s_var],
                                                  np.sum(p_data[t_var][s_var], axis=-1, keepdims=True))
                p_label[t_var][s_var][np.isnan(p_label[t_var][s_var])] = 0

        return {self.floor_variance[0]: p_label}

    def label_fusion(self, subject, force_flag=False, resample=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'  # 'w' if force_flag else 'a'
        timepoints_to_run, time_list = self.prepare_data(subject, force_flag)
        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        if resample: self.resample_lin(subject)

        for tp in timepoints_to_run:
            t_0 = time.time()
            print('     - Timepoint ' + tp.id, end=':', flush=True)

            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)
            if resample:
                p_label_dict = self.compute_p_label_resample_lin(subject, tp, time_list)
            else:
                p_label_dict = self.compute_p_label(subject, tp, time_list)

            # proxy = tp.get_image(**self.conditions_image)

            aff = subject.vox2ras0
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            st_vols = []
            for f_var in self.floor_variance:
                for t_var in self.temp_variance:
                    for s_var in self.spatial_variance:
                        fp_dict['suffix'] = 'dseg'
                        fp_dict['desc'] = 'f' + str(f_var) + 't' + str(t_var) + 's' + str(s_var) + self.interpmethod
                        filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                        p_label = p_label_dict[f_var][t_var][s_var]
                        p_label[np.isnan(p_label)] = 0

                        fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                        true_vol = np.zeros_like(fake_vol)
                        for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul

                        img = nib.Nifti1Image(true_vol, aff)
                        nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                        vols = get_vols_post(p_label, res=pixdim)
                        st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                        st_vols_dict['id'] = filename[:-7]
                        st_vols_dict['t_var'] = t_var if f_var == 1 else str(t_var) + 'f' + str(f_var)
                        st_vols_dict['s_var'] = s_var
                        st_vols_dict['interpmethod'] = self.interpmethod
                        st_vols_dict['type'] = 'posteriors'

                        st_vols += [st_vols_dict]

                        vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                        st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                        st_vols_dict['id'] = filename[:-7]
                        st_vols_dict['t_var'] = t_var if f_var == 1 else str(t_var) + 'f' + str(f_var)
                        st_vols_dict['s_var'] = s_var
                        st_vols_dict['interpmethod'] = self.interpmethod
                        st_vols_dict['type'] = 'seg'

                        st_vols += [st_vols_dict]
                        del p_label, fake_vol, true_vol

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del p_label_dict

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')

class LabelFusionLinTemplate(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None,
                 interpmethod='linear', save=False, process=False, type_map=None, jlf=False):

        self.seg_scope = seg_scop
        self.def_scope = def_scope
        self.results_scope = def_scope + '-temp' if seg_scop == 'synthseg' else def_scope + '-' + seg_scop + '-temp'
        self.results_lin_scope = 'sreg-lin' + '-temp' if seg_scop == 'synthseg' else 'sreg-lin-' + seg_scop + '-temp'
        self.temp_variance = temp_variance if temp_variance is not None else ['inf']
        self.spatial_variance = spatial_variance if spatial_variance is not None else ['inf']
        self.floor_variance = floor_variance if floor_variance is not None else [1]
        self.save = save
        self.process = process
        self.smooth = smooth
        self.time_marker = time_marker
        self.type_map = type_map
        self.jlf = jlf

        seg_suffix = 'dseg' if type_map is not None else 'post'
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': seg_suffix,
                               'scope': seg_scop, 'extension': 'nii.gz'}

        if seg_scop == 'freesurfer':
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_LUT)}
        else:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_ARR)}

        if type_map == 'distance_map':
            self.interpmethod = 'seg-' + interpmethod + '-post'
        elif type_map == 'onehot_map':
            self.interpmethod = 'onehot-' + interpmethod + '-post'
        else:
            self.interpmethod = 'post-' + interpmethod + '-post'

        if jlf: self.interpmethod = self.interpmethod + '-jlf'

        self.interpmethod += '-jac'
        self.channel_chunk = -1

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions
        if not (all([tv == 'inf' for tv in self.temp_variance]) or self.save):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        if len(timepoints) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return None, None

        if 'sreg' in self.def_scope and 'sreg-lin' not in self.def_scope:
            svf_dict = {'sub': subject.id, 'suffix': 'svf'}
            for tp in timepoints:
                if not exists(join(tp.data_dir[self.def_scope],
                                   io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz')):
                    print('Subject: ' + str(
                        subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                    return None, None

        if self.save and not force_flag:
            f = timepoints[-1].id + '_to_' + timepoints[-1].id + '_post.nii.gz'
            if (self.type_map is None and exists(join(subject.data_dir[self.def_scope], 'segmentations', f))) or \
                    (self.type_map == 'distance_map' and exists(
                        join(subject.data_dir[self.def_scope], 'segmentations_distance', f))) \
                    or (self.type_map == 'onehot_map' and exists(
                join(subject.data_dir[self.def_scope], 'segmentations_onehot', f))):
                print('Subject: ' + str(subject.id) + '. DONE')
                return None, None
            else:
                return timepoints, []
        else:
            vol_tsv = {}
            for tp in timepoints:
                last_dir = tp.data_dir[self.results_scope]
                if exists(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')):
                    vol_tsv[tp.id] = io_utils.read_tsv(
                        join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'),
                        k=['id', 't_var', 's_var', 'interpmethod'])

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

                    elif any([(filename[:-7] + '_' + str(t) + 'f' + str(f) + '_' + str(
                            s) + '_' + self.interpmethod not in vol_tsv[tp.id].keys() and f != 1)
                              or (filename[:-7] + '_' + str(t) + '_' + str(s) + '_' + self.interpmethod not in vol_tsv[
                        tp.id].keys() and f == 1) for f in self.floor_variance for t in self.temp_variance for s in
                              self.spatial_variance]):
                        timepoints_to_run.append(tp)

        if not timepoints_to_run:
            print('Subject: ' + str(subject.id) + '. DONE')
            return None, None

        if all([tv == 'inf' for tv in self.temp_variance]) or self.save:
            time_list = {tp.id: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                time_list[tp.id] = float(tp.sess_metadata[self.time_marker])

        return timepoints_to_run, time_list

    def reg2temp(self, proxyref, proxyim_flo, proxyseg_flo, proxyflow):

        if self.type_map == 'distance_map':
            seg = np.array(proxyseg_flo.dataobj)
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32', copy=False)
            proxyseg_flo = nib.Nifti1Image(seg_flo, proxyseg_flo.affine)

        elif self.type_map == 'onehot_map':
            seg = np.array(proxyseg_flo.dataobj)
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)
            proxyseg_flo = nib.Nifti1Image(seg_flo, proxyseg_flo.affine)

        else:
            pass

        proxyseg_res = def_utils.vol_resample(proxyref, proxyseg_flo, proxyflow)
        proxyim_res = def_utils.vol_resample(proxyref, proxyim_flo, proxyflow)

        return proxyim_res, proxyseg_res

    def resample_images(self, subject):
        # Def parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        # Ref parameters
        proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)

        for tp in subject.sessions:
            if exists(join(tp.data_dir[self.results_scope], tp.id + '.seg.nii.gz')): continue
            aff_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'
            svf_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'

            proxyimage = tp.get_image(**self.conditions_image)
            proxyseg = tp.get_image(**self.conditions_seg)

            affine_matrix = np.load(join(tp.data_dir['sreg-lin'], aff_filename))
            v2r_im = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)
            v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)
            proxyimage = nib.Nifti1Image(np.array(proxyimage.dataobj), v2r_im)
            proxyseg = nib.Nifti1Image(np.array(proxyseg.dataobj), v2r_seg)


            if 'lin' not in self.def_scope:
                proxysvf = nib.load(join(tp.data_dir[self.def_scope], svf_filename))
                proxyflow = def_utils.integrate_svf(proxysvf)
            else:
                proxyflow = None

            proxyimage_res, proxyseg_res = self.reg2temp(proxytemplate, proxyimage, proxyseg, proxyflow)

            nib.save(proxyimage_res, join(tp.data_dir[self.results_scope], tp.id + '.im.nii.gz'))
            nib.save(proxyseg_res, join(tp.data_dir[self.results_scope], tp.id + '.seg.nii.gz'))

    def compute_p_label(self, subject, tp_ref, time_list):
        # Def parameters
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        p_data = {
            t_var: {sp_var: np.zeros(subject.image_shape + (len(subject.sessions),), dtype='float32') for sp_var in
                    self.spatial_variance} for t_var in self.temp_variance}
        p_label = {
            t_var: {sp_var: np.zeros(subject.image_shape + (len(self.labels_lut),), dtype='float32') for sp_var in
                    self.spatial_variance} for t_var in self.temp_variance}

        proxyimage_ref = nib.load(join(tp_ref.data_dir[self.results_scope], tp_ref.id + '.im.nii.gz'))
        im = np.array(proxyimage_ref.dataobj)

        for it_tp_flo, tp_flo in enumerate(subject.sessions):
            proxyimage_flo = nib.load(join(tp_flo.data_dir[self.results_scope], tp_flo.id + '.im.nii.gz'))
            proxyseg_flo = nib.load(join(tp_flo.data_dir[self.results_scope], tp_flo.id + '.seg.nii.gz'))

            im_resampled = np.array(proxyimage_flo.dataobj)
            seg_resampled = np.array(proxyseg_flo.dataobj)
            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)

            mean_age_2 = (time_list[tp_flo.id] - time_list[tp_ref.id]) ** 2
            if im_resampled is not None:
                mean_im_2 = (im_resampled - im) ** 2
                if self.smooth is not None:
                    kern = np.ones((self.smooth, self.smooth, self.smooth)) / (
                            self.smooth * self.smooth * self.smooth)
                    mean_im_2 = convolve(mean_im_2, kern, mode='constant')
            else:
                mean_im_2 = None

            del im_resampled

            if 'sreg-synthmorph' in self.def_scope:
                svf_filename_flo = io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **svf_dict}) + '.nii.gz'
                proxysvf_flo = nib.load(join(tp_flo.data_dir[self.def_scope], svf_filename_flo))
                proxyflow = def_utils.integrate_svf(proxysvf_flo)
                jacobian = def_utils.compute_jacobian(np.array(proxyflow.dataobj))

                Msbj = np.load(join(subject.data_dir['sreg-synthmorph'], 'sub-' + subject.id + '_desc-atlas_aff.npy'))
                proxyjac = nib.Nifti1Image(jacobian, Msbj @ proxyflow.affine)
                proxyjac = def_utils.vol_resample(proxyimage_ref, proxyjac)
                jacobian = np.array(proxyjac.dataobj)

                seg_resampled *= jacobian[..., np.newaxis]

            for t_var in self.temp_variance:
                t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
                for s_var in self.spatial_variance:
                    s_ker = 1 if s_var == 'inf' else np.exp(-0.5 / s_var * mean_im_2)
                    p_data[t_var][s_var][..., it_tp_flo] = s_ker * t_ker
                    p_label[t_var][s_var] = np.add(p_label[t_var][s_var],
                                                   p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled,
                                                   out=p_label[t_var][s_var])

        for t_var in self.temp_variance:
            for s_var in self.spatial_variance:
                p_label[t_var][s_var] = np.divide(p_label[t_var][s_var],
                                                  np.sum(p_data[t_var][s_var], axis=-1, keepdims=True))
                p_label[t_var][s_var][np.isnan(p_label[t_var][s_var])] = 0

        return {self.floor_variance[0]: p_label}

    def label_fusion(self, subject, force_flag=False, resample=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'  # 'w' if force_flag else 'a'
        timepoints_to_run, time_list = self.prepare_data(subject, force_flag)
        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        self.resample_images(subject)

        for tp in timepoints_to_run:
            t_0 = time.time()
            print('     - Timepoint ' + tp.id, end=':', flush=True)

            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)
            p_label_dict = self.compute_p_label(subject, tp, time_list)

            # proxy = tp.get_image(**self.conditions_image)

            aff = subject.vox2ras0
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            st_vols = []
            for f_var in self.floor_variance:
                for t_var in self.temp_variance:
                    for s_var in self.spatial_variance:
                        fp_dict['suffix'] = 'dseg'
                        fp_dict['desc'] = 'f' + str(f_var) + 't' + str(t_var) + 's' + str(s_var) + self.interpmethod
                        filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                        p_label = p_label_dict[f_var][t_var][s_var]
                        p_label[np.isnan(p_label)] = 0

                        fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                        true_vol = np.zeros_like(fake_vol)
                        for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul

                        img = nib.Nifti1Image(true_vol, aff)
                        nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                        vols = get_vols_post(p_label, res=pixdim)
                        st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                        st_vols_dict['id'] = filename[:-7]
                        st_vols_dict['t_var'] = t_var if f_var == 1 else str(t_var) + 'f' + str(f_var)
                        st_vols_dict['s_var'] = s_var
                        st_vols_dict['interpmethod'] = self.interpmethod
                        st_vols_dict['type'] = 'posteriors'

                        st_vols += [st_vols_dict]

                        vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                        st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                        st_vols_dict['id'] = filename[:-7]
                        st_vols_dict['t_var'] = t_var if f_var == 1 else str(t_var) + 'f' + str(f_var)
                        st_vols_dict['s_var'] = s_var
                        st_vols_dict['interpmethod'] = self.interpmethod
                        st_vols_dict['type'] = 'seg'

                        st_vols += [st_vols_dict]
                        del p_label, fake_vol, true_vol

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del p_label_dict

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')

class LabelFusionNonLinTemplate(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None, smooth=None,
                 time_marker='time_to_bl_days', spatial_variance=None, floor_variance=None,
                 interpmethod='linear', save=False, process=False, type_map=None, normalise=True):

        self.seg_scope = seg_scop
        self.def_scope = def_scope
        self.results_scope = def_scope + '-temp' if seg_scop == 'synthseg' else def_scope + '-' + seg_scop + '-temp'
        self.results_lin_scope = 'sreg-lin' + '-temp' if seg_scop == 'synthseg' else 'sreg-lin-' + seg_scop + '-temp'
        self.temp_variance = temp_variance if temp_variance is not None else ['inf']
        self.spatial_variance = spatial_variance if spatial_variance is not None else ['inf']
        self.floor_variance = floor_variance if floor_variance is not None else [1]
        self.save = save
        self.process = process
        self.smooth = smooth
        self.time_marker = time_marker
        self.type_map = type_map
        self.normalise = normalise

        seg_suffix = 'dseg' if type_map is not None else 'post'
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': seg_suffix,
                               'scope': seg_scop, 'extension': 'nii.gz'}

        if seg_scop == 'freesurfer':
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_LUT)}
        else:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_ARR)}

        if type_map == 'distance_map':
            self.interpmethod = 'seg-' + interpmethod + '-post'
        elif type_map == 'onehot_map':
            self.interpmethod = 'onehot-' + interpmethod + '-post'
        else:
            self.interpmethod = 'post-' + interpmethod + '-post'

        self.interpmethod += '-nonlinjac'
        if not self.normalise: self.interpmethod += '-trueprob'

        self.channel_chunk = -1

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions
        if not (all([tv == 'inf' for tv in self.temp_variance]) or self.save):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        if len(timepoints) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return None, None

        if 'sreg' in self.def_scope and 'sreg-lin' not in self.def_scope:
            svf_dict = {'sub': subject.id, 'suffix': 'svf'}
            for tp in timepoints:
                if not exists(join(tp.data_dir[self.def_scope],
                                   io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz')):
                    print('Subject: ' + str(
                        subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                    return None, None

        if self.save and not force_flag:
            f = timepoints[-1].id + '_to_' + timepoints[-1].id + '_post.nii.gz'
            if (self.type_map is None and exists(join(subject.data_dir[self.def_scope], 'segmentations', f))) or \
                    (self.type_map == 'distance_map' and exists(
                        join(subject.data_dir[self.def_scope], 'segmentations_distance', f))) \
                    or (self.type_map == 'onehot_map' and exists(
                join(subject.data_dir[self.def_scope], 'segmentations_onehot', f))):
                print('Subject: ' + str(subject.id) + '. DONE')
                return None, None
            else:
                return timepoints, []
        else:
            vol_tsv = {}
            for tp in timepoints:
                last_dir = tp.data_dir[self.results_scope]
                if exists(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')):
                    vol_tsv[tp.id] = io_utils.read_tsv(
                        join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'),
                        k=['id', 't_var', 's_var', 'interpmethod'])

        #             vol_tsv[tp.id] = io_utils.read_tsv(
        #                 join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'),
        #                 k=['id', 't_var', 's_var', 'interpmethod', 'type'])
        #
        #             st_vols = []
        #             for k, v in vol_tsv[tp.id].items():
        #                 if None in v.keys(): v.pop(None)
        #                 if '-trueprob' in v['interpmethod']: continue
        #                 st_vols += [v]
        #
        #             fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [str(k) for k in ASEG_APARC_ARR]
        #             vols_dir = join(tp.data_dir[self.results_scope],
        #                             'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')
        #
        #             write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite='w')
        #
        # print('Subject: ' + str(subject.id) + '. DONE')
        # return None, None

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

                    elif any([(filename[:-7] + '_' + str(t) + 'f' + str(f) + '_' + str(
                            s) + '_' + self.interpmethod not in vol_tsv[tp.id].keys() and f != 1)
                              or (filename[:-7] + '_' + str(t) + '_' + str(s) + '_' + self.interpmethod not in vol_tsv[
                        tp.id].keys() and f == 1) for f in self.floor_variance for t in self.temp_variance for s in
                              self.spatial_variance]):
                        timepoints_to_run.append(tp)


        if not timepoints_to_run:
            print('Subject: ' + str(subject.id) + '. DONE')
            return None, None

        if all([tv == 'inf' for tv in self.temp_variance]) or self.save:
            time_list = {tp.id: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                time_list[tp.id] = float(tp.sess_metadata[self.time_marker])

        return timepoints_to_run, time_list

    def reg2temp(self, proxyref, proxyim_flo, proxyseg_flo, proxyflow):

        if self.type_map == 'distance_map':
            seg = np.array(proxyseg_flo.dataobj)
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32', copy=False)
            proxyseg_flo = nib.Nifti1Image(seg_flo, proxyseg_flo.affine)

        elif self.type_map == 'onehot_map':
            seg = np.array(proxyseg_flo.dataobj)
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.one_hot_encoding(seg, categories=list(self.labels_lut.keys())).astype('float32', copy=False)
            proxyseg_flo = nib.Nifti1Image(seg_flo, proxyseg_flo.affine)

        else:
            pass

        proxyseg_res = def_utils.vol_resample(proxyref, proxyseg_flo, proxyflow)
        proxyim_res = def_utils.vol_resample(proxyref, proxyim_flo, proxyflow)

        return proxyim_res, proxyseg_res

    def resample_images(self, subject):
        # Def parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        # Ref parameters
        proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)

        for tp in subject.sessions:
            if all([exists(join(tp.data_dir[self.results_scope], tp.id + '.seg.nii.gz')),
                    exists(join(tp.data_dir[self.results_scope], tp.id + '.jac.nii.gz')),
                    exists(join(tp.data_dir[self.results_scope], tp.id + '.im.nii.gz'))]):
                continue
            aff_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'
            svf_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'

            proxyimage = tp.get_image(**self.conditions_image)
            proxyseg = tp.get_image(**self.conditions_seg)

            affine_matrix = np.load(join(tp.data_dir['sreg-lin'], aff_filename))
            v2r_im = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)
            v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)
            proxyimage = nib.Nifti1Image(np.array(proxyimage.dataobj), v2r_im)
            proxyseg = nib.Nifti1Image(np.array(proxyseg.dataobj), v2r_seg)


            if 'lin' not in self.def_scope:
                proxysvf = nib.load(join(tp.data_dir[self.def_scope], svf_filename))
                proxyflow = def_utils.integrate_svf(proxysvf)
                jac_image = def_utils.compute_jacobian(np.array(proxyflow.dataobj))
                proxyjac_res = nib.Nifti1Image(jac_image, proxyflow.affine)
                nib.save(proxyjac_res, join(tp.data_dir[self.results_scope], tp.id + '.jac.nii.gz'))

            else:
                proxyflow = None

            proxyimage_res, proxyseg_res = self.reg2temp(proxytemplate, proxyimage, proxyseg, proxyflow)

            nib.save(proxyimage_res, join(tp.data_dir[self.results_scope], tp.id + '.im.nii.gz'))
            nib.save(proxyseg_res, join(tp.data_dir[self.results_scope], tp.id + '.seg.nii.gz'))

    def reg2image(self, subject, tp, p_label):
        # Def parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        # Ref parameters
        proxytemplate = nib.Nifti1Image(p_label, subject.vox2ras0)

        # if exists(join(tp.data_dir[self.results_scope], tp.id + '.seg.nii.gz')): return
        aff_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'
        svf_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'

        proxyimage = tp.get_image(**self.conditions_image)
        affine_matrix = np.load(join(tp.data_dir['sreg-lin'], aff_filename))
        v2r_im = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)
        proxyimage = nib.Nifti1Image(np.array(proxyimage.dataobj), v2r_im)

        if 'lin' not in self.def_scope:
            proxysvf = nib.load(join(tp.data_dir[self.def_scope], svf_filename))
            proxyflow = def_utils.integrate_svf(proxysvf, inverse=True)

        else:
            proxyflow = None

        proxyseg_res = def_utils.vol_resample(proxyimage, proxytemplate, proxyflow)
        return np.array(proxyseg_res.dataobj)

    def compute_p_label(self, subject, tp_ref, time_list):
        # Def parameters
        p_data = {
            t_var: {sp_var: np.zeros(subject.image_shape + (len(subject.sessions),), dtype='float32') for sp_var in
                    self.spatial_variance} for t_var in self.temp_variance}
        p_label = {
            t_var: {sp_var: np.zeros(subject.image_shape + (len(self.labels_lut),), dtype='float32') for sp_var in
                    self.spatial_variance} for t_var in self.temp_variance}

        proxyimage_ref = nib.load(join(tp_ref.data_dir[self.results_scope], tp_ref.id + '.im.nii.gz'))
        im = np.array(proxyimage_ref.dataobj)

        for it_tp_flo, tp_flo in enumerate(subject.sessions):
            proxyimage_flo = nib.load(join(tp_flo.data_dir[self.results_scope], tp_flo.id + '.im.nii.gz'))
            proxyseg_flo = nib.load(join(tp_flo.data_dir[self.results_scope], tp_flo.id + '.seg.nii.gz'))

            im_resampled = np.array(proxyimage_flo.dataobj)
            seg_resampled = np.array(proxyseg_flo.dataobj)
            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)

            mean_age_2 = (time_list[tp_flo.id] - time_list[tp_ref.id]) ** 2
            if im_resampled is not None:
                mean_im_2 = (im_resampled - im) ** 2
                ###
                # img = nib.Nifti1Image(mean_im_2, np.eye(4))
                # nib.save(img, 'mean_im_2.nii.gz')
                # mean_im_2_mask = (mean_im_2 < 10) & (np.sum(seg_resampled[..., 1:]) > 0.5)
                # mean_im_2_filt = mean_im_2[mean_im_2_mask]
                # plt.figure()
                # plt.hist(mean_im_2_filt, bins=100, density=True)
                # plt.ylim(0, 0.5)
                # plt.grid(True)
                # plt.savefig(tp_flo.id + '_mean_im_2_hist.png')
                # plt.close()
                ###
                if self.smooth is not None:
                    kern = np.ones((self.smooth, self.smooth, self.smooth)) / (self.smooth * self.smooth * self.smooth)
                    mean_im_2 = convolve(mean_im_2, kern, mode='constant')
                    ###
                    # img = nib.Nifti1Image(mean_im_2, np.eye(4))
                    # nib.save(img, 'mean_im_2_gauss.nii.gz')
                    # mean_im_2_filt = mean_im_2[mean_im_2_mask &  (mean_im_2 < 10)]
                    # plt.figure()
                    # plt.hist(mean_im_2_filt, bins=100, density=True)
                    # plt.ylim(0, 0.5)
                    # plt.xlim(0, 10)
                    # plt.grid()
                    # plt.savefig(tp_flo.id + '_mean_im_2_hist_gauss.png')
                    # plt.close()
                    ###

            else:
                mean_im_2 = None

            del im_resampled

            if 'sreg-synthmorph' in self.def_scope:
                proxyjac = nib.load(join(tp_ref.data_dir[self.results_scope], tp_ref.id + '.jac.nii.gz'))
                # jacobian = np.array(proxyjac_ref.dataobj)
                #
                # Msbj = np.load(join(subject.data_dir['sreg-synthmorph'], 'sub-' + subject.id + '_desc-atlas_aff.npy'))
                # proxyjac = nib.Nifti1Image(jacobian, Msbj @ proxyjac_ref.affine)
                proxyjac = def_utils.vol_resample(proxyimage_ref, proxyjac)
                jacobian = np.array(proxyjac.dataobj)

                seg_resampled *= jacobian[..., np.newaxis]

            for t_var in self.temp_variance:
                t_ker = 1/len(time_list) if t_var == 'inf' else 1/np.sqrt(2*np.pi*t_var)*np.exp(-0.5 / t_var * mean_age_2)
                if self.time_marker == 'time_to_bl_days': t_ker = t_ker*365
                for s_var in self.spatial_variance:
                    s_ker = 1/len(time_list) if s_var == 'inf' else 1/np.sqrt(2*np.pi*s_var)*np.exp(-0.5 / s_var * mean_im_2)
                    p_data[t_var][s_var][..., it_tp_flo] = s_ker * t_ker
                    p_label[t_var][s_var] = np.add(p_label[t_var][s_var],
                                                   p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled,
                                                   out=p_label[t_var][s_var])

        for t_var in self.temp_variance:
            for s_var in self.spatial_variance:
                if self.normalise:
                    p_label[t_var][s_var] = np.divide(p_label[t_var][s_var], np.sum(p_data[t_var][s_var], axis=-1, keepdims=True))
                p_label[t_var][s_var][np.isnan(p_label[t_var][s_var])] = 0

        return p_label

    def label_fusion(self, subject, force_flag=False, resample=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'  # 'w' if force_flag else 'a'
        timepoints_to_run, time_list = self.prepare_data(subject, force_flag)
        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        self.resample_images(subject)

        for tp in timepoints_to_run:
            t_0 = time.time()
            print('     - Timepoint ' + tp.id, end=':', flush=True)

            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)
            p_label_dict = self.compute_p_label(subject, tp, time_list)

            # proxy = tp.get_image(**self.conditions_image)
            aff = subject.vox2ras0
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            st_vols = []
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    fp_dict['suffix'] = 'dseg'
                    fp_dict['desc'] = 't' + str(t_var) + 's' + str(s_var) + self.interpmethod
                    filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                    p_label = p_label_dict[t_var][s_var]
                    p_label[np.isnan(p_label)] = 0

                    if self.normalise:
                        vols = get_vols_post(p_label)
                        st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                        st_vols_dict['id'] = filename[:-7]
                        st_vols_dict['t_var'] = t_var
                        st_vols_dict['s_var'] = s_var
                        st_vols_dict['interpmethod'] = self.interpmethod
                        st_vols_dict['type'] = 'posteriors'

                        st_vols += [st_vols_dict]

                    p_label = self.reg2image(subject, tp, p_label)
                    fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                    true_vol = np.zeros_like(fake_vol)
                    for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul

                    img = nib.Nifti1Image(true_vol, aff)
                    nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                    vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                    st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                    st_vols_dict['id'] = filename[:-7]
                    st_vols_dict['t_var'] = t_var
                    st_vols_dict['s_var'] = s_var
                    st_vols_dict['interpmethod'] = self.interpmethod
                    st_vols_dict['type'] = 'seg'

                    st_vols += [st_vols_dict]
                    del p_label, fake_vol, true_vol

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del p_label_dict

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')

class AdaptiveFusionVOX(object):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None,
                 time_marker='time_to_bl_days',  interpmethod='linear', type_map=None,
                 fusion_method='post', name='ADAPTVOX', **kwargs):

        self.name=name
        self.seg_scope = seg_scop
        self.def_scope = def_scope
        self.results_scope = def_scope if seg_scop == 'synthseg' else def_scope + '-' + seg_scop
        self.temp_variance = temp_variance if temp_variance is not None else ['inf']

        self.time_marker = time_marker
        self.type_map = type_map
        self.fusion_method = fusion_method

        seg_suffix = 'dseg' if type_map is not None else 'post'
        self.conditions_image = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': 'T1w', 'scope': 'synthseg'}
        self.conditions_seg = {'space': 'orig', 'acquisition': '1', 'run': '01', 'suffix': seg_suffix,
                               'scope': seg_scop, 'extension': 'nii.gz'}

        if seg_scop == 'freesurfer':
            self.labels_lut = {k: it_k for it_k, k in enumerate(ASEG_LUT)}
        else:
            self.labels_lut = {k: it_k for it_k, k in enumerate(POST_ARR)}

        if type_map == 'distance_map':
            self.interpmethod = 'seg-' + interpmethod + '-' + fusion_method
        elif type_map == 'onehot_map':
            self.interpmethod = 'onehot-' + interpmethod + '-' + fusion_method
        else:
            self.interpmethod = 'post-' + interpmethod + '-' + fusion_method

        self.channel_chunk = 20

    def prepare_data(self, subject, force_flag):
        print('\n  o Reading the input files')
        timepoints = subject.sessions
        if not all([tv == 'inf' for tv in self.temp_variance]):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        if len(timepoints) == 1:
            print('has only 1 timepoint. No segmentation is computed.')
            return None, None

        if 'sreg' in self.def_scope and 'sreg-lin' not in self.def_scope:
            svf_dict = {'sub': subject.id, 'suffix': 'svf'}
            for tp in timepoints:
                if not exists(join(tp.data_dir[self.def_scope],
                                   io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz')):
                    print('Subject: ' + str(
                        subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                    return None, None

        vol_tsv = {}
        for tp in timepoints:
            last_dir = tp.data_dir[self.results_scope]
            if exists(join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')):
                vol_tsv[tp.id] = io_utils.read_tsv(
                    join(last_dir, 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv'),
                    k=['id', 't_var', 's_var', 'interpmethod', 'type'])

        #         st_vols = []
        #         for k, v in vol_tsv[tp.id].items():
        #             if None in v.keys(): v.pop(None)
        #             if v['s_var'] == 'ADAPTROI': continue
        #             st_vols += [v]
        #
        #         fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [str(k) for k in ASEG_APARC_ARR]
        #         vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')
        #
        #         write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite='w')
        #
        #         os.remove(join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols1.tsv'))
        # print('Subject: ' + str(subject.id) + '. DONE')
        # return None, None

        timepoints_to_run = []
        if force_flag:
            timepoints_to_run = timepoints
        elif DEBUG:
            timepoints_to_run = timepoints[0:1]
        else:
            for tp in timepoints:
                filename = tp.get_files(**self.conditions_image)[0]
                if tp.id not in vol_tsv.keys():
                    timepoints_to_run.append(tp)
                elif any([(filename[:-7] + '_' + str(t) + '_' + self.name + '_' + self.interpmethod not in vol_tsv[tp.id].keys()) for t in self.temp_variance]):
                        timepoints_to_run.append(tp)

        if not timepoints_to_run:
            print('Subject: ' + str(subject.id) + '. DONE')
            return None, None

        if all([tv == 'inf' for tv in self.temp_variance]):
            time_list = {tp.id: 0 for tp in timepoints}
        else:
            time_list = {}
            for tp in timepoints:
                time_list[tp.id] = float(tp.sess_metadata[self.time_marker])

        return timepoints_to_run, time_list

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
        aff_dict_ref = {'sub': subject.id, 'desc': 'aff'}

        ## To remove
        if exists(join(tp_ref.data_dir['sreg-lin'],
                       io_utils.build_bids_fileame({**{'ses': tp_ref.id, 'run': '01'}, **aff_dict_ref}) + '.npy')):
            shutil.move(join(tp_ref.data_dir['sreg-lin'],
                             io_utils.build_bids_fileame({**{'ses': tp_ref.id, 'run': '01'}, **aff_dict_ref}) + '.npy'),
                        join(tp_ref.data_dir['sreg-lin'],
                             io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict_ref}) + '.npy'))
        ##
        affine_matrix_ref = np.load(join(tp_ref.data_dir['sreg-lin'],
                                         io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict_ref}) + '.npy'))
        svf_filename_ref = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'

        # Flo parameters
        aff_dict_flo = {'sub': subject.id, 'desc': 'aff'}

        ## To remove
        if exists(join(tp_flo.data_dir['sreg-lin'],
                       io_utils.build_bids_fileame({**{'ses': tp_flo.id, 'run': '01'}, **aff_dict_ref}) + '.npy')):
            shutil.move(join(tp_flo.data_dir['sreg-lin'],
                             io_utils.build_bids_fileame({**{'ses': tp_flo.id, 'run': '01'}, **aff_dict_ref}) + '.npy'),
                        join(tp_flo.data_dir['sreg-lin'],
                             io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_ref}) + '.npy'))
        ##
        affine_matrix_flo = np.load(join(tp_flo.data_dir['sreg-lin'],
                                         io_utils.build_bids_fileame({**{'ses': tp_flo.id}, **aff_dict_flo}) + '.npy'))
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

    def process_tp(self, subject, tp_ref, tp_flo, proxyim_ref):

        proxyimage_flo = tp_flo.get_image(**self.conditions_image)
        proxyseg_flo = tp_flo.get_image(**self.conditions_seg)
        v2r_flo = proxyimage_flo.affine.copy()
        if self.type_map == 'distance_map':
            seg = np.array(proxyseg_flo.dataobj)
            mask_flo = seg > 0
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut).astype('float32',
                                                                                                            copy=False)
            proxyseg_flo = nib.Nifti1Image(seg_flo, proxyseg_flo.affine)
            proxyseg_flo = def_utils.vol_resample(proxyimage_flo, proxyseg_flo, mode='nearest')
            seg_flo = np.array(proxyseg_flo.dataobj)

        elif self.type_map == 'onehot_map':
            seg = np.array(proxyseg_flo.dataobj)
            mask_flo = seg > 0
            seg[seg > 1999] = 42
            seg[seg > 999] = 3
            seg_flo = fn_utils.one_hot_encoding(seg, categories=self.labels_lut).astype('float32',copy=False)

            proxyseg_flo = nib.Nifti1Image(seg_flo, proxyseg_flo.affine)
            proxyseg_flo = def_utils.vol_resample(proxyimage_flo, proxyseg_flo, mode='nearest')
            seg_flo = np.array(proxyseg_flo.dataobj)

        else:
            proxyseg_flo = def_utils.vol_resample(proxyimage_flo, proxyseg_flo)
            seg_flo = np.array(proxyseg_flo.dataobj)
            mask_flo = np.sum(seg_flo[..., 1:], -1) > 0

        mask_flo, crop_coord = fn_utils.crop_label(mask_flo > 0, margin=10, threshold=0)
        image_flo = fn_utils.apply_crop(np.array(proxyimage_flo.dataobj), crop_coord)
        seg_flo = fn_utils.apply_crop(seg_flo, crop_coord)
        v2r_flo[:3, 3] = v2r_flo[:3] @ np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0], 1])

        proxyimage = nib.Nifti1Image(image_flo, v2r_flo)
        proxyseg = nib.Nifti1Image(seg_flo, v2r_flo)

        im_resampled, seg_resampled = self.register_timepoints_st(subject, tp_ref, tp_flo, proxyim_ref,
                                                                  proxyimage_flo=proxyimage, proxyseg_flo=proxyseg)

        return im_resampled, seg_resampled



    def compute_im_stats(self, subject):
        # Def parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        # Ref parameters
        proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)
        im_list = []
        seg_list = []
        for tp in subject.sessions:
            aff_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'
            svf_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'

            proxyimage = tp.get_image(**self.conditions_image)
            proxyseg = tp.get_image(**self.conditions_seg)

            affine_matrix = np.load(join(tp.data_dir['sreg-lin'], aff_filename))
            v2r_im = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)
            v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)
            proxyimage = nib.Nifti1Image(np.array(proxyimage.dataobj), v2r_im)
            proxyseg = nib.Nifti1Image(np.array(proxyseg.dataobj), v2r_seg)

            if 'lin' not in self.def_scope:
                proxysvf = nib.load(join(tp.data_dir[self.def_scope], svf_filename))
                proxyflow = def_utils.integrate_svf(proxysvf)
            else:
                proxyflow = None

            proxyim_res = def_utils.vol_resample(proxytemplate, proxyimage, proxyflow)
            im_list += [np.array(proxyim_res.dataobj)]
            proxyseg_res = def_utils.vol_resample(proxytemplate, proxyseg, proxyflow, mode='nearest')
            seg_list += [fn_utils.one_hot_encoding(np.array(proxyseg_res.dataobj), categories=self.labels_lut).astype('float32', copy=False)]

        seg_array = np.stack(seg_list, axis=0)
        del seg_list
        seg_voting = np.sum(seg_array, axis=0)
        seg_it_label = np.argmax(seg_voting, axis=-1)

        atlas_prior = np.zeros_like(seg_it_label)
        for l, it_l in self.labels_lut.items(): atlas_prior[seg_it_label == it_l] = l

        im_array = np.stack(im_list, axis=0)
        del im_list
        im_mean = np.mean(im_array, axis=0)
        im_var = np.mean((im_array - im_mean[np.newaxis])**2, axis=0)

        return im_mean, im_var, atlas_prior

    def resample2im(self, subject, tp_ref, im_mean, im_var, atlas_prior):

        # Def parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}
        proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)

        aff_filename = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict}) + '.npy'
        svf_filename = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'
        if 'lin' not in self.def_scope:
            proxysvf = nib.load(join(tp_ref.data_dir[self.def_scope], svf_filename))
            proxyflow = def_utils.integrate_svf(proxysvf, inverse=True)
        else:
            proxyflow = None

        # Reference
        proxyref = tp_ref.get_image(**self.conditions_image)
        affine_matrix = np.load(join(tp_ref.data_dir['sreg-lin'], aff_filename))
        v2r_im = np.matmul(np.linalg.inv(affine_matrix), proxyref.affine)
        proxyref = nib.Nifti1Image(np.array(proxyref.dataobj), v2r_im)

        # Atlas
        if self.type_map == 'distance_map':
            atlas_prior[atlas_prior > 1999] = 42
            atlas_prior[atlas_prior > 999] = 3
            atlas_prior = fn_utils.compute_distance_map(
                atlas_prior, soft_seg=False, labels_lut=np.unique(list(self.labels_lut.keys()))
            ).astype('float32', copy=False)

        elif self.type_map == 'onehot_map':
            # atlas_prior[atlas_prior > 1999] = 42
            # atlas_prior[atlas_prior > 999] = 3
            atlas_prior = fn_utils.one_hot_encoding(atlas_prior,
                                                    categories=self.labels_lut).astype('float32', copy=False)

        else:
            pass

        # Deform to reference
        proxyimage = nib.Nifti1Image(im_mean, proxytemplate.affine)
        proxyim_mean = def_utils.vol_resample(proxyref, proxyimage, proxyflow)

        proxyimage = nib.Nifti1Image(im_var, proxytemplate.affine)
        proxyim_var = def_utils.vol_resample(proxyref, proxyimage, proxyflow)

        proxyimage = nib.Nifti1Image(atlas_prior, proxytemplate.affine)
        proxyatlas = def_utils.vol_resample(proxyref, proxyimage, proxyflow)

        return np.array(proxyim_mean.dataobj), np.array(proxyim_var.dataobj), np.array(proxyatlas.dataobj)

    def compute_p_label(self, tp, subject, time_list, im_mean=None, im_var=None, atlas_prior=None):

        timepoints = subject.sessions
        if not (all([tv == 'inf' for tv in self.temp_variance])):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        proxyimage = tp.get_image(**self.conditions_image)
        im_mean, im_var, atlas_prior = self.resample2im(subject, tp, im_mean=im_mean, im_var=im_var, atlas_prior=atlas_prior)
        if self.fusion_method == 'seg':
            num_classes = atlas_prior.shape[-1]
            atlas_prior = np.argmax(atlas_prior, axis=-1)
            atlas_prior = fn_utils.one_hot_encoding(atlas_prior, num_classes=num_classes).astype('float32',  copy=False)

        p_data = {t_var: np.zeros(proxyimage.shape + (len(timepoints),), dtype='float32') for t_var in self.temp_variance}
        p_label = {t_var: atlas_prior.copy() for t_var in self.temp_variance}

        del atlas_prior

        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp.id == tp_flo.id:
                proxyseg_ref = tp_flo.get_image(**self.conditions_seg)
                im_resampled = np.array(proxyimage.dataobj)
                mean_age_2 = 0
                mean_im_2 = (im_resampled - im_mean) ** 2

                # pdb.set_trace()
                # _, crop_coord = fn_utils.crop_label(np.array(proxyseg_ref)>0, margin=10)
                if self.type_map == 'distance_map':
                    seg = np.array(proxyseg_ref.dataobj)
                    seg[seg > 1999] = 42
                    seg[seg > 999] = 3
                    seg_resampled = fn_utils.compute_distance_map(seg, soft_seg=False,
                                                                  labels_lut=self.labels_lut).astype('float32',
                                                                                                     copy=False)
                    proxyseg = nib.Nifti1Image(seg_resampled, proxyseg_ref.affine)
                    proxyseg = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
                    seg_resampled = np.array(proxyseg.dataobj)

                elif self.type_map == 'onehot_map':
                    seg = np.array(proxyseg_ref.dataobj)
                    # seg[seg > 1999] = 42
                    # seg[seg > 999] = 3
                    seg_resampled = fn_utils.one_hot_encoding(seg, categories=self.labels_lut).astype(
                        'float32', copy=False)
                    proxyseg = nib.Nifti1Image(seg_resampled, proxyseg_ref.affine)
                    proxyseg = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
                    seg_resampled = np.array(proxyseg.dataobj)

                else:
                    if np.sum(proxyseg_ref.affine - proxyimage.affine) > 1e-4:
                        proxyseg = def_utils.vol_resample(proxyimage, proxyseg_ref, mode='linear')
                        seg_resampled = np.array(proxyseg.dataobj)
                    else:
                        seg_resampled = np.array(proxyseg_ref.dataobj)

            else:
                im_resampled, seg_resampled = self.process_tp(subject, tp, tp_flo, proxyimage)

                mean_age_2 = (time_list[tp_flo.id] - time_list[tp.id]) ** 2
                mean_im_2 = (im_resampled - im_mean) ** 2


            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            if self.fusion_method == 'seg':
                num_classes = seg_resampled.shape[-1]
                seg_resampled_max = np.argmax(seg_resampled, axis=-1)
                seg_resampled = fn_utils.one_hot_encoding(seg_resampled_max, num_classes=num_classes).astype('float32', copy=False)

            s_ker = np.exp(-0.5 / im_var * mean_im_2)
            s_ker[np.isnan(s_ker)] = 1

            del im_resampled

            for t_var in self.temp_variance:
                t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
                p_data[t_var][..., it_tp_flo] = t_ker * s_ker
                p_label[t_var] = np.add(p_label[t_var], p_data[t_var][..., it_tp_flo, np.newaxis] * seg_resampled, out=p_label[t_var])

        for t_var in self.temp_variance:
            p_label[t_var] = np.divide(p_label[t_var], np.sum(p_data[t_var], axis=-1, keepdims=True))
            p_label[t_var][np.isnan(p_label[t_var])] = 0
        return p_label

    def label_fusion(self, subject, force_flag=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'  # 'w' if force_flag else 'a'
        timepoints_to_run, time_list = self.prepare_data(subject, force_flag)
        if timepoints_to_run is None:
            return

        im_mean, im_var, atlas_prior = self.compute_im_stats(subject)

        print('  o Computing the segmentation')
        for tp in timepoints_to_run:
            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)

            t_0 = time.time()
            print('        - Timepoint ' + tp.id, end=':', flush=True)
            p_label_dict = self.compute_p_label(tp, subject, time_list, im_mean=im_mean.copy(), im_var=im_var.copy(), atlas_prior=atlas_prior.copy())
            proxy = tp.get_image(**self.conditions_image)

            aff = proxy.affine
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            st_vols = []
            for t_var in self.temp_variance:
                fp_dict['suffix'] = 'dseg'
                fp_dict['desc'] = 'f1' + 't' + str(t_var) + 'sADAPTVOX' + self.interpmethod
                filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                p_label = p_label_dict[t_var]

                fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                true_vol = np.zeros_like(fake_vol)
                for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul

                img = nib.Nifti1Image(true_vol, aff)
                nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                vols = get_vols_post(p_label, res=pixdim)
                st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                st_vols_dict['id'] = filename[:-7]
                st_vols_dict['t_var'] = t_var
                st_vols_dict['s_var'] = 'ADAPTVOX'
                st_vols_dict['interpmethod'] = self.interpmethod
                st_vols_dict['type'] = 'posteriors'

                st_vols += [st_vols_dict]

                vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                st_vols_dict['id'] = filename[:-7]
                st_vols_dict['t_var'] = t_var
                st_vols_dict['s_var'] = 'ADAPTVOX'
                st_vols_dict['interpmethod'] = self.interpmethod
                st_vols_dict['type'] = 'seg'

                st_vols += [st_vols_dict]
                del p_label, fake_vol, true_vol

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del p_label_dict

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')

class AdaptiveFusionROI(AdaptiveFusionVOX):

    def compute_atlas_prior(self, subject):
        # Def parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}

        # Ref parameters
        proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)
        seg_list = []
        for tp in subject.sessions:
            aff_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **aff_dict}) + '.npy'
            svf_filename = io_utils.build_bids_fileame({**{'ses': tp.id}, **svf_dict}) + '.nii.gz'

            proxyseg = tp.get_image(**self.conditions_seg)

            affine_matrix = np.load(join(tp.data_dir['sreg-lin'], aff_filename))
            v2r_seg = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)
            proxyseg = nib.Nifti1Image(np.array(proxyseg.dataobj), v2r_seg)

            if 'lin' not in self.def_scope:
                proxysvf = nib.load(join(tp.data_dir[self.def_scope], svf_filename))
                proxyflow = def_utils.integrate_svf(proxysvf)
            else:
                proxyflow = None

            proxyseg_res = def_utils.vol_resample(proxytemplate, proxyseg, proxyflow, mode='nearest')
            seg_list += [fn_utils.one_hot_encoding(np.array(proxyseg_res.dataobj), categories=self.labels_lut).astype('float32',copy=False)]

        seg_array = np.stack(seg_list, axis=0)
        del seg_list
        seg_voting = np.sum(seg_array, axis=0)
        seg_it_label = np.argmax(seg_voting, axis=-1)

        atlas_prior = np.zeros_like(seg_it_label)
        for l, it_l in self.labels_lut.items(): atlas_prior[seg_it_label == it_l] = l

        return atlas_prior

    def resample2im(self, subject, tp_ref, **kwargs):
        proxyref_im = tp_ref.get_image(**self.conditions_image)
        proxyref_seg = tp_ref.get_image(**self.conditions_seg)
        segref = np.array(proxyref_seg.dataobj)

        stats_dict = {seg_label: [] for seg_label in np.unique(segref)}
        for tp in subject.sessions:
            proxyflo_im = tp.get_image(**self.conditions_image)
            proxyflo_seg = tp.get_image(**self.conditions_seg)
            im = np.array(proxyflo_im.dataobj)
            seg = np.array(proxyflo_seg.dataobj)
            for seg_label in np.unique(seg):
                stats_dict[seg_label] += [im[seg==seg_label]]

        im_mean = np.zeros(proxyref_im.shape)
        im_var = np.zeros(proxyref_im.shape)
        for seg_label, seg_vals in stats_dict.items():
            seg_vals_con = np.concatenate(seg_vals)
            im_mean[segref == seg_label] = np.mean(seg_vals_con)
            im_var[segref == seg_label] = np.mean((seg_vals_con - np.mean(seg_vals_con))**2)

        return im_mean, im_var

    def compute_p_label(self, tp, subject, time_list, **kwargs):

        timepoints = subject.sessions
        if not (all([tv == 'inf' for tv in self.temp_variance])):
            timepoints = list(filter(lambda t: not np.isnan(float(t.sess_metadata[self.time_marker])), timepoints))

        proxyimage = tp.get_image(**self.conditions_image)
        im_mean, im_var = self.resample2im(subject, tp)

        p_data = {t_var: np.zeros(proxyimage.shape + (len(timepoints),), dtype='float32') for t_var in self.temp_variance}
        p_label = {t_var: np.zeros(proxyimage.shape + (len(self.labels_lut),)) for t_var in self.temp_variance}

        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp.id == tp_flo.id:
                proxyseg_ref = tp_flo.get_image(**self.conditions_seg)
                im_resampled = np.array(proxyimage.dataobj)
                mean_age_2 = 0
                mean_im_2 = (im_resampled - im_mean) ** 2

                if self.type_map == 'distance_map':
                    seg = np.array(proxyseg_ref.dataobj)
                    seg[seg > 1999] = 42
                    seg[seg > 999] = 3
                    seg_resampled = fn_utils.compute_distance_map(seg, soft_seg=False,
                                                                  labels_lut=self.labels_lut).astype('float32',
                                                                                                     copy=False)
                    proxyseg = nib.Nifti1Image(seg_resampled, proxyseg_ref.affine)
                    proxyseg = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
                    seg_resampled = np.array(proxyseg.dataobj)

                elif self.type_map == 'onehot_map':
                    seg = np.array(proxyseg_ref.dataobj)
                    # seg[seg > 1999] = 42
                    # seg[seg > 999] = 3
                    seg_resampled = fn_utils.one_hot_encoding(seg, categories=self.labels_lut).astype(
                        'float32', copy=False)
                    proxyseg = nib.Nifti1Image(seg_resampled, proxyseg_ref.affine)
                    proxyseg = def_utils.vol_resample(proxyimage, proxyseg, mode='nearest')
                    seg_resampled = np.array(proxyseg.dataobj)

                else:
                    if np.sum(proxyseg_ref.affine - proxyimage.affine) > 1e-4:
                        proxyseg = def_utils.vol_resample(proxyimage, proxyseg_ref, mode='linear')
                        seg_resampled = np.array(proxyseg.dataobj)
                    else:
                        seg_resampled = np.array(proxyseg_ref.dataobj)

            else:
                im_resampled, seg_resampled = self.process_tp(subject, tp, tp_flo, proxyimage)

                mean_age_2 = (time_list[tp_flo.id] - time_list[tp.id]) ** 2
                mean_im_2 = (im_resampled - im_mean) ** 2

            if self.type_map == 'distance_map': seg_resampled = softmax(seg_resampled, axis=-1)
            if self.fusion_method == 'seg':
                num_classes = seg_resampled.shape[-1]
                seg_resampled_max = np.argmax(seg_resampled, axis=-1)
                seg_resampled = fn_utils.one_hot_encoding(seg_resampled_max, num_classes=num_classes).astype('float32',
                                                                                                             copy=False)

            s_ker = np.exp(-0.5 / im_var * mean_im_2)
            s_ker[np.isnan(s_ker)] = 1

            del im_resampled

            for t_var in self.temp_variance:
                t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
                p_data[t_var][..., it_tp_flo] = t_ker * s_ker
                p_label[t_var] = np.add(p_label[t_var], p_data[t_var][..., it_tp_flo, np.newaxis] * seg_resampled,
                                        out=p_label[t_var])

        for t_var in self.temp_variance:
            p_label[t_var] = np.divide(p_label[t_var], np.sum(p_data[t_var], axis=-1, keepdims=True))
            p_label[t_var][np.isnan(p_label[t_var])] = 0
        return p_label

    def label_fusion(self, subject, force_flag=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'  # 'w' if force_flag else 'a'
        timepoints_to_run, time_list = self.prepare_data(subject, force_flag)
        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        for tp in timepoints_to_run:
            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)

            t_0 = time.time()
            print('        - Timepoint ' + tp.id, end=':', flush=True)
            p_label_dict = self.compute_p_label(tp, subject, time_list)
            proxy = tp.get_image(**self.conditions_image)

            aff = proxy.affine
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            st_vols = []
            for t_var in self.temp_variance:
                fp_dict['suffix'] = 'dseg'
                fp_dict['desc'] = 'f1' + 't' + str(t_var) + 'sADAPTROI' + self.interpmethod
                filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                p_label = p_label_dict[t_var]
                p_label[np.isnan(p_label)] = 0

                fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                true_vol = np.zeros_like(fake_vol)
                for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul

                img = nib.Nifti1Image(true_vol, aff)
                nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                vols = get_vols_post(p_label, res=pixdim)
                st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                st_vols_dict['id'] = filename[:-7]
                st_vols_dict['t_var'] = t_var
                st_vols_dict['s_var'] = 'ADAPTROI'
                st_vols_dict['interpmethod'] = self.interpmethod
                st_vols_dict['type'] = 'posteriors'

                st_vols += [st_vols_dict]

                vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                st_vols_dict['id'] = filename[:-7]
                st_vols_dict['t_var'] = t_var
                st_vols_dict['s_var'] = 'ADAPTROI'
                st_vols_dict['interpmethod'] = self.interpmethod
                st_vols_dict['type'] = 'seg'

                st_vols += [st_vols_dict]
                del p_label, fake_vol, true_vol

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del p_label_dict

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')

class AdaptiveFusionCONSTANT(AdaptiveFusionVOX):

    def __init__(self, def_scope='sreg-synthmorph', seg_scop='synthseg', temp_variance=None,
                 time_marker='time_to_bl_days', interpmethod='linear', type_map=None,
                 fusion_method='post', sp_var='inf', name='ADAPTCONST'):

        super().__init__(def_scope=def_scope,
                         seg_scop=seg_scop,
                         temp_variance=temp_variance,
                         time_marker=time_marker,
                         interpmethod=interpmethod,
                         type_map=type_map,
                         fusion_method=fusion_method,
                         name=name)
        self.sp_var = sp_var

    def resample2im(self, subject, tp_ref, atlas_prior, im_mean, **kwargs):

        # Def parameters
        aff_dict = {'sub': subject.id, 'desc': 'aff'}
        svf_dict = {'sub': subject.id, 'suffix': 'svf'}
        proxytemplate = nib.Nifti1Image(np.zeros(subject.image_shape), subject.vox2ras0)

        aff_filename = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **aff_dict}) + '.npy'
        svf_filename = io_utils.build_bids_fileame({**{'ses': tp_ref.id}, **svf_dict}) + '.nii.gz'
        if 'lin' not in self.def_scope:
            proxysvf = nib.load(join(tp_ref.data_dir[self.def_scope], svf_filename))
            proxyflow = def_utils.integrate_svf(proxysvf, inverse=True)
        else:
            proxyflow = None

        # Reference
        proxyref = tp_ref.get_image(**self.conditions_image)
        affine_matrix = np.load(join(tp_ref.data_dir['sreg-lin'], aff_filename))
        v2r_im = np.matmul(np.linalg.inv(affine_matrix), proxyref.affine)
        proxyref = nib.Nifti1Image(np.array(proxyref.dataobj), v2r_im)

        # Atlas
        if self.type_map == 'distance_map':
            atlas_prior[atlas_prior > 1999] = 42
            atlas_prior[atlas_prior > 999] = 3
            atlas_prior = fn_utils.compute_distance_map(
                atlas_prior, soft_seg=False, labels_lut=np.unique(list(self.labels_lut.keys()))
            ).astype('float32', copy=False)

        elif self.type_map == 'onehot_map':
            # atlas_prior[atlas_prior > 1999] = 42
            # atlas_prior[atlas_prior > 999] = 3
            atlas_prior = fn_utils.one_hot_encoding(atlas_prior,
                                                    categories=self.labels_lut).astype('float32', copy=False)

        else:
            pass

        # Deform to reference
        proxyimage = nib.Nifti1Image(im_mean, proxytemplate.affine)
        proxyim_mean = def_utils.vol_resample(proxyref, proxyimage, proxyflow)

        proxyimage = nib.Nifti1Image(atlas_prior, proxytemplate.affine)
        proxyatlas = def_utils.vol_resample(proxyref, proxyimage, proxyflow)

        return np.array(proxyim_mean.dataobj), self.sp_var, np.array(proxyatlas.dataobj)

    def label_fusion(self, subject, force_flag=False, *args, **kwargs):

        print('Subject: ' + str(subject.id), end=' ', flush=True)

        attach_overwrite = 'a'  # 'w' if force_flag else 'a'
        timepoints_to_run, time_list = self.prepare_data(subject, force_flag)
        if timepoints_to_run is None:
            return

        im_mean, im_var, atlas_prior = self.compute_im_stats(subject)

        print('  o Computing the segmentation')
        for tp in timepoints_to_run:
            filename = tp.get_files(**self.conditions_image)[0]
            fp_dict = io_utils.get_bids_fileparts(filename)

            t_0 = time.time()
            print('        - Timepoint ' + tp.id, end=':', flush=True)
            p_label_dict = self.compute_p_label(tp, subject, time_list, im_mean=im_mean.copy(), atlas_prior=atlas_prior.copy())
            proxy = tp.get_image(**self.conditions_image)

            aff = proxy.affine
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            st_vols = []
            for t_var in self.temp_variance:
                fp_dict['suffix'] = 'dseg'
                fp_dict['desc'] = 'f1' + 't' + str(t_var) + 'sADAPTCONST' + self.interpmethod
                filename_seg = io_utils.build_bids_fileame(fp_dict) + '.nii.gz'

                p_label = p_label_dict[t_var]

                fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                true_vol = np.zeros_like(fake_vol)
                for it_ul, ul in enumerate(self.labels_lut.keys()): true_vol[fake_vol == it_ul] = ul

                img = nib.Nifti1Image(true_vol, aff)
                nib.save(img, join(tp.data_dir[self.results_scope], filename_seg))

                vols = get_vols_post(p_label, res=pixdim)
                st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                st_vols_dict['id'] = filename[:-7]
                st_vols_dict['t_var'] = t_var
                st_vols_dict['s_var'] = 'ADAPTCONST'
                st_vols_dict['interpmethod'] = self.interpmethod
                st_vols_dict['type'] = 'posteriors'

                st_vols += [st_vols_dict]

                vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                st_vols_dict = {k: vols[v] for k, v in self.labels_lut.items()}
                st_vols_dict['id'] = filename[:-7]
                st_vols_dict['t_var'] = t_var
                st_vols_dict['s_var'] = 'ADAPTCONST'
                st_vols_dict['interpmethod'] = self.interpmethod
                st_vols_dict['type'] = 'seg'

                st_vols += [st_vols_dict]
                del p_label, fake_vol, true_vol

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in ASEG_APARC_ARR]
            vols_dir = join(tp.data_dir[self.results_scope], 'sub-' + subject.id + '_ses-' + tp.id + '_vols.tsv')

            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite=attach_overwrite)

            del p_label_dict

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject.id) + '. DONE')


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


