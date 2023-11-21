import csv
import pdb
from os import listdir
from os.path import exists, join, basename, dirname
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import softmax

from setup import *
from utils.labels import *
from utils import fn_utils,  def_utils


class LabelFusion(object):

    def __init__(self, bids_loader, args, chunk_size=None):
        self.bids_loader = bids_loader
        self.seg_scope = args.seg_scope
        self.def_scope = args.field
        self.output_scope = args.field + '-' + args.seg_scope
        self.output_dir = join(DERIVATIVES_DIR, self.output_scope)
        self.tvar = args.tvar if args.tvar is not None else [np.inf]
        self.svar = args.svar if args.svar is not None else [np.inf]
        self.time_marker = args.tm
        self.type_map = args.type_map
        self.save_seg = args.save_seg

        self.im_1mm_ent = {'acquisition': '1', 'extension': 'nii.gz', 'suffix': 'T1w', 'scope': self.seg_scope}
        self.im_ent = {'acquisition': None, 'extension': 'nii.gz', 'suffix': 'T1w', 'scope': self.seg_scope}
        self.seg_ent = {'acquisition': '1', 'suffix': 'T1wdseg', 'scope': self.seg_scope, 'extension': 'nii.gz'}


        self.all_labels_flag = args.all_labels
        if self.seg_scope == 'freesurfer' and self.all_labels_flag is False:
            self.labels_lut = ASEG_LUT
        elif self.seg_scope == 'synthseg' and self.all_labels_flag is False:
            self.labels_lut = SYNTHSEG_LUT
        elif self.seg_scope == 'synthseg' and self.all_labels_flag:
            self.labels_lut = SYNTHSEG_APARC_LUT
        else:
            self.labels_lut = SYNTHSEG_LUT

        if self.all_labels_flag:
            self.vols_fname = '_desc-all_vols.tsv'
        else:
            self.vols_fname = '_vols.tsv'

        if self.type_map == 'distance_map':
            self.interpmethod = 'seg'
        elif self.type_map == 'onehot_map':
            self.interpmethod = 'onehot'
        elif self.type_map == 'gauss_map':
            self.interpmethod = 'gauss'
        else:
            self.interpmethod = 'post'

        self.chunk_size = (192, 192, 192) if chunk_size is None else chunk_size
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

        print('\n  o Reading the input files.')
        timepoints = self.bids_loader.get_session(subject=subject)
        sess_tsv = self.bids_loader.get(suffix='sessions', extension='tsv', subject=subject)

        sess_df = sess_tsv[0].get_df()
        sess_df = sess_df.set_index('session_id')
        sess_df = sess_df[~sess_df.index.duplicated(keep='last')]

        # if time marker is not available but could be inferred
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

        if not (all([tv == np.inf for tv in self.tvar])):
            timepoints = list(filter(lambda t: not np.isnan(float(sess_df.loc[t][self.time_marker])), timepoints))

        timepoints = list(filter(lambda t: len(self.bids_loader.get(**{'subject': subject, 'session': t, **self.seg_ent})) > 0, timepoints))
        timepoints = list(filter(lambda t: len(self.bids_loader.get(**{**{'subject': subject, 'session': t}, **self.im_ent})) > 0, timepoints))

        if len(timepoints) == 1:
            print('[done] Only 1 timepoint available. Skipping.')
            return None

        svf_ent = {'suffix': 'svf', 'scope': 'uslr-nonlin', 'extension': 'nii.gz'}
        aff_ent = { 'suffix': 'T1w', 'desc': 'aff', 'scope': 'uslr-lin',  'extension': 'npy'}
        if self.def_scope == 'uslr-nonlin':
            for tp in timepoints:
                tp_ent = {'subject': subject, 'session': tp}
                svf_files = self.bids_loader.get(**tp_ent, **svf_ent)
                if len(svf_files) != 1:
                    print('[error] N=' + str(len(svf_files)) + ' SVFs files available; refine the search. Skipping.')
                    return subject

        elif self.def_scope == 'uslr-lin':
            for tp in timepoints:
                tp_ent = {'subject': subject, 'session': tp}
                aff_files = self.bids_loader.get(**tp_ent, **aff_ent)
                if len(aff_files) != 1:
                    print('[error] N=' + str(len(aff_files)) + ' affine files available; refine the search. Skipping.')
                    return subject
        vol_tsv = {}
        for tp in timepoints:
            tp_ent = {'subject': subject, 'session': tp}
            seg_tsv = self.bids_loader.get(suffix='vols', scope=self.output_scope, extension='tsv', **tp_ent)
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

        else:
            for tp in timepoints:
                if tp not in vol_tsv.keys():
                    timepoints_to_run.append(tp)
                else:
                    if not self.svar and not self.tvar:
                        timepoints_to_run.append(tp)
                    elif all([len(vol_tsv[tp].loc[t, s, self.interpmethod]) == 1 for t in self.tvar for s in self.svar]):
                        timepoints_to_run.append(tp)

        if not timepoints_to_run:
            print('[done] Subject: ' + str(subject) + ' has been already processed.')
            return None

        if all([tv == np.inf for tv in self.tvar]):
            time_dict = {tp: 0 for tp in timepoints}
        else:
            time_dict = {}
            for tp in timepoints:
                time_dict[tp] = float(sess_df.loc[tp][self.time_marker])


        return (timepoints_to_run, timepoints, time_dict)

    def register_timepoints_st(self, subject, tp_ref, tp_flo, proxyref, proxyimage_flo=None, proxyseg_flo=None,
                               im_mode='linear', seg_mode='linear'):
        '''
        :param subject:
        :param tp_ref:
        :param tp_flo:
        :param image_flo:
        :param seg_flo:
        :param im_mode: 'linear', 'nearest'.
        :param seg_mode: 'linear', 'nearest'.
        :return:
        '''

        # Ref parameters
        aff_dict = {'subject': subject, 'desc': 'aff', 'suffix': 'T1w', 'scope': 'uslr-lin', 'extension': 'npy'}
        svf_dict = {'subject': subject, 'suffix': 'svf', 'scope': 'uslr-nonlin', 'extension': 'nii.gz', 'space': None}

        affine_file_ref = self.bids_loader.get(**{**aff_dict, 'session': tp_ref})
        if len(affine_file_ref) != 1:
            print('[error] not valid affine files. Skipping.')
            return subject

        affine_matrix_ref = np.load(affine_file_ref[0].path)
        svf_filename_ref = self.bids_loader.get(**{**svf_dict, 'session': tp_ref})

        # Flo parameters
        affine_file_flo = self.bids_loader.get(**{**aff_dict, 'session': tp_flo})
        if len(affine_file_flo) != 1:
            print('[error] not valid affine files. Skipping.')
            return subject

        affine_matrix_flo = np.load(affine_file_flo[0].path)
        svf_filename_flo = self.bids_loader.get(**{**svf_dict, 'session': tp_flo})

        proxyflow = None
        if self.def_scope != 'uslr-lin':
            if len(svf_filename_flo) != 1 or len(svf_filename_ref) != 1:
                print('[error] not valid SVF files. Skipping.')
                return subject
            tp_svf = nib.load(svf_filename_ref[0].path)
            tp_flo_svf = nib.load(svf_filename_flo[0].path)

            svf = np.asarray(tp_flo_svf.dataobj) - np.asarray(tp_svf.dataobj)
            proxysvf = nib.Nifti1Image(svf.astype('float32'), tp_svf.affine)
            proxyflow = def_utils.integrate_svf(proxysvf)

        # Deform
        v2r_ref = np.matmul(np.linalg.inv(affine_matrix_ref), proxyref.affine)
        proxyref_align = nib.Nifti1Image(np.zeros(proxyref.shape), v2r_ref)
        im_resampled = None
        seg_resampled = None

        if proxyimage_flo is not None:
            v2r_target = np.matmul(np.linalg.inv(affine_matrix_flo), proxyimage_flo.affine)
            proxyflo_im = nib.Nifti1Image(np.array(proxyimage_flo.dataobj), v2r_target)
            im_mri = def_utils.vol_resample_fast(proxyref_align, proxyflo_im, proxyflow=proxyflow, mode=im_mode)

            im_resampled = np.array(im_mri.dataobj)
            del im_mri

        if proxyseg_flo is not None:
            v2r_target = np.matmul(np.linalg.inv(affine_matrix_flo), proxyseg_flo.affine)

            if len(proxyseg_flo.shape) > 3:
                seg_list = []
                for it_c in range(0, proxyseg_flo.shape[-1], self.channel_chunk):
                    pflo_seg = nib.Nifti1Image(np.array(
                        proxyseg_flo.dataobj[..., it_c:it_c+self.channel_chunk]), v2r_target)
                    seg_mri = def_utils.vol_resample_fast(proxyref_align, pflo_seg, proxyflow=proxyflow, mode=seg_mode)

                    # In case the last iteration has only 1 channel (it squeezes due to batch dimension)
                    if len(seg_mri.shape) == 3:
                        seg_list += [np.array(seg_mri.dataobj)[..., np.newaxis]]
                    else:
                        seg_list += [np.array(seg_mri.dataobj)]

                    del seg_mri

                seg_resampled = np.concatenate(seg_list, axis=-1)
            else:
                proxyflo_seg = nib.Nifti1Image(np.array(proxyseg_flo.dataobj), v2r_target)
                seg_resampled = np.array(def_utils.vol_resample(proxyref_align, proxyflo_seg,
                                                                proxyflow=proxyflow, mode=seg_mode).dataobj)

        return im_resampled, seg_resampled

    def compute_seg_map(self, proxyimage, proxyseg):

        v2r_ref = proxyseg.affine.copy()
        proxyimage_flo = def_utils.vol_resample(proxyseg, proxyimage)
        if self.type_map == 'distance_map':
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            seg = fn_utils.compute_distance_map(seg, soft_seg=False, labels_lut=self.labels_lut)
            seg = seg.astype('float32', copy=False)

        elif self.type_map == 'onehot_map':
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            seg = fn_utils.one_hot_encoding(seg, categories=self.labels_lut).astype('float32', copy=False)

        elif self.type_map == 'gauss_map':
            seg = np.array(proxyseg.dataobj)
            mask = seg > 0
            seg = fn_utils.one_hot_encoding(seg, categories=self.labels_lut).astype('float32', copy=False)
            for it_label in range(seg.shape[-1]):
                seg[..., it_label] = gaussian_filter(seg[..., it_label], sigma=2)

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

    def process_tp(self, subject, tp_ref, tp_flo, proxyim_ref, im_d, seg_d):

        im_file = im_d[tp_ref]
        seg_file = seg_d[tp_ref]

        _, proxyseg = self.compute_seg_map(nib.load(im_file.path), nib.load(seg_file.path))

        proxyimage = nib.load(im_file.path)
        image = fn_utils.gaussian_smoothing_voxel_size(proxyimage, [1, 1, 1])
        proxyimage = nib.Nifti1Image(image, proxyimage.affine)

        if all([s == np.inf for s in self.svar]):
            proxyimage = None

        seg_mode = 'nearest'
        if self.type_map == 'distance_map':
            seg_mode = 'distance'
        elif self.type_map in ['onehot_map', 'gauss_map']:
            seg_mode = 'linear'

        data_resampled = self.register_timepoints_st(subject, tp_ref, tp_flo, proxyim_ref, proxyimage_flo=proxyimage,
                                                     proxyseg_flo=proxyseg, seg_mode=seg_mode)

        return data_resampled

    def compute_p_label(self, timepoints, subject, tp, proxyim_ref, proxyseg_ref, chunk, im_d, seg_d):

        v2r_ref = proxyim_ref.affine
        chunk_size = (chunk[0][1]-chunk[0][0], chunk[1][1]-chunk[1][0], chunk[2][1]-chunk[2][0])
        p_data = np.zeros(chunk_size + (1, len(timepoints)), dtype='float32')
        p_label = np.zeros(chunk_size + (len(np.unique(list(self.labels_lut.values()))), len(timepoints)), dtype='float32')

        im = np.array(proxyim_ref.dataobj[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]])
        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp == tp_flo:
                im_res = im
                seg_res = np.array(proxyseg_ref.dataobj[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1],
                                                        chunk[2][0]: chunk[2][1]])

            else:
                data_res = self.process_tp(subject, tp, tp_flo, proxyim_ref, im_d, seg_d)
                if isinstance(data_res, str):
                    return data_res
                im_res = data_res[0][chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]]
                seg_res = data_res[1][chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]]

            if self.type_map == 'distance_map':
                seg_res = softmax(seg_res, axis=-1)


            p_data[..., it_tp_flo] = im_res[..., np.newaxis]
            p_label[..., it_tp_flo] = seg_res
            del im_res, seg_res

        return (im[..., np.newaxis, np.newaxis] - p_data)**2, p_label, v2r_ref

    def label_fusion(self, subject, force_flag=False, *args, **kwargs):

        data_init = self.prepare_data(subject, force_flag)
        if data_init is None:
            return
        elif isinstance(data_init, str):
            return data_init
        else:
            timepoints_to_run, timepoints, time_list = data_init

        print('  o Computing the segmentation.')
        im_d = {}
        seg_d = {}
        for tp in timepoints_to_run:
            im_file = self.bids_loader.get(subject=subject, session=tp, **self.im_ent)
            seg_file = self.bids_loader.get(subject=subject, session=tp, **self.seg_ent)

            if len(seg_file) > 1:
                txtf = list(filter(lambda f: 'txt' in f and 'dseg' in f, listdir(seg_file[0].path)))
                if len(txtf) != 1:
                    print('[error] seg file for subject ' + subject + ' and session ' + tp + ' is not found. Skipping')
                    return subject

                seg_file = [f.filename[:-7] == txtf[0][:-4] for f in seg_file]
                if len(seg_file) != 1:
                    print('[error] seg file for subject ' + subject + ' and session ' + tp + ' is not found. Skipping')
                    return subject

            if len(im_file) > 1:
                txtf = list(filter(lambda f: 'txt' in f and 'dseg' not in f, listdir(seg_file[0].path)))
                if len(txtf) != 1:
                    print('[error] More than one image file found. Skipping.')
                    return subject

                im_file = [f.filename[:-7] == txtf[0][:-4] for f in im_file]
                if len(im_file) != 1:
                    print('[error] More than one image file found. Skipping.')
                    return subject

            im_d[tp] = im_file[0]
            seg_d[tp] = seg_file[0]

        for tp in timepoints_to_run:
            t_0 = time.time()
            print('        - Timepoint ' + tp, end=': ', flush=True)
            output_dir_tp = join(self.output_dir, 'sub-' + subject, 'ses-' + tp, 'anat')
            if not exists(output_dir_tp): os.makedirs(output_dir_tp)

            proxyimage, proxyseg = nib.load(im_d[tp].path), nib.load(seg_d[tp].path)
            image = fn_utils.gaussian_smoothing_voxel_size(proxyimage, [1, 1, 1])
            proxyimage = nib.Nifti1Image(image, proxyimage.affine)

            proxyimage, proxyseg = self.compute_seg_map(proxyimage, proxyseg)

            seg_out_ent = {
                'subject': subject,
                'session': tp,
                'extension': 'nii.gz',
                'suffix': 'T1wdseg'
            }

            st_vols = []
            st_vols_dict = {t_var: {s_var: {k: 0 for k in self.labels_lut.keys()} for s_var in self.svar} for t_var in self.tvar}
            st_vols_dict_norm = {t_var: {s_var: {k: 0 for k in self.labels_lut.keys()} for s_var in self.svar} for t_var in self.tvar}

            chunk_list = self.get_chunk_list(proxyimage)
            for it_chunk, chunk in enumerate(chunk_list):
                end_chunk = ', '
                if it_chunk == len(chunk_list) - 1:
                    end_chunk = '.'

                print(str(it_chunk + 1) + '/' + str(len(chunk_list)), end=end_chunk, flush=True)


                output_label = self.compute_p_label(timepoints, subject, tp, proxyimage, proxyseg, chunk, im_d, seg_d)
                if isinstance(output_label, str):
                    return output_label

                time_arr = np.array([v for v in time_list.values()])
                mean_age_2 = (time_arr - time_list[tp]) ** 2

                pixdim = np.sqrt(np.sum(output_label[2] * output_label[2], axis=0))[:-1]
                for t_var in self.tvar:
                    t_ker = 1 if t_var in [np.inf, 0] else np.exp(-0.5 / t_var * mean_age_2)
                    for s_var in self.svar:
                        s_ker = 1 if s_var in [np.inf, 0] else np.exp(-0.5 / s_var * output_label[0])
                        p_data = s_ker * t_ker
                        del s_ker

                        p_label = np.zeros(output_label[1].shape[:-1])
                        if float(s_var) > 0 and float(t_var) > 0:
                            for it_t in range(output_label[1].shape[-1]):
                                if len(p_data.shape) == 1:
                                    p_label += p_data * output_label[1][..., it_t]  # , axis=-1)
                                else:
                                    p_label += p_data[...,  it_t] * output_label[1][..., it_t]#, axis=-1)
                        else:
                            it_ref = [it_t for it_t, t in enumerate(timepoints) if t == tp][0]
                            p_label = output_label[1][..., it_ref]

                        del p_data

                        p_label = p_label / np.sum(p_label, axis=-1, keepdims=True)
                        p_label[np.isnan(p_label)] = 0

                        fake_vol = np.argmax(p_label, axis=-1).astype('int16')
                        if self.save_seg:
                            seg_filepath = self.bids_loader.build_path(
                                {**seg_out_ent, 'desc': 't' + str(t_var) + 's' + str(s_var) + self.interpmethod},
                                scope=self.output_scope, path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                absolute_paths=False)

                            true_vol = np.zeros_like(fake_vol)
                            for it_ul, ul in enumerate(self.labels_lut.keys()):
                                true_vol[fake_vol == it_ul] = ul

                            if it_chunk == 0:
                                true_final_vol = np.zeros(proxyimage.shape)
                            else:
                                proxytmp = nib.load(join(self.output_dir, seg_filepath))
                                true_final_vol = np.array(proxytmp.dataobj)

                            true_final_vol[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1],
                                           chunk[2][0]: chunk[2][1]] = true_vol

                            img = nib.Nifti1Image(true_final_vol, output_label[2])
                            nib.save(img, join(self.output_dir, seg_filepath))

                            del true_final_vol, true_vol

                        vols_norm = get_vols_post(p_label, res=pixdim)
                        vols = get_vols(fake_vol, res=pixdim, labels=list(self.labels_lut.values()))
                        for k, v in self.labels_lut.items():
                            st_vols_dict_norm[t_var][s_var][k] = st_vols_dict_norm[t_var][s_var][k] + vols_norm[v]
                            st_vols_dict[t_var][s_var][k] = st_vols_dict[t_var][s_var][k] + vols[v]

                        del p_label, fake_vol

                del output_label

            for t_var in self.tvar:
                for s_var in self.svar:

                    st_d = st_vols_dict_norm[t_var][s_var]
                    st_d['id'] = im_d[tp].filename[:-7]
                    st_d['t_var'] = t_var
                    st_d['s_var'] = s_var
                    st_d['interpmethod'] = self.interpmethod
                    st_d['type'] = 'posteriors'
                    st_vols += [st_d]

                    st_d = st_vols_dict[t_var][s_var]
                    st_d['id'] = im_d[tp].filename[:-7]
                    st_d['t_var'] = t_var
                    st_d['s_var'] = s_var
                    st_d['interpmethod'] = self.interpmethod
                    st_d['type'] = 'seg'
                    st_vols += [st_d]

            fieldnames = ['id', 't_var', 's_var', 'interpmethod', 'type'] + [k for k in list(self.labels_lut.keys())]
            vols_dir = join(output_dir_tp, 'sub-' + subject + '_ses-' + tp + self.vols_fname)
            write_volume_results(st_vols, vols_dir, fieldnames=fieldnames, attach_overwrite='a')

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')

            # -------- #

        print('Subject: ' + str(subject) + '. DONE')

def write_volume_results(volume_dict, filepath, fieldnames=None, attach_overwrite='a'):
    if fieldnames is None:
        fieldnames = ['id', 't_var', 's_var'] + list(SYNTHSEG_LUT.keys())

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


