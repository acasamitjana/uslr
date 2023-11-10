import os
import pdb
from os.path import join, basename, exists
import shutil
import copy
import json

from bids import BIDSLayout
import nibabel as nib

from setup import *
from utils.io_utils import read_tsv

class BIDSLoader(object):

    def __init__(self, data_dir=BIDS_DIR, sid_list=None, **kwargs):
        self.sid_list = sid_list
        self.data_dir = data_dir
        self.derivatives = {}

        self._initialize_dataset(data_dir, **kwargs)

    def _initialize_dataset(self, data_dir, **kwargs):

        self.layout = BIDSLayout(data_dir)
        participants_file = self.layout.get_file(filename= join(data_dir, 'participants.tsv'))
        if participants_file is None: raise ValueError('[BIDSLeader - __init__]: no participants.tsv file is found.')
        participants_file = participants_file.get_df()
        participants_file.set_index('participant_id', inplace=True)
        self.participants_file = participants_file.T

    def add_derivatives(self, derivative_dir):
        self.layout.add_derivatives(path=derivative_dir)

    @property
    def subject_id_list(self):
        return self._get_subject_ids

    @property
    def subject_list(self):
        slist = []
        for sid in self._get_subject_ids:
            slist.append(self.layout.get(subject=sid))
        return slist

    def session_ids(self, sbj_id):
        return self._get_session_ids(sbj_id)

    def _get_subject_ids(self):
        if self.sid_list is None:
            return self.layout.get(return_type='id', target='subject')
        else:
            return self.sid_list

    def _get_session_ids(self, subject_id):
        return self.layout.get(subject=subject_id, return_type='id', target='session')

    def _get_subject_session(self, subject_id):
        return self.layout.get(subject=subject_id, target='session')

    def _get_subject(self, subject_id):
        return self.layout.get(subject=subject_id)

    def _get_subject_file(self, subject_id):
        return self.layout.get(subject=subject_id, extension='tsv')

    def _get_session_images(self, subject_id):
        return self.layout.get(subject=subject_id, extension='.nii.gz', suffix='T1w')

    def _get_session(self, subject_id, session_id):
        return self.layout.get(subject=subject_id, session=session_id)

    def _get_session_image(self, subject_id, session_id):
        return self.layout.get(subject=subject_id, session=session_id, extension='.nii.gz', suffix='T1w')

    def _get_image(self, subject, session, **kwargs):
        files = self.layout.get(subject=subject, session=session, extension='nii.gz', **kwargs)
        if files is None or len(files) != 1:
            print('[BIDSLoader] Please, refine your search filters. Subject: ' + subject + ' session: ' + session +
                  '. Hint: check entities and scopes available.')
            return
        return nib.load(files[0].path)

    def _get_file(self, subject, session, **kwargs):
        files = self.layout.get(subject=subject, session=session, **kwargs)

        if files is None or len(files) != 1:
            # print('[BIDSLoader] Please, refine your search filters. Subject: ' + subject + ' session: ' + session +
            #       '. Hint: check entities and scopes available.')
            return
        return files[0]


class T1wLoader(BIDSLoader):
    def _initialize_dataset(self, data_dir, **kwargs):
        self.participants_file = read_tsv(join(data_dir, 'participants.tsv'))
        self.subject_dict = {}
        subject_ids = self._get_subject_ids()
        for sbj_id in subject_ids:
            sbj_dir = join(self.data_dir, 'sub-' + sbj_id)
            self.subject_dict[sbj_id] = Subject(sbj_id, sbj_dir, sbj_metadata=self.participants_file[sbj_id])

    def add_derivatives(self, derivative_dir):
        # super(T1wLoader, self).add_derivatives(derivative_dir=derivative_dir)
        for sbj_id, sbj in self.subject_dict.items():
            if not exists(join(derivative_dir, 'sub-' + sbj_id)): os.makedirs(join(derivative_dir, 'sub-' + sbj_id))
            sbj.add_derivatives(derivative_name=basename(derivative_dir), derivative_dir=join(derivative_dir, 'sub-' + sbj_id))

    def _get_subject_ids(self):
        if self.sid_list is None:
            sid_list = list(filter(lambda x: 'sub-' in x and os.path.isdir(join(self.data_dir, x)), os.listdir(self.data_dir)))
            self.sid_list = [s.split('sub-')[1] for s in sid_list if s.split('sub-')[1] in self.participants_file.keys()]
        return self.sid_list

    @property
    def subject_id_list(self):
        return list(self.subject_dict.keys())

    @property
    def subject_list(self):
        return list(self.subject_dict.values())

    @property
    def image_shape(self):
        ishape = [0,0,0]
        for sbj in self.subject_list:
            image_shape = sbj.image_shape
            for it in range(3):
                if image_shape[it] > ishape[it]:
                    ishape[it] = image_shape[it]

        return tuple(ishape)


class Subject(object):
    def __init__(self, sbj_id, sbj_dir, sbj_metadata, **kwargs):
        self.sbj_id = sbj_id
        self.data_dir = {'bids': sbj_dir}
        self.sbj_dict = sbj_metadata
        self._initialize_subject(**kwargs)

        files = [sd for sd in os.listdir(sbj_dir) if not os.path.isdir(join(sbj_dir, sd))]
        self.files = {'bids': files}
        self.json_files = {'bids': [f for f in files if 'json' in f]}
        self.image_files = {'bids': [f for f in files if 'nii.gz' in f]}

    def _initialize_subject(self, **kwargs):

        # session_df = self.sessions_file.get_df()
        # session_df.set_index('session_id', inplace=True)
        # session_df = session_df.T
        #
        # t2 = time.time()
        sess_tsv = join(self.data_dir['bids'], 'sub-' + self.sbj_id + '_sessions.tsv')
        if os.path.exists(sess_tsv):
            self.sessions_file = read_tsv(sess_tsv)

        # else:
        #     shutil.rmtree(self.data_dir['bids'])
        #     self.sessions_file = None
        #     return

        self.session_dict = {}
        for sess_id in self.sessions_file.keys():
            sess_dir = join(self.data_dir['bids'], 'ses-' + sess_id, 'anat')
            # if not os.path.exists(sess_dir):
            #     continue
            # elif os.path.exists(join(self.data_dir['bids'], 'ses-' + sess_id, 'ses-' + sess_id)):
            #     import shutil
            #     shutil.rmtree(join(self.data_dir['bids'], 'ses-' + sess_id, 'ses-' + sess_id))
            sess_metadata = self.sessions_file[sess_id] if self.sessions_file is not None else None
            self.session_dict[sess_id] = Session(sess_id, sess_dir, sess_metadata)

    def _get_session_ids(self):
        return list(self.session_dict.keys())

    def add_derivatives(self, derivative_name, derivative_dir):
        self.data_dir[derivative_name] = derivative_dir
        if self.sessions_file is None: return

        for sess_id, sess in self.session_dict.items():
            sess_derivative_dir = join(derivative_dir,  'ses-' + sess_id, 'anat')
            if not exists(sess_derivative_dir): os.makedirs(sess_derivative_dir)
            sess.add_derivatives(derivative_name=derivative_name, derivative_dir=sess_derivative_dir)



        if os.path.exists(derivative_dir):
            files = [sd for sd in os.listdir(derivative_dir) if not os.path.isdir(join(derivative_dir, sd))]
        else:
            files = []

        # for f in files:
        #     if os.path.islink(join(derivative_dir, f)) and '/home/acasamitjana' in os.readlink(join(derivative_dir, f)):
        #         print(os.readlink(join(derivative_dir, f)))
        #         syml = os.readlink(join(derivative_dir, f))
        #         syml = syml.replace('/home/acasamitjana', '/mnt/HDD')
        #         os.remove(join(derivative_dir, f))
        #         os.symlink(syml, join(derivative_dir, f))

        self.files[derivative_name] = files
        self.json_files[derivative_name]  = [f for f in files if 'json' in f]
        self.image_files [derivative_name] = [f for f in files if 'nii.gz' in f]


    def _filter_files(self, suffix=None, extension=None, space=None, acquisition=None, run=None, desc=None, scope=None):
        '''
        :param suffix: (str). Type of image: 'T1w', 'mask', 'dseg']
        :param space: (None, str). Space of the image: ['None, 'orig', 'SUBJECT']. The SUBJECT space is the common
        space for all subject timepoints
        :param res: (None, str). Resolution of the image [None (original res), '1' (resampled)]
        :param desc: (None, str). Description:
                    - desc=None
                    - desc='resampled': resampled by using the vox2ras
                    - desc='*KERNEL_size*':  for longitudinal segmentations
                    - desc='cog': sessions's center of gravity
                    - desc='affine': sessions`s affine matrix to subject space
                    - desc='svf': session's SVFs to subject space.
        :param run: (None, str). Run, by default not specified, so all files will be returned; but it could be any num.
        :param scope: (None, str). Scope, by default rawdata but could be derivative names
        :return:
        '''
        # conditions = {}
        # if suffix is not None:  conditions['suffix'] = modality
        # if extension is not None: conditions['extension'] = extension
        # if space is not None: conditions['space'] = space
        # if acq is not None: conditions['acquisition'] = acq
        # if desc is not None: conditions['desc'] = desc
        # if scope is not None: conditions['scope'] = scope
        # self.layout.get(subject=self.sbj_id, session=self.sess_id, **conditions)


        files = self.files['bids'] if scope is None else self.files[scope]

        conditions = []
        if space is not None: conditions.append('space-' + space)
        if acquisition is not None: conditions.append('acq-' + acquisition)
        if desc is not None: conditions.append('desc-' + desc)
        if run is not None: conditions.append('run-' + run)

        keep_cond = []
        for c in conditions:
            if any([c.split('-')[0] in f for f in files]):
                keep_cond.append(c)

        if suffix is not None: keep_cond.append(suffix)
        if extension is not None: keep_cond.append(extension)

        out_files = []
        for f in files:
            # keep_cond = [c for c in conditions if c.split('-')[0] in f]
            # pdb.set_trace()
            # if keep_cond:
            if all([c in f for c in keep_cond]):
                out_files.append(f)

        return out_files

        # warnings.warn('[BIDSLoader-Session] File not found. Conditions: ' + ','.join(conditions))

    def get_files(self, suffix=None, extension=None, space=None, acquisition=None, run=None, desc=None, scope=None):
        '''
        :param suffix: (str). Type of image: 'T1w', 'mask', 'dseg']
        :param space: (None, str). Space of the image: ['None, 'orig', 'SUBJECT']. The SUBJECT space is the common
        space for all subject timepoints
        :param acquisition: (None, str). Resolution of the image [None (original res), '1' (resampled)]
        :param desc: (None, str). Description:
                    - desc=None
                    - desc='resampled': resampled by using the vox2ras
                    - desc='*KERNEL_size*':  for longitudinal segmentations
                    - desc='cog': sessions's center of gravity
                    - desc='affine': sessions`s affine matrix to subject space
                    - desc='svf': session's SVFs to subject space.
        :param scope: (None, str). Scope, by default rawdata but could be derivative names

        :return:
        '''
        f = self._filter_files(suffix=suffix, extension=extension, space=space, acquisition=acquisition,  run=run, desc=desc, scope=scope)
        return f

    @property
    def id(self):
        return self.sbj_id

    @property
    def sessions_id(self):
        return list(self.session_dict.keys())

    @property
    def sessions(self):
        return list(self.session_dict.values())

    @property
    def image_shape(self):
        try:
            return nib.load(join(self.data_dir['sreg-lin'], 'sub-' + self.id + '_desc-linTemplate_T1w.nii.gz')).shape
        except:
            print('[BIDS LOADER] -- WARNING -- No image_shape for the subject ' + str(self.id))
            return (256, 256, 256)

    @property
    def vox2ras0(self):
        try:
            return nib.load(join(self.data_dir['sreg-lin'], 'sub-' + self.id + '_desc-linTemplate_T1w.nii.gz')).affine
        except:
            print('[BIDS LOADER] -- WARNING -- No vox2ras0 for the subject ' + str(self.id))
            import numpy as np
            return np.eye(4)

    def items(self):
        for k, v in self.session_dict.items():
            yield k, v


class Session(object):
    def __init__(self, sess_id, sess_dir, sess_metadata):
        self.sess_id = sess_id
        self.sess_metadata = sess_metadata # None
        self.data_dir = {'bids': sess_dir}

        files = []
        if exists(sess_dir):
            files = os.listdir(sess_dir)
        # for f in files:
        #     if os.path.islink(join(sess_dir, f)):
        #         print(os.readlink(join(sess_dir, f)))
        #         syml = os.readlink(join(sess_dir, f))
        #         syml = syml.replace('/home/acasamitjana', '/mnt/HDD')
        #         os.remove(join(sess_dir, f))
        #         os.symlink(syml, join(sess_dir, f))
        self.files = {'bids': files}
        self.json_files = {'bids': [f for f in files if 'json' in f]}
        self.image_files = {'bids': [f for f in files if 'nii.gz' in f]}

    def add_derivatives(self, derivative_name, derivative_dir):
        self.data_dir[derivative_name] = derivative_dir

        if os.path.exists(derivative_dir):
            files = os.listdir(derivative_dir)
        else:
            files = []

        for f in files:
            if 'acq_' in f:
                shutil.move(join(derivative_dir, f), join(derivative_dir, f.replace('acq_', 'acq-')))

        self.files[derivative_name] = files
        self.json_files[derivative_name]  = [f for f in files if 'json' in f]
        self.image_files [derivative_name] = [f for f in files if 'nii.gz' in f]

    @property
    def id(self):
        return self.sess_id

    @property
    def num_runs(self):
        num_runs = 1
        run_list = [fp.split('-')[1] for f in self.files['bids'] for fp in f.split('_') if 'run' in fp]
        if run_list:
            num_runs = max([int(rl) for rl in run_list])

        return num_runs

    def _filter_files(self, suffix=None, extension=None, space=None, acquisition=None, run=None, desc=None, scope=None):
        '''
        :param suffix: (str). Type of image: 'T1w', 'mask', 'dseg']
        :param space: (None, str). Space of the image: ['None, 'orig', 'SUBJECT']. The SUBJECT space is the common
        space for all subject timepoints
        :param res: (None, str). Resolution of the image [None (original res), '1' (resampled)]
        :param desc: (None, str). Description:
                    - desc=None
                    - desc='resampled': resampled by using the vox2ras
                    - desc='*KERNEL_size*':  for longitudinal segmentations
                    - desc='cog': sessions's center of gravity
                    - desc='affine': sessions`s affine matrix to subject space
                    - desc='svf': session's SVFs to subject space.
        :param run: (None, str). Run, by default not specified, so all files will be returned; but it could be any num.
        :param scope: (None, str). Scope, by default rawdata but could be derivative names
        :return:
        '''

        # conditions = {}
        # if suffix is not None:  conditions['suffix'] = modality
        # if extension is not None: conditions['extension'] = extension
        # if space is not None: conditions['space'] = space
        # if acq is not None: conditions['acquisition'] = acq
        # if desc is not None: conditions['desc'] = desc
        # if scope is not None: conditions['scope'] = scope
        # self.layout.get(subject=self.sbj_id, session=self.sess_id, **conditions)

        files = self.files['bids'] if scope is None else self.files[scope]
        if suffix is not None: files = list(filter(lambda x: suffix in x, files))

        conditions = []
        if space is not None: conditions.append('space-' + space)
        if acquisition is not None: conditions.append('acq-' + acquisition)
        if desc is not None: conditions.append('desc-' + desc)
        if run is not None: conditions.append('run-' + run)


        keep_cond = []
        for c in conditions:
            if any([c.split('-')[0] in f for f in files]):
                keep_cond.append(c)

        if suffix is not None: keep_cond.append(suffix)
        if extension is not None: keep_cond.append(extension)

        out_files = []
        for f in files:
            # keep_cond = [c for c in conditions if c.split('-')[0] in f]
            # pdb.set_trace()
            # if keep_cond:
            if all([c in f for c in keep_cond]):
                out_files.append(f)

        return out_files

        # warnings.warn('[BIDSLoader-Session] File not found. Conditions: ' + ','.join(conditions))

    def get_files(self, suffix=None, extension=None, space=None, acquisition=None, run=None, desc=None, scope=None):
        '''
        :param suffix: (str). Type of image: 'T1w', 'mask', 'dseg']
        :param space: (None, str). Space of the image: ['None, 'orig', 'SUBJECT']. The SUBJECT space is the common
        space for all subject timepoints
        :param acquisition: (None, str). Resolution of the image [None (original res), '1' (resampled)]
        :param desc: (None, str). Description:
                    - desc=None
                    - desc='resampled': resampled by using the vox2ras
                    - desc='*KERNEL_size*':  for longitudinal segmentations
                    - desc='cog': sessions's center of gravity
                    - desc='affine': sessions`s affine matrix to subject space
                    - desc='svf': session's SVFs to subject space.
        :param scope: (None, str). Scope, by default rawdata but could be derivative names

        :return:
        '''
        f = self._filter_files(suffix=suffix, extension=extension, space=space, acquisition=acquisition,  run=run, desc=desc, scope=scope)
        return f

    def get_image(self, extension='nii.gz', suffix='T1w', space=None, acquisition=None, run=None, desc=None, scope=None):
        '''
        :param suffix: (str). Type of image: 'T1w', 'mask', 'dseg']
        :param space: (None, str). Space of the image: ['None, 'orig', 'SUBJECT']. The SUBJECT space is the common
        space for all subject timepoints
        :param acquisition: (None, str). Resolution of the image [None (original res), '1' (resampled)]
        :param desc: (None, str). Description:
                    - desc=None
                    - desc='resampled': resampled by using the vox2ras
                    - desc='*KERNEL_size*':  for longitudinal segmentations
                    - desc='cog': sessions's center of gravity
                    - desc='affine': sessions`s affine matrix to subject space
                    - desc='svf': session's SVFs to subject space.
        :param scope: (None, str). Scope, by default rawdata but could be derivative names

        :return:
        '''
        f = self._filter_files(suffix, extension=extension, space=space, acquisition=acquisition, run=run, desc=desc, scope=scope)
        if not f:
            return None

        # if len(f)>1:
        #     print('[BIDS_LOADER] ++ WARNING ++ More than two files are found with this query.')

        return nib.load(join(self.data_dir[scope], f[0]))

    def get_image_metadata(self, suffix='T1w', space=None, acquisition=None, run=None, desc=None, scope=None):
        '''
        :param suffix: (str). Type of image: 'T1w', 'mask', 'dseg']
        :param space: (None, str). Space of the image: ['None, 'orig', 'SUBJECT']. The SUBJECT space is the common
        space for all subject timepoints
        :param acquisition: (None, str). Resolution of the image [None (original res), '1' (resampled)]
        :param desc: (None, str). Description:
                    - desc=None
                    - desc='resampled': resampled by using the vox2ras
                    - desc='*KERNEL_size*':  for longitudinal segmentations
                    - desc='cog': sessions's center of gravity
                    - desc='affine': sessions`s affine matrix to subject space
                    - desc='svf': session's SVFs to subject space.
        :param scope: (None, str). Scope, by default rawdata but could be derivative names
        :return:
        '''
        p = self._filter_files(suffix, extension='json', space=space, acquisition=acquisition, desc=desc,
                               run=run, scope=scope)
        f = open(p[0], 'r')
        return json.load(f)

    def get_metadata(self):
        # if self.sess_metadata is None:
        #     print('Please, initialise subject metadata (subject._initialize_metadata()))')
        return self.sess_metadata
