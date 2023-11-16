import numpy as np
import os


repo_home = os.environ.get('PYTHONPATH')
ctx_labels = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_labels.npy'))
ctx_names = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_names.npy'))
APARC_DICT = {k: v for k, v in zip(ctx_labels, ctx_names) if v.lower() != 'background'}
APARC_DICT_REV = {v: k for k, v in zip(ctx_labels, ctx_names) if v.lower() != 'background'}

subcortical_labels = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_labels.npy'))
subcortical_labels = np.concatenate((subcortical_labels, [24]))
subcortical_names = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_names.npy'))
subcortical_names = np.concatenate((subcortical_names, ['csf']))
SYNTHSEG_DICT = {k: v for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}
SYNTHSEG_DICT_REV = {v: k for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}


aseg_names = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'aseg_segmentation_names.npy'))
aseg_labels = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'aseg_segmentation_labels.npy'))
ASEG_DICT = {k: v for k, v in zip(aseg_labels, aseg_names) if v.lower() != 'background'}
ASEG_DICT_REV = {v: k for k, v in zip(aseg_labels, aseg_names) if v.lower() != 'background'}

SYNTHSEG_LUT = {k: it_k for it_k, k in enumerate(np.unique(subcortical_labels))}
SYNTHSEG_LUT = {**SYNTHSEG_LUT, **{k: SYNTHSEG_LUT[3] if k < 2000 else SYNTHSEG_LUT[42] for k in ctx_labels if k != 0}}
ASEG_LUT = {k: it_k for it_k, k in enumerate(np.unique(aseg_labels))}
ASEG_LUT = {**ASEG_LUT, **{k: ASEG_LUT[3] if k < 2000 else ASEG_LUT[42] for k in ctx_labels if k != 0}}
SYNTHSEG_APARC_LUT = {k: it_k for it_k, k in enumerate(np.unique(np.concatenate((subcortical_labels, ctx_labels), axis=0)))}
ASEG_APARC_LUT = {k: it_k for it_k, k in enumerate(np.unique(np.concatenate((aseg_labels, ctx_labels), axis=0)))}


CLUSTER_DICT = {
    'Gray': [53, 17, 51, 12, 54, 18, 50, 11, 58, 26, 42, 3],
    'CSF': [4, 5, 43, 44, 15, 14, 24],
    'Thalaumus': [49, 10],
    'Pallidum': [52, 13],
    'VentralDC': [28, 60],
    'Brainstem': [16],
    'WM': [41, 2],
    'cllGM': [47, 8],
    'cllWM': [46, 7]
}

CSF_LABELS = [24] # CLUSTER_DICT['CSF']
