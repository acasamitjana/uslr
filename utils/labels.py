import numpy as np
import os
import csv

from setup import SYNTHSEG_DIR


repo_home = os.environ.get('PYTHONPATH')
ctx_labels = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_labels.npy'))
ctx_names = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_names.npy'))
APARC_DICT = {k: v for k, v in zip(ctx_labels, ctx_names) if v.lower() != 'background'}
APARC_DICT_REV = {v: k for k, v in zip(ctx_labels, ctx_names) if v.lower() != 'background'}

subcortical_labels = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_labels.npy'))
subcortical_names = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_names.npy'))
SYNTHSEG_DICT = {k: v for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}
SYNTHSEG_DICT_REV = {v: k for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}

# ctx_lh_names = [
#     'ctx-lh-bankssts',
#     'ctx-lh-caudalanteriorcingulate',
#     'ctx-lh-caudalmiddlefrontal',
#     'ctx-lh-cuneus',
#     'ctx-lh-entorhinal',
#     'ctx-lh-fusiform',
#     'ctx-lh-inferiorparietal',
#     'ctx-lh-inferiortemporal',
#     'ctx-lh-isthmuscingulate',
#     'ctx-lh-lateraloccipital',
#     'ctx-lh-lateralorbitofrontal',
#     'ctx-lh-lingual',
#     'ctx-lh-medialorbitofrontal',
#     'ctx-lh-middletemporal',
#     'ctx-lh-parahippocampal',
#     'ctx-lh-paracentral',
#     'ctx-lh-parsopercularis',
#     'ctx-lh-parsorbitalis',
#     'ctx-lh-parstriangularis',
#     'ctx-lh-pericalcarine',
#     'ctx-lh-postcentral',
#     'ctx-lh-posteriorcingulate',
#     'ctx-lh-precentral',
#     'ctx-lh-precuneus',
#     'ctx-lh-rostralanteriorcingulate',
#     'ctx-lh-rostralmiddlefrontal',
#     'ctx-lh-superiorfrontal',
#     'ctx-lh-superiorparietal',
#     'ctx-lh-superiortemporal',
#     'ctx-lh-supramarginal',
#     'ctx-lh-frontalpole',
#     'ctx-lh-temporalpole',
#     'ctx-lh-transversetemporal',
#     'ctx-lh-insula'
# ]
# ctx_rh_names = [n.replace('lh', 'rh') for n in ctx_lh_names]
#
# APARC_DICT = {
#     **{k: v for k, v in zip(np.concatenate((np.arange(1,4) + 1000, np.arange(5,36) + 1000)), ctx_lh_names)},
#     **{k: v for k, v in zip(np.concatenate((np.arange(1,4) + 2000, np.arange(5,36) + 2000)), ctx_rh_names)}
# }
# APARC_DICT_REV = {v: k for k,v in APARC_DICT.items()}

# ------------------------------------------- #
# Cluster labels for inhomogeneity correction #
# ------------------------------------------- #

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

# ----------- #
# ASEG labels #
# ----------- #

POST_ARR = np.array([0,  2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47,
                     49, 50, 51, 52, 53, 54, 58, 60])
APARC_ARR = np.concatenate((np.arange(1, 4) + 1000, np.arange(5, 36) + 1000,
                            np.arange(1, 4) + 2000, np.arange(5, 36) + 2000), axis=0)

APARC_LUT = {k: it_k for it_k, k in enumerate(APARC_ARR)}
POST_LUT = {k: it_k for it_k, k in enumerate(POST_ARR)}
POST_LUT = {**POST_LUT, **{k: POST_LUT[3] if k < 2000 else POST_LUT[42] for k in APARC_ARR}}

POST_AND_APARC_ARR = np.unique(np.concatenate((POST_ARR, APARC_ARR), axis=0))
POST_AND_APARC_LUT = {k: it_k for it_k, k in enumerate(POST_AND_APARC_ARR)}

ASEG_DICT = {
    0: 'background',
    2: 'left cerebral white matter',
    3: 'left cerebral cortex',
    4: 'left lateral ventricle',
    5: 'left inferior lateral ventricle',
    7: 'left cerebellum white matter',
    8: 'left cerebellum cortex',
    10: 'left thalamus',
    11: 'left caudate',
    12: 'left putamen',
    13: 'left pallidum',
    14: '3rd ventricle',
    15: '4th ventricle',
    16: 'brain-stem',
    17: 'left hippocampus',
    18: 'left amygdala',
    24: 'CSF',
    26: 'left accumbens area',
    28: 'left ventral DC',
    30: 'left vessel',
    31: 'left choroid-plexus',
    41: 'right cerebral white matter',
    42: 'right cerebral cortex',
    43: 'right lateral ventricle',
    44: 'right inferior lateral ventricle',
    46: 'right cerebellum white matter',
    47: 'right cerebellum cortex',
    49: 'right thalamus',
    50: 'right caudate',
    51: 'right putamen',
    52: 'right pallidum',
    53: 'right hippocampus',
    54: 'right amygdala',
    58: 'right accumbens area',
    60: 'right ventral DC',
    62: 'right vessel',
    63: 'right choroid plexus',
    77: 'WM hypo',
    80: 'non WM hypo',
    85: 'optic chiasm',
    251: 'cc posterior',
    252: 'cc mid posterior',
    253: 'cc central',
    254: 'cc mid anterior',
    255: 'cc anterior'
}

ASEG_DICT = {k: ASEG_DICT[k] for k in sorted(ASEG_DICT.keys(), key=lambda x: x)}
ASEG_DICT_REV = {v: k for k,v in ASEG_DICT.items()}
ASEG_LUT = {k: it_k for it_k, k in enumerate(ASEG_DICT.keys())}
ASEG_ARR = np.array(list(ASEG_DICT.keys()), dtype='int')

ASEG_APARC_ARR = np.unique(np.concatenate((ASEG_ARR, APARC_ARR), axis=0))
APARC_TO_ASEG_LUT = {**POST_LUT, **{k: POST_LUT[3] for k in APARC_ARR}}
# --------------- #
# SUBFIELD labels #
# --------------- #

HP_LABELS_REV = {
    'Parasubiculum': 203,
    'HATA': 211,
    'Fimbria': 212,
    'Hippocampal_fissure': 215,
    'HP_tail':  226,
    'Presubiculum_head':  233,
    'Presubiculum_body':  234,
    'Subiculum_head': 235,
    'Subiculum_body': 236,
    'CA1-head': 237,
    'CA1-body': 238,
    'CA3-head': 239,
    'CA3-body': 240,
    'CA4-head': 241,
    'CA4-body': 242,
    'GC-ML-DG-head': 243,
    'GC-ML-DG-body': 244,
    'molecular_layer_HP-head':  245,
    'molecular_layer_HP-body': 246,
}
AM_LABELS_REV = {
    'Lateral-nucleus': 7001,
    'Basal-nucleus': 7003,
    'Central-nucleus': 7005,
    'Medial-nucleus': 7006,
    'Cortical-nucleus': 7007,
    'Accessory-Basal-nucleus': 7008,
    'Corticoamygdaloid-transitio': 7009,
    'Anterior-amygdaloid-area-AAA': 7010,
    'Paralaminar-nucleus': 7015,
}

SUBFIELDS_LABELS_REV = {**{'Background': 0}, **HP_LABELS_REV, **AM_LABELS_REV}
SUBFIELDS_LABELS_REV = {k: SUBFIELDS_LABELS_REV[k] for k in sorted(SUBFIELDS_LABELS_REV.keys(), key=lambda x: SUBFIELDS_LABELS_REV[x])}
SUBFIELDS_LABELS = {v: k for k,v in SUBFIELDS_LABELS_REV.items()}
SUBFIELDS_LABELS_LUT = {k: it_k for it_k, k in enumerate(SUBFIELDS_LABELS.keys())}
SUBFIELDS_LABELS_ARR = np.array(list(SUBFIELDS_LABELS.keys()), dtype='int')

EXTENDED_SUBFIELDS_LABELS = SUBFIELDS_LABELS.copy()
EXTENDED_SUBFIELDS_LABELS[10000] = 'TotalHP'
EXTENDED_SUBFIELDS_LABELS[10001] = 'TotalAM'
EXTENDED_SUBFIELDS_LABELS = {k: v for k, v in sorted(EXTENDED_SUBFIELDS_LABELS.items(), key=lambda x: x[0])}
EXTENDED_SUBFIELDS_LABELS_REV = {v: k for k,v in EXTENDED_SUBFIELDS_LABELS.items()}
EXTENDED_SUBFIELDS_LABELS_LUT = {k: it_k for it_k, k in enumerate(EXTENDED_SUBFIELDS_LABELS.keys())}
EXTENDED_SUBFIELDS_LABELS_ARR = np.array(list(EXTENDED_SUBFIELDS_LABELS.keys()), dtype='int')

# ------------------- #
# SEGMENTATION labels #
# ------------------- #
# LABELS_TO_WRITE = ['Thalamus', 'Lateral-Ventricle', 'Hippocampus', 'Amygdala', 'Caudate', 'Pallidum', 'Putamen', 'Accumbens', 'Inf-Lat-Ventricle']
# KEEP_LABELS_STR = ['Background'] + ['Right-' + l for l in LABELS_TO_WRITE] + ['Left-' + l for l in LABELS_TO_WRITE]
# UNIQUE_LABELS = np.asarray([lab for labstr, lab in LABEL_DICT.items() if labstr in KEEP_LABELS_STR], dtype=np.uint8)
# KEEP_LABELS_IDX = [SYNTHSEG_LUT[ul] for ul in UNIQUE_LABELS]

FS_DICT = {
    'Background': 0,
    'Right-Hippocampus': 53,
    'Left-Hippocampus': 17,
    'Right-Lateral-Ventricle': 43,
    'Left-Lateral-Ventricle': 4,
    'Right-Thalamus': 49,
    'Left-Thalamus': 10,
    'Right-Amygdala': 54,
    'Left-Amygdala': 18,
    'Right-Putamen': 51,
    'Left-Putamen': 12,
    'Right-Pallidum': 52,
    'Left-Pallidum': 13,
    'Right-Cerebrum-WM': 41,
    'Left-Cerebrum-WM': 2,
    'Right-Cerebellar-WM': 46,
    'Left-Cerebellar-WM': 7,
    'Right-Cerebrum-GM': 42,
    'Left-Cerebrum-GM': 3,
    'Right-Cerebellar-GM': 47,
    'Left-Cerebellar-GM': 8,
    'Right-Caudate': 50,
    'Left-Caudate': 11,
    'Brainstem': 16,
    '4th-Ventricle': 15,
    '3rd-Ventricle': 14,
    'Right-Accumbens': 58,
    'Left-Accumbens': 26,
    'Right-VentralDC': 60,
    'Left-VentralDC': 28,
    'Right-Inf-Lat-Ventricle': 44,
    'Left-Inf-Lat-Ventricle': 5,
}
FS_DICT = {v: k for k,v in FS_DICT.items()}



if not os.path.exists(os.path.join(SYNTHSEG_DIR, 'synthseg_lut.txt')):

    labels_abbr = {
        0: 'BG',
        2: 'L-Cerebral-WM',
        3: 'L-Cerebral-GM',
        4: 'L-Lat-Vent',
        5: 'L-Inf-Lat-Vent',
        7: 'L-Cerebell-WM',
        8: 'L-Cerebell-GM',
        10: 'L-TH',
        11: 'L-CAU',
        12: 'L-PU',
        13: 'L-PA',
        14: '3-Vent',
        15: '4-Vent',
        16: 'BS',
        17: 'L-HIPP',
        18: 'L-AM',
        26: 'L-ACC',
        28: 'L-VDC',
        41: 'R-Cerebral-WM',
        42: 'R-Cerebral-GM',
        43: 'R-Lat-Vent',
        44: 'R-Inf-Lat-Vent',
        46: 'R-Cerebell-WM',
        47: 'R-Cerebell-WM',
        49: 'R-TH',
        50: 'R-CAU',
        51: 'R-PU',
        52: 'R-PA',
        53: 'R-HIPP',
        54: 'R-AM',
        58: 'R-ACC',
        60: 'R-VDC',
    }

    fs_lut = {0: {'name': 'Background', 'R': 0, 'G': 0, 'B': 0}}
    with open(os.path.join(os.environ['FREESURFER_HOME'], 'FreeSurferColorLUT.txt'), 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            info = [r for r in row[None][0].split(' ') if r != '']
            if len(info) < 5: continue
            try:
                fs_lut[int(info[0])] = {'name': info[1].lower().replace('-', ' '), 'R': info[2], 'G': info[3],
                                        'B': info[4]}
            except:
                continue

    header = ['index', 'name', 'abbreviation', 'R', 'G', 'B', 'mapping']
    label_dict = [
        {'index': label, 'name': fs_lut[label]['name'],
         'abbreviation': labels_abbr[label] if label in labels_abbr else fs_lut[label]['name'],
         'R': fs_lut[label]['R'], 'G': fs_lut[label]['G'], 'B': fs_lut[label]['B'], 'mapping': it_label}
        for it_label, label in enumerate(POST_AND_APARC_ARR)
    ]

    with open(os.path.join(SYNTHSEG_DIR, 'synthseg_lut.txt'), 'w') as csvfile:
        csvreader = csv.DictWriter(csvfile, fieldnames=header, delimiter='\t')
        csvreader.writeheader()
        csvreader.writerows(label_dict)
