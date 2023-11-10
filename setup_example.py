import os
import pdb
import json

filename_entities = ['subject', 'session', 'run', 'acquisition', 'suffix', 'extension', 'task', 'tracer', 'reconstruction']
anat_modalities = ['T1w', 'T2w', 'T2star', 'T2starw', 'FLAIR', 'FLASH', 'PD', 'PDw', 'PDT2']
BIDS_PATH_PATTERN = [
    "sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<T1w|T2w|T2star|T2starw|FLAIR|FLASH|PD|PDw|PDT2|inplaneT[12]|angio|dseg|posteriors|svf|T1wmask|T1wdseg|T2wmask|T2wdseg|FLAIRmask|FLAIRdseg>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<func>|func}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<bold|cbv|sbref>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<pet>|pet}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_trc-{tracer}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<pet>}{extension<.nii|.nii.gz|.json||.txt|.npy>|.nii.gz}",
]

# MRI Templates
repo_home = os.environ.get('PYTHONPATH')

MNI_TEMPLATE = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_t1norm_tal_nlin_sym_09a.nii.gz')
MNI_to_ATLAS = os.path.join(repo_home, 'data', 'atlas', 'mni_to_synthmorph_atlas.aff.npy')

MNI_ATLAS_TEMPLATE = os.path.join(repo_home, 'data', 'atlas', 'mni_reg_to_synthmorph_atlas.nii.gz')
MNI_ATLAS_TEMPLATE_SEG = os.path.join(repo_home, 'data', 'atlas', 'mni_reg_to_synthmorph_atlas.seg.nii.gz')
MNI_ATLAS_TEMPLATE_MASK = os.path.join('data', 'atlas', 'mni_reg_to_synthmorph_atlas.mask.nii.gz')

MNI_TEMPLATE_SEG = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_synthseg_tal_nlin_sym_09a.nii.gz')
MNI_TEMPLATE_MASK = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_mask_tal_nlin_sym_09a.nii.gz')

# BIDS directories ---- Environment variables.
BIDS_DIR = os.environ['BIDS_DIR']
if BIDS_DIR[-1] == '/': BIDS_DIR = BIDS_DIR[:-1]
if 'SLR_DIR' in os.environ.keys():
    SLR_DIR = os.environ['SLR_DIR']
else:
    SLR_DIR = os.path.dirname(BIDS_DIR)

DERIVATIVES_DIR = os.path.join(SLR_DIR, 'derivatives')
RESULTS_DIR = os.path.join(SLR_DIR, 'results')

if not BIDS_DIR: raise ValueError("Please, specify environment variable DB")

SYNTHSEG_DIR = os.path.join( os.path.dirname(BIDS_DIR), 'synthseg')
USLR_LIN_DIR = os.path.join(DERIVATIVES_DIR, 'uslr-lin')
USLR_NONLIN_DIR = os.path.join(DERIVATIVES_DIR, 'uslr-nonlin')

if not os.path.exists(SYNTHSEG_DIR): os.makedirs(SYNTHSEG_DIR)
data_descr_path = os.path.join(SYNTHSEG_DIR, 'dataset_description.json')
if not os.path.exists(data_descr_path):
    data_descr = {}
    data_descr['Name'] = 'synthseg'
    data_descr['BIDSVersion'] = '1.0.2'
    data_descr['GeneratedBy'] = [{'Name': 'synthseg'}]
    data_descr['Description'] = 'SynthSeg segmentation using Freesurfer 7.3.2'
    data_descr_path = os.path.join(SYNTHSEG_DIR, 'dataset_description.json')
    json_object = json.dumps(data_descr, indent=4)
    with open(data_descr_path, 'w') as outfile:
        outfile.write(json_object)

if not os.path.exists(USLR_LIN_DIR): os.makedirs(USLR_LIN_DIR)
data_descr_path = os.path.join(USLR_LIN_DIR, 'dataset_description.json')
if not os.path.exists(data_descr_path):
    data_descr = {}
    data_descr['Name'] = 'uslr-lin'
    data_descr['BIDSVersion'] = '1.0.2'
    data_descr['GeneratedBy'] = [{'Name': 'uslr-lin'}]
    data_descr['Description'] = 'USLR pipeline linear stream'
    json_object = json.dumps(data_descr, indent=4)
    with open(data_descr_path, 'w') as outfile:
        outfile.write(json_object)

if not os.path.exists(USLR_NONLIN_DIR): os.makedirs(USLR_NONLIN_DIR)
data_descr_path = os.path.join(USLR_NONLIN_DIR, 'dataset_description.json')
if not os.path.exists(data_descr_path):
    data_descr = {}
    data_descr['Name'] = 'uslr-nonlin'
    data_descr['BIDSVersion'] = '1.0.2'
    data_descr['GeneratedBy'] = [{'Name': 'uslr-nonlin'}]
    data_descr['Description'] = 'USLR pipeline non-linear stream'
    data_descr_path = os.path.join(USLR_NONLIN_DIR, 'dataset_description.json')
    json_object = json.dumps(data_descr, indent=4)
    with open(data_descr_path, 'w') as outfile:
        outfile.write(json_object)


VERBOSE = os.environ['VERBOSE'] if 'VERBOSE' in os.environ.keys() else False
if VERBOSE:
    print('     ')
    print('     ')
    print('DATASET USED ($BIDS_DIR): ' + BIDS_DIR)
    print('DERIVATIVES_DIR: ' + DERIVATIVES_DIR)
    print('SYNTHSEG DIR: ' + SYNTHSEG_DIR)
    print('USLR-LIN DIR: ' + USLR_LIN_DIR)
    print('USLR-NONLIN DIR: ' + USLR_NONLIN_DIR)
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = 'cpu'

