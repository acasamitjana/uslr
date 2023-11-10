# imports
import os
import pdb
from os.path import join, exists, dirname
from os import makedirs, rmdir
import time
from argparse import ArgumentParser
import shutil

# third party imports
import numpy as np
import itertools
import bids
import nibabel as nib

# project imports
from src import bids_loader
from utils.slr_utils import initialize_graph_linear
from utils.fn_utils import compute_centroids_ras
from utils.synthmorph_utils import labels_registration
from utils.io_utils import get_run
from setup import *


print('\n\n\n\n\n')
print('# ----------------------------- #')
print('# Linear SLR: initialize graph  #')
print('# ----------------------------- #')
print('\n\n')

#####################
# Global parameters #
#####################
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--slr_dir', default=DERIVATIVES_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--force', action='store_true')

arguments = arg_parser.parse_args()
bidsdir = arguments.bids
initial_subject_list = arguments.subjects
force_flag = arguments.force
if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'synthseg')
slr_dir = '/media/biofisica/BIG_DATA/ADNI-T1' # os.path.dirname(bidsdir)
slr_lin_dir = os.path.join(slr_dir, 'derivatives', 'slr-lin')
if not exists(slr_lin_dir): makedirs(slr_lin_dir)

data_descr_path = join(slr_lin_dir, 'dataset_description.json')
if not exists(data_descr_path):
    data_descr = {}
    data_descr['Name'] = 'slr-lin'
    data_descr['BIDSVersion'] = '1.0.2'
    data_descr['GeneratedBy'] = [{'Name': 'slr-lin'}]
    data_descr['Description'] = 'USLR Linear stream'

    json_object = json.dumps(data_descr, indent=4)
    with open(data_descr_path, 'w') as outfile:
        outfile.write(json_object)

###################
# Tree parameters #
###################

print('Loading dataset. \n')
db_file = join(dirname(BIDS_DIR), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False, database_path=db_file)

bids_loader.add_derivatives(seg_dir)
bids_loader.add_derivatives(slr_lin_dir)
subject_list = bids_loader.get_subjects() if initial_subject_list is None else initial_subject_list

####################
# Run registration #
####################

missing_subjects = []
run_subject = 0

for it_subject, subject in enumerate(subject_list):
    print('Subject : ' + subject, end='. ', flush=True)

    t_init = time.time()
    timepoints = bids_loader.get_session(subject=subject)

    deformations_dir = join(slr_lin_dir, 'sub-' + subject, 'deformations')
    if not exists(deformations_dir): makedirs(deformations_dir)
    linear_template = join(slr_lin_dir, 'sub-' + subject, 'anat', 'sub-' + subject + '_desc-linTemplate_T1w.nii.gz')

    if len(timepoints) == 1:
        print('It has only 1 timepoint. No registration is made.')
        continue

    timepoints = list(filter(lambda x: len(bids_loader.get(extension='nii.gz', subject=subject, session=x, suffix='T1w')) > 0, timepoints))
    # timepoints = list(filter(lambda x: not exists(join(dirname(bids_loader.get(scope='synthseg',subject=subject, session=x, return_type='filename')[0]), 'excluded_file.txt')), timepoints))

    if not all([len(bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, session=tp, suffix=['T1wdseg', 'dseg'])) > 0 for tp in timepoints]):
        missing_subjects.append(subject)
        print('Not all timepoints are pre-processed.')
        continue

    if all([exists(join(deformations_dir, str(tp_ref) + '_to_' + str(tp_flo) + '.aff')) for tp_ref, tp_flo in itertools.combinations(timepoints, 2)]) and not force_flag:
        print('It has been already processed.')
        continue

    print('Computing centroids.')
    try:
        centroid_dict = {}
        ok = {}
        for tp in timepoints:
            seg_file = bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, session=tp, suffix=['T1wdseg', 'dseg'])
            centroid_dict[tp], ok[tp] = compute_centroids_ras(seg_file[0].path, labels_registration)

        first_repeated = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

            filename = str(tp_ref) + '_to_' + str(tp_flo)

            ref_seg_file = bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, session=tp_ref, suffix=['T1wdseg', 'dseg'], regex_search=True)
            ref_seg_file = list(filter(lambda x: 'run' not in x.entities.keys() or ('run' in x.entities.keys() and x.entities['run'] == '01'), ref_seg_file))
            flo_seg_file = bids_loader.get(scope='synthseg', extension='nii.gz', subject=subject, session=tp_flo, suffix=['T1wdseg', 'dseg'], regex_search=True)
            ref_proxy = nib.load(ref_seg_file[0].path)
            flo_proxy = nib.load(flo_seg_file[0].path)

            if exists(join(deformations_dir, filename + '.aff')) and not force_flag: continue

            print('* Registering T=' + str(tp_ref) + ' and T=' + str(tp_flo) + '.')
            initialize_graph_linear([centroid_dict[tp_ref], centroid_dict[tp_flo]], join(deformations_dir, filename + '.aff'))

            if not exists(join(deformations_dir, filename + '.aff')):
                print(join(deformations_dir, filename + '.aff'))
    except:
        missing_subjects.append(subject)

    print('Total registration time: ' + str(np.round(time.time() - t_init, 2)))

print('Missed subjects: ')
print(missing_subjects)
print('\n')
print('# --------- FI ---------------- #')
print('\n')
