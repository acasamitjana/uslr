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
from utils.uslr_utils import initialize_graph_linear
from utils.fn_utils import compute_centroids_ras
from utils.synthmorph_utils import labels_registration
from setup import *


def process_subject(subject, bids_loader, force_flag=False):
    sub_str = 'sub-' + subject
    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x: len(bids_loader.get(extension='nii.gz', subject=subject, session=x, suffix='T1w')) > 0, timepoints))

    seg_dict = {'scope':'synthseg', 'extension':'nii.gz', 'subject':subject, 'suffix':['T1wdseg', 'dseg']}

    deformations_dir = join(DIR_PIPELINES['uslr-lin'], sub_str, 'deformations')
    if not exists(deformations_dir): makedirs(deformations_dir)

    if len(timepoints) == 1:
        print('[done] It has only 1 timepoint. No registration is made.')
        return

    if not all([len(bids_loader.get(**{**seg_dict, 'session': tp})) > 0 for tp in timepoints]):
        print('[error] Not all timepoints are segmented. Please, run preprocess/synthseg.py first.')
        return [subject]

    if all([exists(join(deformations_dir, str(tp_ref) + '_to_' + str(tp_flo) + '.npy'))
            for tp_ref, tp_flo in itertools.combinations(timepoints, 2)]) and not force_flag:
        print('[done] It has been already processed. ')
        return []

    print(' * Computing centroids.')

    centroid_dict = {}
    ok = {}
    for tp in timepoints:
        seg_file = bids_loader.get(**{**seg_dict, 'session': tp})
        if len(seg_file) != 1:
            print('[error] More than one segmentation file found. Skipping.')
            return [subject]

        centroid_dict[tp], ok[tp] = compute_centroids_ras(seg_file[0].path, labels_registration)

    print(' * Pairwise registration.')
    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
        print('   > registering T=' + str(tp_ref) + ' and t=' + tp_flo + ';', end=' ', flush=True)

        filename = str(tp_ref) + '_to_' + str(tp_flo)
        if exists(join(deformations_dir, filename + '.npy')) and not force_flag:
            print('done.')
            continue

        initialize_graph_linear([centroid_dict[tp_ref], centroid_dict[tp_flo]], join(deformations_dir, filename + '.aff'))

        if not exists(join(deformations_dir, filename + '.npy')):
            print(join(deformations_dir, filename + '.npy'))
        print('done.')

if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# ----------------------------- #')
    print('# Linear USLR: initialize graph  #')
    print('# ----------------------------- #')
    print('\n\n')

    parser = ArgumentParser(description='Computes linear pairwise registration between images of the graph.')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--force', action='store_true')

    arguments = parser.parse_args()
    bids_dir = arguments.bids
    force_flag = arguments.force

    print('Loading dataset. \n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['uslr-lin'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if arguments.subjects is None else arguments.subjects

    ####################
    # Run registration #
    ####################

    failed_subjects = []
    for it_subject, subject in enumerate(subject_list):
        print('\nSubject : ' + subject)
        t_init = time.time()
        try:
            process_subject(subject, bids_loader, force_flag=force_flag)
        except:
            failed_subjects.append(subject)

        print('Total registration time: ' + str(np.round(time.time() - t_init, 2)) + '\n')

    f = open(join(LOGS_DIR, 'initialize_graph_linear.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) +
          '. See ' + join(LOGS_DIR, 'initialize_graph_linear.txt') + ' for more information.' )
    print('\n')
    print('# --------- FI (USLR-LIN: graph initialization) --------- #')
    print('\n')
