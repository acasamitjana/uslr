import pdb
from os.path import exists, join, basename, dirname
from os import makedirs
import time
from argparse import ArgumentParser
from datetime import date, datetime
import subprocess
import json
import shutil

import bids
import numpy as np
import nibabel as nib

# project imports
from src import uslr
from utils.io_utils import write_json_derivatives
from utils import synthmorph_utils
from setup import *

def process_subject(subject, bids_loader, cp_shape, force_flag=False):
    im_ent = {'scope': basename(DIR_PIPELINES['uslr-lin']), 'space': 'SUBJECT', 'acquisition': 1,
              'subject': subject, 'suffix': 'T1w', 'extension': 'nii.gz'}
    svf_ent = {'scope': 'uslr-nonlin', 'subject': subject, 'suffix': 'svf'}

    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x:  len(bids_loader.get(session=x, **im_ent)) > 0, timepoints))
    if len(timepoints) == 1:
        print('[warning] it has only 1 timepoint. Skipping.')
        return subject

    dir_nonlin_subj = join(DIR_PIPELINES['uslr-lin'], 'sub-' + subject)
    deformations_dir = join(dir_nonlin_subj, 'deformations')

    linear_template = {}
    for file in bids_loader.get(subject=subject, desc='linTemplate', extension='nii.gz'):
        if 'dseg' in file.entities['suffix']:
            linear_template['dseg'] = file
        elif 'mask' in file.entities['suffix']:
            linear_template['mask'] = file
        elif file.entities['suffix'] == 'T1w':
            linear_template['image'] = file

    exp_dict = {
        'date': date.today().strftime("%d/%m/%Y"),
        'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        'cost': cost,
    }
    write_json_derivatives([1, 1, 1], linear_template['image'].shape,
                           join(dir_nonlin_subj, 'sub-' + subject + '_desc-nonlinTemplate_anat.json'),
                           extra_kwargs=exp_dict)

    if not exists(join(deformations_dir, str(timepoints[-2]) + '_to_' + str(timepoints[-1]) + '.svf.nii.gz')):
        print('[error] No observations found. Skipping.')
        return subject

    # Check if subject has been processed.
    if not force_flag:
        svf_file_list = []
        for tp in timepoints:
            svf_file_list += bids_loader.get(**svf_ent, session=tp)
        if len(timepoints) == len(svf_file_list):
            print('[done] It has already been processed. Skipping.')
            return

    ####################################################################################################
    ####################################################################################################
    t_init = time.time()

    svf_v2r = np.load(join(dir_nonlin_subj, 'sub-' + subject + '_desc-svf_v2r.npy'))

    if verbose: print('  [' + str(subject) + ' - Building the graph] Reading transforms ...')

    class Value():
        def __init__(self, mid): self.id = mid

    timepoints_class = [Value(mid=m) for m in timepoints]
    graph_structure = uslr.init_st2(timepoints_class, deformations_dir, cp_shape, se=None)  # ball(3))
    R, M, W, NK = graph_structure

    if verbose: print('  [' + str(subject) + ' - USLR-nonlin] Computing the graph ...')


    if cost == 'l2':
        Tres = uslr.st2_L2_global(R, W, len(timepoints))

    else:
        Tres = uslr.st2_L1_chunks(R, M, W, len(timepoints), num_cores=4)

    for it_tp, tp in enumerate(timepoints):
        dir_nonlin_sess = join(dir_nonlin_subj, 'ses-' + tp, 'anat')
        if not exists(dir_nonlin_sess): makedirs(dir_nonlin_sess)

        filename = bids_loader.build_path({'subject': subject, 'suffix': 'T1w', 'session': tp},
                                          path_patterns=BIDS_PATH_PATTERN, validate=False, absolute_paths=False)
        filename = basename(filename)
        filename = filename.replace('T1w', 'svf')
        img = nib.Nifti1Image(np.transpose(Tres[..., it_tp], axes=(1, 2, 3, 0)).astype('float32'), svf_v2r)
        nib.save(img, join(dir_nonlin_sess, filename))

    subprocess.call(['rm', '-rf', deformations_dir])
    if verbose: print('  [' + str(subject) + ' - USLR-nonlin] Total Elapsed time: ' + str(time.time() - t_init) + '\n')


if __name__ == '__main__':
    
    print('\n\n\n\n\n')
    print('# ------------------------------ #')
    print('# Non-linear SLR: compute graph  #')
    print('# ------------------------------ #')
    print('\n\n')
    
    
    # Parameters
    parser = ArgumentParser(description='Runs the non-linear longitudinal registration algorithm to compute the latent'
                                        ' transforms, i.e., solve the spanning tree')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--factor', type=int, default=2, help="(default=2) downsample factor to run the algorithm.")
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    bids_dir = args.bids
    cost = args.cost
    initial_subject_list = args.subjects
    force_flag = args.force
    verbose = args.verbose
    factor = args.factor

    print('Loading dataset ...\n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['uslr-nonlin'])
    bids_loader.add_derivatives(DIR_PIPELINES['uslr-lin'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if args.subjects is None else args.subjects

    atlas = nib.load(synthmorph_utils.atlas_file)
    cp_shape = tuple([s // factor for s in atlas.shape])

    for it_subject, subject in enumerate(subject_list):

        failed_subjects = []
        for it_subject, subject in enumerate(subject_list):
            print('* Subject: ' + subject + '. (' + str(it_subject) + '/' + str(len(subject_list)) + ').')

            try:
                ms = process_subject(subject, bids_loader, cp_shape, force_flag=force_flag)
            except:
                ms = subject

            if ms is not None:
                failed_subjects.append(ms)

        f = open(join(LOGS_DIR, 'computee_graph_nonlinear.txt'), 'w')
        f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
        f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

        print('\n')
        print('Total failed subjects ' + str(len(failed_subjects)) +
              '. See ' + join(LOGS_DIR, 'compute_graph_nonlinear.txt') + ' for more information.')
        print('\n')
        print('# --------- FI (USLR-NONLIN: compute latent transforms) --------- #')
        print('\n')

