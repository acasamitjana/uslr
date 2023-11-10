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
from src import bids_loader, slr
from utils.io_utils import build_bids_fileame
from utils import synthmorph_utils
from setup import *

print('\n\n\n\n\n')
print('# ------------------------------ #')
print('# Non-linear SLR: compute graph  #')
print('# ------------------------------ #')
print('\n\n')

#####################
# Global parameters #
#####################


# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2', 'gurobi'], help='Likelihood cost function')
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--force', action='store_true')
arg_parser.add_argument('--verbose', action='store_true')

arguments = arg_parser.parse_args()
bidsdir = arguments.bids
cost = arguments.cost
initial_subject_list = arguments.subjects
force_flag = arguments.force
verbose = arguments.verbose
factor = 2

##############
# Processing #
##############
atlas_slr = nib.load(synthmorph_utils.atlas_file)
cp_shape = tuple([s//factor for s in atlas_slr.shape])

if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'synthseg')
slr_dir = '/media/biofisica/BIG_DATA/ADNI-T1' # os.path.dirname(bidsdir)
slr_lin_dir = os.path.join(slr_dir, 'derivatives', 'slr-lin')
slr_nonlin_dir = os.path.join(slr_dir, 'derivatives', 'slr-nonlin')

db_file = join(os.path.dirname(BIDS_DIR), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False, database_path=db_file)
bids_loader.add_derivatives(seg_dir)
bids_loader.add_derivatives(slr_lin_dir)
bids_loader.add_derivatives(slr_nonlin_dir)
subject_list = bids_loader.get_subjects() if initial_subject_list is None else initial_subject_list

PROCESS_FLAG = False
for it_subject, subject in enumerate(subject_list):
    print('* Subject: ' + subject + '. (' + str(it_subject) + '/' + str(len(subject_list)) + ').')

    if subject == 'ADNI051S1296':
        PROCESS_FLAG = True
    if not PROCESS_FLAG:
        continue

    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x: len(bids_loader.get(extension='nii.gz', subject=subject, session=x, suffix='T1w')) > 0, timepoints))
    # timepoints = list(filter(lambda x: not exists(join(dirname(bids_loader.get(scope='synthseg',subject=subject, session=x, return_type='filename')[0]), 'excluded_file.txt')), timepoints))

    deformations_dir = join(slr_nonlin_dir, 'sub-' + subject, 'deformations')
    dir_nonlin_subj = join(slr_nonlin_dir, 'sub-' + subject)
    if not exists(dir_nonlin_subj): makedirs(dir_nonlin_subj)

    exp_dict = {
        'date': date.today().strftime("%d/%m/%Y"),
        'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        'cost': cost,
    }
    if not exists(join(dir_nonlin_subj, 'sub-' + subject + '_desc-nonlinTemplate_anat.json')):
        json_object = json.dumps(exp_dict, indent=4)
        with open(join(dir_nonlin_subj, 'sub-' + subject + '_desc-nonlinTemplate_anat.json'), "w") as outfile:
            outfile.write(json_object)

    ent_im = {'scope': 'slr-lin', 'space': 'SUBJECT', 'acquisition': 1, 'extension': 'nii.gz', 'subject': subject, 'suffix': 'T1w'}
    im_files = bids_loader.get(**ent_im)
    image_list = []
    for i_file in im_files:
        session = i_file.entities['session']
        if 'run' in i_file.entities.keys():
            session += '.' + str(i_file.entities['run'])
        image_list.append(session)

    if not image_list:
        print('-- WARNING -- Subject: ' + subject + ' has not data available.')
        continue

    if len(image_list) == 1:
        dir_lin_sess = join(slr_lin_dir, 'sub-' + subject)

        linear_template = {}
        for file in bids_loader.get(subject=subject, desc='linTemplate', extension='nii.gz'):
            if 'dseg' in file.entities['suffix']:
                linear_template['dseg'] = file
            elif 'mask' in file.entities['suffix']:
                linear_template['mask'] = file
            elif file.entities['suffix'] == 'T1w':
                linear_template['image'] = file

        nonlinear_template = join(dir_nonlin_subj, linear_template['image'].filename.replace('linTemplate', 'nonlinTemplate'))
        nonlinear_template_mask = join(dir_nonlin_subj, linear_template['mask'].filename.replace('linTemplate', 'nonlinTemplate'))
        nonlinear_template_seg = join(dir_nonlin_subj, linear_template['dseg'].filename.replace('linTemplate', 'nonlinTemplate'))

        if not exists(nonlinear_template): subprocess.call(['cp', linear_template['image'].path, nonlinear_template])
        if not exists(nonlinear_template_mask): subprocess.call(['cp', linear_template['mask'].path, nonlinear_template_mask])
        if not exists(nonlinear_template_seg):  subprocess.call(['cp', linear_template['dseg'].path, nonlinear_template_seg])

        print('Skipping. It has only 1 timepoint.')
        continue

    if not exists(join(deformations_dir, str(image_list[-2]) + '_to_' + str(image_list[-1]) + '.svf.nii.gz')):
        print('!!! WARNING: No observations found. Skipping subject ' + subject + '.')
        continue


    # Check if subject has been processed.
    if not force_flag:
        svf_file_list = []
        for tp in image_list:
            ent_tp = {'session': tp}
            if '.' in tp:
                tp, run = tp.split('.')
                ent_tp = {'session': tp, 'run': run}
            svf_file_list += bids_loader.get(scope='slr-nonlin', subject=subject, suffix='svf', **ent_tp)

        if len(image_list) == len(svf_file_list):
            # TO REMOVE
            year_list = []
            month_list = []
            day_list = []
            for svff in svf_file_list:
                try:
                    _, m, d, t, y = list(filter(lambda x: x != '', time.ctime(os.path.getmtime(svff.path)).split(' ')))
                    print(y)
                    year_list.append(y)
                    month_list.append(m)
                    day_list.append(d)
                except:
                    pdb.set_trace()

            if all([m == 'Sep' for m in month_list]) and all([int(d) >= 22 for d in day_list]):
                print('It has been already processed.')
                if exists(deformations_dir):
                    subprocess.call(['rm', '-rf', deformations_dir])
                continue
            elif all([m == 'Oct' for m in month_list]) and all([int(d) <= 6 for d in day_list]) and all([y == '2023' for m in year_list]):
                print('It has been already processed.')
                if exists(deformations_dir):
                    subprocess.call(['rm', '-rf', deformations_dir])
                continue
            else:
                for svff in svf_file_list:
                    subprocess.call(['rm', '-rf', svff.path])
            # TO REMOVE

            # TO UNCOMMENT
            # print('Skipping: it has already been processed.')
            # continue
            # TO UNCOMMENT


    ####################################################################################################
    ####################################################################################################
    svf_v2r = np.load(join(dir_nonlin_subj, 'sub-' + subject + '_desc-svf_v2r.npy'))

    if verbose: print('[' + str(subject) + ' - Building the graph] Reading transforms ...')
    t_init = time.time()


    class Value():
        def __init__(self, mid): self.id = mid

    timepoints_class = [Value(mid=m) for m in image_list]
    graph_structure = slr.init_st2(timepoints_class, deformations_dir, cp_shape, se=None)#ball(3))
    R, M, W, NK = graph_structure

    if verbose: print('[' + str(subject) + ' - Building the graph] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

    if verbose: print('[' + str(subject) + ' - SLR] Computimg the graph ...')
    t_init = time.time()

    if cost == 'l2':
        Tres = slr.st2_L2_global(R, W, len(image_list))

    else:
        Tres = slr.st2_L1_chunks(R, M, W, len(image_list), solver=cost, num_cores=4)


    for it_tp, tp in enumerate(image_list):
        dir_nonlin_sess = join(dir_nonlin_subj, 'ses-' + tp, 'anat')
        if not exists(dir_nonlin_sess): makedirs(dir_nonlin_sess)
        ent_tp = {'session': tp}
        if '.' in tp:
            tp, run = tp.split('.')
            ent_tp = {'session': tp, 'run': run}

        filename = basename(bids_loader.build_path({'subject': subject,  'suffix': 'T1w', **ent_tp}, path_patterns=BIDS_PATH_PATTERN, validate=False))
        filename = filename.replace('T1w', 'svf')
        img = nib.Nifti1Image(np.transpose(Tres[..., it_tp], axes=(1, 2, 3, 0)).astype('float32'), svf_v2r)
        nib.save(img, join(dir_nonlin_sess, filename))

    subprocess.call(['rm', '-rf', deformations_dir])
    if verbose: print('[' + str(subject) + ' - SLR] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

print('\n# ---------- DONE -------------- #')
print('Warning! Remove PROCESS_FLAG AND DATE AND TIME PROCESSING.')