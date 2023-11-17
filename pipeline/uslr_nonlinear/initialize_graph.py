# imports
from setup import *

import copy
from os.path import join, exists, dirname, basename
from os import makedirs
import time
from argparse import ArgumentParser
import bids

# third party imports
import numpy as np
import nibabel as nib
import itertools

# project imports
from utils.uslr_utils import initialize_graph_nonlinear
from utils import synthmorph_utils
from utils.fn_utils import compute_centroids_ras


def process_subject(subject, bids_loader, reg_args, force_flag=False):
    im_ent = {'scope': basename(DIR_PIPELINES['uslr-lin']), 'space': 'SUBJECT', 'acquisition': '1',
              'subject': subject, 'suffix': 'T1w', 'extension': 'nii.gz'}

    mask_ent = copy.deepcopy(im_ent)
    mask_ent['suffix'] = ['T1wmask', 'mask']

    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x:  len(bids_loader.get(session=x, **im_ent)) > 0, timepoints))
    if not timepoints:
        print('[warning] Subject: ' + subject + ' has not timepoints available. Skipping')
        return subject

    dir_nonlin_sess = join(DIR_PIPELINES['uslr-nonlin'], 'sub-' + subject)
    if not exists(dir_nonlin_sess): makedirs(dir_nonlin_sess)

    if len(timepoints) == 1:
        print('[warning] it has only 1 timepoint. Skipping.')
        return subject

    deformations_dir = join(dir_nonlin_sess, 'deformations')
    if not exists(deformations_dir): makedirs(deformations_dir)

    linear_template = {}
    for file in bids_loader.get(subject=subject, desc='linTemplate', extension='nii.gz'):
        if 'dseg' in file.entities['suffix']:
            linear_template['dseg'] = file
        elif 'mask' in file.entities['suffix']:
            linear_template['mask'] = file
        elif file.entities['suffix'] == 'T1w':
            linear_template['image'] = file

    if not exists(linear_template['image'].path):
        print('Skipping: images not available. Please, run first the linear SLR stream.')
        return subject

    if not exists(join(dir_nonlin_sess, 'sub-' + subject + '_desc-atlas_aff.npy')):
        centroid_sbj, ok = compute_centroids_ras(linear_template['dseg'], synthmorph_utils.labels_registration)
        centroid_atlas = np.load(synthmorph_utils.atlas_cog_file)

        Msbj = synthmorph_utils.getM(centroid_atlas[:, ok > 0], centroid_sbj[:, ok > 0], use_L1=False)
        np.save(join(dir_nonlin_sess, 'sub-' + subject + '_desc-atlas_aff.npy'), Msbj)

    else:
        Msbj = np.load(join(dir_nonlin_sess, 'sub-' + subject + '_desc-atlas_aff.npy'))

    if not exists(join(dir_nonlin_sess, 'sub-' + subject + '_desc-svf_v2r.npy')):
        Amri = nib.load(synthmorph_utils.atlas_file)
        Aaff = Amri.affine.astype('float32')
        svf_v2r = Aaff.copy()
        for c in range(3): svf_v2r[:-1, c] = svf_v2r[:-1, c] * 2
        svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([0.5, 0.5, 0.5]) - 1))
        np.save(join(dir_nonlin_sess, 'sub-' + subject + '_desc-svf_v2r.npy'), Msbj @ svf_v2r)


    if all([exists(join(deformations_dir, str(tp_ref) + '_to_' + str(tp_flo) + '.svf.nii.gz')) for tp_ref, tp_flo in
            itertools.combinations(timepoints, 2)]) and not force_flag:
        print('[done] It has already been processed.')
        return subject

    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
        print('   o From T=' + str(tp_ref) + ' to T=' + str(tp_flo) + '.', end='', flush=True)
        t_init = time.time()

        filename = str(tp_ref) + '_to_' + str(tp_flo)

        im_ref = bids_loader.get(**im_ent, session=tp_ref)
        mask_ref = bids_loader.get(**mask_ent, session=tp_ref)
        if len(im_ref) != 1 or len(mask_ref) != 1:
            print('[error] Image and mask files for T=' + str(tp_ref) + ' are not found.', end=' ', flush=True)
            print('Please, refine the search. Resuming subject ' + subject)
            return subject

        im_flo = bids_loader.get(**im_ent, session=tp_flo)
        mask_flo = bids_loader.get(**mask_ent, session=tp_flo)
        if len(im_flo) != 1 or len(mask_flo) != 1:
            print('[error] Image and mask files for T=' + str(tp_flo) + ' are not found.', end=' ', flush=True)
            print('Please, refine the search. Resuming subject ' + subject)
            return subject

        dict_ref = {'image': im_ref[0].path, 'mask': mask_ref[0].path}
        dict_flo = {'image': im_flo[0].path, 'mask': mask_flo[0].path}

        initialize_graph_nonlinear([dict_ref, dict_flo], Msbj, results_dir=deformations_dir,
                                   filename=filename, epochs=reg_args.max_iter, full_size=False,
                                   grad_penalty=reg_args.grad_penalty, int_resolution=reg_args.int_resolution)

        print(' Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)))

if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# --------------------------------- #')
    print('# Non-linear SLR: initialize graph  #')
    print('# --------------------------------- #')
    print('\n\n')

    parser = ArgumentParser(description='Computes non-linear pairwise registration between images of the graph.')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--max_iter', type=int, default=5, help='Epochs for registration refinement.')
    parser.add_argument('--int_resolution', type=int, default=2, help='Downsample resolution of the SVF integration '
                                                                          '(the lower the smoother).')
    parser.add_argument('--grad_penalty', type=float, default=1, help='Penalty for the gradient on instance refinement. '
                                                                          'It should be linked by the expected amount of '
                                                                          'deformation.')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    bids_dir = args.bids
    init_subject_list = args.subjects
    force_flag = args.force

    if not args.cpu:
        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
        device = 'cuda:0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'


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

    ####################
    # Run registration #
    ####################
    failed_subjects = []
    for it_subject, subject in enumerate(subject_list):
        print('* Subject: ' + subject + '. (' + str(it_subject) + '/' + str(len(subject_list)) + ').')
        try:
            ms = process_subject(subject, bids_loader, args, force_flag=force_flag)
        except:
            ms = subject

        if ms is not None:
            failed_subjects.append(ms)

    f = open(join(LOGS_DIR, 'initialize_graph_nonlinear.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) +
          '. See ' + join(LOGS_DIR, 'initialize_graph_nonlinear.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (USLR-NONLIN: initialize graph) --------- #')
    print('\n')
