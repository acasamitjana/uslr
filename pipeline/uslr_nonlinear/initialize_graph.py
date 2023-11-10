# imports
import pdb
from os.path import join, exists
from os import makedirs, rmdir
import time
from argparse import ArgumentParser
import shutil
import bids

# third party imports
import numpy as np
import nibabel as nib
import itertools

# project imports
from setup import *
from utils.slr_utils import initialize_graph_nonlinear_multimodal
from utils import io_utils, synthmorph_utils
from utils.fn_utils import compute_centroids_ras

#####################
# Global parameters #
#####################
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--force', action='store_true')
arg_parser.add_argument('--max_iter', type=int, default=5, help='Epochs for registration refinement.')
arg_parser.add_argument('--int_resolution', type=int, default=2, help='Downsample resolution of the SVF integration '
                                                                      '(the lower the smoother).')
arg_parser.add_argument('--grad_penalty', type=float, default=1, help='Penalty for the gradient on instance refinement. '
                                                                      'It should be linked by the expected amount of '
                                                                      'deformation.')
arg_parser.add_argument('--cpu', action='store_true')

arguments = arg_parser.parse_args()
bidsdir = arguments.bids
initial_subject_list = arguments.subjects
force_flag = arguments.force

if not arguments.cpu:
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
    device = 'cuda:0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = 'cpu'

print('\n\n\n\n\n')
print('# --------------------------------- #')
print('# Non-linear SLR: initialize graph  #')
print('# --------------------------------- #')
print('\n\n')


###################
# Tree parameters #
###################
print('Loading dataset ...\n')
Amri = nib.load(synthmorph_utils.atlas_file)
Aaff = Amri.affine.astype('float32')

if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'synthseg')
slr_dir = '/media/biofisica/BIG_DATA/ADNI-T1' # os.path.dirname(bidsdir)
slr_lin_dir = os.path.join(slr_dir, 'derivatives', 'slr-lin')
slr_nonlin_dir = os.path.join(slr_dir, 'derivatives', 'slr-nonlin')
if not exists(slr_nonlin_dir): makedirs(slr_nonlin_dir)

data_descr_path = join(slr_nonlin_dir, 'dataset_description.json')
if not exists(data_descr_path):
    data_descr = {}
    data_descr['Name'] = 'slr-nonlin'
    data_descr['BIDSVersion'] = '1.0.2'
    data_descr['GeneratedBy'] = [{'Name': 'slr-nonlin'}]
    data_descr['Description'] = 'USLR Nonlinear stream'

    json_object = json.dumps(data_descr, indent=4)
    with open(data_descr_path, 'w') as outfile:
        outfile.write(json_object)

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
idx_process = [it_s for it_s, s in enumerate(subject_list) if s == 'ADNI051S1296'][0]
subject_list = subject_list[idx_process + 1:]

####################
# Run registration #
####################
missing_subjects = []
for it_subject, subject in enumerate(subject_list):
    print('* Subject: ' + subject + '. (' + str(it_subject) + '/' + str(len(subject_list)) + ').')
    try:
        timepoints = bids_loader.get_session(subject=subject)
        ent_im = {'scope': 'slr-lin', 'space': 'SUBJECT', 'acquisition': 1, 'extension': 'nii.gz', 'subject': subject, 'suffix':'T1w'}

        dir_lin_sess = join(slr_lin_dir, 'sub-' + subject)
        dir_nonlin_sess = join(slr_nonlin_dir, 'sub-' + subject)
        if not exists(dir_nonlin_sess): makedirs(dir_nonlin_sess)

        linear_template = {}
        for file in bids_loader.get(subject=subject, desc='linTemplate', extension='nii.gz'):
            if 'dseg' in file.entities['suffix']:
                linear_template['dseg'] = file
            elif 'mask' in file.entities['suffix']:
                linear_template['mask'] = file
            elif file.entities['suffix'] == 'T1w':
                linear_template['image'] = file

        nonlinear_template = join(dir_nonlin_sess, linear_template['image'].filename.replace('linTemplate', 'nonlinTemplate'))

        if not exists(linear_template['image'].path):
            print('Skipping: images not available. Please, run first the linear SLR stream.')
            continue

        if len(timepoints) == 1:
            print('Skipping: it has only 1 timepoint. ')
            continue

        if not exists(join(dir_nonlin_sess, 'sub-' + subject + '_desc-atlas_aff.npy')):
            centroid_sbj, ok = compute_centroids_ras(linear_template['dseg'], synthmorph_utils.labels_registration)
            centroid_atlas = np.load(synthmorph_utils.atlas_cog_file)

            Msbj = synthmorph_utils.getM(centroid_atlas[:, ok > 0], centroid_sbj[:, ok > 0], use_L1=False)
            np.save(join(dir_nonlin_sess, 'sub-' + subject + '_desc-atlas_aff.npy'), Msbj)

        else:
            Msbj = np.load(join(dir_nonlin_sess, 'sub-' + subject + '_desc-atlas_aff.npy'))

        im_files = bids_loader.get(**ent_im)
        if not im_files:
            missing_subjects.append(subject)
            print('-- WARNING -- Subject: ' + subject + ' has not data available.')
            continue

        image_list = []
        for i_file in im_files:
            session = i_file.entities['session']
            if 'run' in i_file.entities.keys():
                session += '.' + str(i_file.entities['run'])
            image_list.append(session)

        deformations_dir = join(dir_nonlin_sess, 'deformations')
        if not exists(deformations_dir): makedirs(deformations_dir)

        if all([exists(join(deformations_dir, str(tp_ref) + '_to_' + str(tp_flo) + '.svf.nii.gz')) for tp_ref, tp_flo in
                itertools.combinations(image_list, 2)]):
            print('Skipping: it has already been processed.')
            continue

        tempdir = join('/tmp', 'slr_nonlin_bias', subject)
        if not exists(tempdir): makedirs(tempdir)

        if not exists(join(dir_nonlin_sess, 'sub-' + subject + '_desc-svf_v2r.npy')):
            svf_v2r = Aaff.copy()
            for c in range(3): svf_v2r[:-1, c] = svf_v2r[:-1, c] * 2
            svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([0.5, 0.5, 0.5]) - 1))
            np.save(join(dir_nonlin_sess, 'sub-' + subject + '_desc-svf_v2r.npy'), Msbj @ svf_v2r)

        for tp_ref, tp_flo in itertools.combinations(image_list, 2):
            print('   o From T=' + str(tp_ref) + ' to T=' + str(tp_flo) + '.', end='', flush=True)

            filename = str(tp_ref) + '_to_' + str(tp_flo)
            filename_rev = str(tp_flo) + '_to_' + str(tp_ref)

            t_init = time.time()

            ent_sess = {'session': tp_ref}
            ent_sessmask = {'suffix': ['T1wmask', 'mask'], 'session': tp_ref}
            if '.' in tp_ref:
                tp, run = tp_ref.split('.')
                ent_sess = {'session': tp, 'run': run}
                ent_sessmask = {'suffix': ['T1wmask', 'mask'], 'session': tp, 'run': run}

            im_ref = bids_loader.get(**{**ent_im, **ent_sess})
            mask_ref = bids_loader.get(**{**ent_im, **ent_sessmask})
            if len(im_ref) != 1 or len(mask_ref) != 1:
                missing_subjects.append(subject)
                print('[WARNING] Image and mask files for T='+ str(tp_ref) + ' are not found. Please, refine the search.')
                print('          Resuming subject ' + subject)
                break

            dict_ref = {'image': im_ref[0].path, 'mask': mask_ref[0].path}

            ent_mod = {'session': tp_flo}
            ent_modmask = {'suffix': ['T1wmask', 'mask'], 'session': tp_flo}
            if '.' in tp_flo:
                tp, run = tp_flo.split('.')
                ent_mod = {'session': tp, 'run': run}
                ent_modmask = {'suffix': ['T1wmask', 'mask'], 'session': tp, 'run': run}

            im_flo = bids_loader.get(**{**ent_im, **ent_mod})
            mask_flo = bids_loader.get(**{**ent_im, **ent_modmask})
            if len(im_flo) != 1 or len(mask_flo) != 1:
                missing_subjects.append(subject)
                print('\n[WARNING] Image and mask files for T=' + str(tp_flo) + ' are not found. Please, refine the search.')
                print('          Resuming subject ' + subject + ' and T=' + str(tp_flo))
                break
            dict_flo = {'image': im_flo[0].path, 'mask': mask_flo[0].path}

            initialize_graph_nonlinear_multimodal([dict_ref, dict_flo], Msbj, results_dir=deformations_dir,
                                                  filename=filename, epochs=arguments.max_iter,
                                                  grad_penalty=arguments.grad_penalty, full_size=False,
                                                  int_resolution=arguments.int_resolution)

            print(' Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)))
    except:
        pass
    print('  -- DONE -- Subject ' + subject + ' has been registered.')
    print('\n')

    # shutil.rmtree(tempdir)

print('\n# ---------- DONE -------------- #')
