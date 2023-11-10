import time
import os
import pdb
from os.path import exists, join
from argparse import ArgumentParser

import nibabel as nib
import bids
from skimage.transform import rescale

# project imports
from setup import *
from utils.labels import *
from utils import synthmorph_utils, def_utils

print('\n\n\n\n\n')
print('# ------------------------------------------- #')
print('# SVF projected to TEMPLATE using Pole Ladder #')
print('# ------------------------------------------- #')
print('\n\n')


#####################
# Global parameters #
#####################
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--scope', default='sreg-synthmorph-l1',
                        choices=['sreg-synthmorph', 'synthmorph', 'sreg-lin', 'sreg-synthmorph-l1'])
arg_parser.add_argument('--template', default='SYNTHMORPH', choices=['MNI', 'SYNTHMORPH'])
arg_parser.add_argument('--factor', default=2, type=int, help="Downsampling factor of the SVF fields.")
arg_parser.add_argument('--restrict_tp', action='store_true',
                        help="If True, it restricts the computation on the FIRST and LAST timepoints.")
arg_parser.add_argument('--force', action='store_true')
arg_parser.add_argument('--tm', default='time_to_bl_days', choices=['time_to_bl_days', 'age'],
                        help="Time marker. What metric of time to use (related to the temp_variance parameter.")

arguments = arg_parser.parse_args()
bidsdir = arguments.bids
initial_subject_list = arguments.subjects
scope = arguments.scope     
template_str = arguments.template
force_flag = arguments.force
restrict_tp = arguments.restrict_tp
factor = arguments.factor
time_marker = arguments.tm

###############
# Data loader #
###############
if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'synthseg')
slr_lin_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'slr-lin')
slr_nonlin_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'slr-nonlin')

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


if template_str == 'MNI':
    ref_template = MNI_TEMPLATE
    ref_template_seg = MNI_TEMPLATE_SEG
elif template_str == 'SYNTHMORPH':
    ref_template = synthmorph_utils.atlas_file
    ref_template_seg = synthmorph_utils.atlas_seg_file

else:
    raise ValueError('Please, specify a valid template name.')


proxyatlas = nib.load(synthmorph_utils.atlas_file)
proxytemplate = nib.load(ref_template)
proxytemplateseg = nib.load(ref_template_seg)
template_shape = proxytemplate.shape

################
# Read volumes #
################
print('Reading volumes.')
for it_subject, subject in enumerate(subject_list):
    t_init = time.time()

    print(' - Subject: ' + str(subject) + '. Timepoints: ', end=' ', flush=True)
    timepoints = bids_loader.get_session(subject=subject)

    sess_tsv = bids_loader.get(suffix='sessions', extension='tsv', subject=subject)
    sess_df = sess_tsv[0].get_df()
    sess_df = sess_df.set_index('session_id')
    sess_df = sess_df[~sess_df.index.duplicated(keep='last')]

    slr_nonlin_dir_subj = join(slr_nonlin_dir, 'sub-' + subject)
    svf_dict = {'sub': subject, 'suffix': 'svf'}
    if len(timepoints) <= 1:
        print('Not Processed. This subject has only 1 timepoint.')
        continue

    svf_template_filename = 'sub-' + subject + '_space-' + template_str +'_desc-field_nonlinear.nii.gz'
    if not exists(join(slr_nonlin_dir_subj, svf_template_filename)):
        print('Not Processed. SVF file to template not found.')
        continue

    anat_template_filename = 'sub-' + subject + '_desc-nonlinTemplate_T1w.nii.gz'
    if not exists(join(slr_nonlin_dir_subj, anat_template_filename)):
        continue

    proxysvf_template = nib.load(join(slr_nonlin_dir_subj, svf_template_filename))
    proxyanat_template = nib.load(join(slr_nonlin_dir_subj, anat_template_filename))

    for tp in timepoints:
        if tp == timepoints[-1]:
            print(tp, end='.\n')
        else:
            print(tp, end=', ', flush=True)

        svf_template_image = np.array(proxysvf_template.dataobj)

        svf_tp_file = bids_loader.get(scope='slr-nonlin', suffix='svf', subject=subject, session=tp, extension='nii.gz', regex_search=False)
        if len(svf_tp_file) != 1:
            pdb.set_trace()
            print("Skipping. Other than 1 SVF file found for subject: " + subject + " and tp: " + tp + ". " + str(len(svf_tp_file)))
            continue

        svf_tp_temp_fname = bids_loader.build_path({**svf_tp_file[0].entities, 'space': template_str}, validate=False,
                                                   path_patterns=BIDS_PATH_PATTERN, strict=False, absolute_paths=False)

        if not exists(join(slr_nonlin_dir, svf_tp_temp_fname)) or force_flag:

            proxysvf_tp = nib.load(svf_tp_file[0].path)
            svf_image = np.array(proxysvf_tp.dataobj)

            if svf_image.shape[0] == 3: svf_image = np.transpose(svf_image, axes=(1, 2, 3, 0))

            # Up-scale SVF
            svf_shape = svf_image.shape[:3]
            svf_image = rescale(svf_image, [factor]*3 + [1])

            # svf_v2r = proxycoef.affine.copy()
            # for c in range(3):
            #     svf_v2r[:-1, c] = svf_v2r[:-1, c] / factor
            # svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([factor]*3) - 1))

            # Linear resampling
            long_svf = svf_image
            # if template_str != 'SYNTHMORPH':
            #     long_svf = -svf_image
            # else:
            #     long_svf = svf_image

            # Pole Ladder
            steps = int(np.ceil(np.sqrt(np.sum(svf_template_image ** 2, axis=-1)).max() / 0.5)) #+ 10
            long_svf_T = def_utils.pole_ladder(long_svf, svf_template_image, steps)


            # Down-scale SVF
            svf_image = rescale(long_svf_T, [1 / factor] * 3 + [1])
            svf_v2r = proxytemplate.affine.copy()
            for c in range(3):
                svf_v2r[:-1, c] = svf_v2r[:-1, c] * factor
            svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([1/factor] * 3) - 1))

            # Save
            proxy_long_svf_T = nib.Nifti1Image(svf_image, svf_v2r)
            nib.save(proxy_long_svf_T, join(slr_nonlin_dir, svf_tp_temp_fname))

    print('DONE. Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)) + ' seconds.')

print('\n# ---------- DONE -------------- #')