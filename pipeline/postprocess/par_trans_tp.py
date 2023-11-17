import pdb

from setup import *

import time
import copy
from os.path import exists, join, dirname
from argparse import ArgumentParser

import nibabel as nib
import bids

# project imports
from utils.labels import *
from utils import synthmorph_utils, def_utils, fn_utils


def process_subject(subject, bids_loader, args):
    failed_tp = []
    timepoints = bids_loader.get_session(subject=subject)
    if len(timepoints) <= 1:
        print('[done] This subject has only 1 timepoint. Skipping')
        return

    sess_tsv = bids_loader.get(suffix='sessions', extension='tsv', subject=subject)
    sess_df = sess_tsv[0].get_df()
    sess_df = sess_df.set_index('session_id')
    sess_df = sess_df[~sess_df.index.duplicated(keep='last')]

    uslr_nonlin_dir_subj = join(DIR_PIPELINES['uslr-nonlin'], 'sub-' + subject)
    uslr_mni_dir_sbj = join(DIR_PIPELINES['subject-mni'], 'sub-' + subject)
    svf_ent = {'sub': subject, 'suffix': 'svf'}

    svf_template_filename = 'sub-' + subject + '_space-' + args.template + '_desc-field_svf.nii.gz'
    svf_image = bids_loader.get(subject=subject, scope='subject-mni', space=args.template, desc='field', suffix='svf')
    if len(svf_image) != 1:
        print('[error]. N=' + str(len(svf_image)) + ' SVF file(s) to template are found (expected=1). Skipping')
        return failed_tp.append(subject)

    template_image = bids_loader.get(subject=subject, scope='uslr-nonlin', desc='nonlinTemplate', suffix='T1w')
    if len(template_image) != 1:
        print('[error] N=' + str(len(template_image)) + ' subject-specific template are found (expected=1). Skipping')
        return failed_tp.append(subject)

    proxysvf_template = nib.load(svf_image[0].path)
    svf_template_image = np.array(proxysvf_template.dataobj)
    proxyanat_template = nib.load(template_image[0].path)

    for tp in timepoints:
        print(' * Session: ' + tp, end='; ', flush=True)

        svf_tp_file = bids_loader.get(scope='uslr-nonlin', suffix='svf', subject=subject, session=tp,
                                      extension='nii.gz', regex_search=False)
        if len(svf_tp_file) != 1:
            N = len(svf_tp_file)
            print('[error] N=' + str(N) + ' SVF found for subject: ' + subject + ' and tp: ' + tp + '. Skipping.')
            failed_tp.append('sub-' + subject + '_ses-' + tp)
            continue

        svf_tp_file = svf_tp_file[0]
        svf_tp_temp_fname = bids_loader.build_path({**svf_tp_file.entities, 'space': args.template}, validate=False,
                                                   path_patterns=BIDS_PATH_PATTERN, strict=False, absolute_paths=False)
        svf_tp_temp_fpath = join(DIR_PIPELINES['uslr-nonlin'], svf_tp_temp_fname)
        if exists(svf_tp_temp_fpath) and not args.force:
            print('[done] This subject has already been processed.')
            return

        proxysvf_tp = nib.load(svf_tp_file.path)
        svf_aff = proxysvf_tp.affine
        long_svf_arr = np.array(proxysvf_tp.dataobj)
        #
        if long_svf_arr.shape[0] == 3:
            long_svf_arr = np.transpose(long_svf_arr, axes=(1, 2, 3, 0))

        if exists(join(DIR_PIPELINES['uslr-nonlin'], 'sub-' + subject, 'sub-' + subject + '_desc-atlas_aff.npy')):
            M = np.load(join(DIR_PIPELINES['uslr-nonlin'],  'sub-' + subject, 'sub-' + subject + '_desc-atlas_aff.npy'))
        else:
            print('[error] Affine matrix to SynthMorph atlas not found. Skipping.')
            failed_tp.append('sub-' + subject + '_ses-' + tp)
            continue

        # Up-scale SVF:
        # Just to adapt to the resolution of SVF to MNI, but we do not rescale it to not alter the true
        # SVF. Moreover it will be downscaled again later on.
        print('upscaling svf; ', end='', flush=True)
        svf_atlas_v2r = copy.deepcopy(proxyatlas.affine)
        for c in range(3): svf_atlas_v2r[:-1, c] = svf_atlas_v2r[:-1, c] * args.factor
        svf_atlas_v2r[:-1, -1] -= svf_atlas_v2r[:-1, :-1] @ (0.5 * (np.array([1 / args.factor]*3) - 1))
        new_vox_size = np.sqrt(np.sum(proxyatlas.affine * proxyatlas.affine, axis=0))[:-1]
        long_svf_arr, _ = fn_utils.rescale_voxel_size(long_svf_arr, svf_atlas_v2r, new_vox_size, not_aliasing=True)

        # Pole Ladder
        print('running pole ladder; ', end='', flush=True)
        steps = int(np.ceil(np.sqrt(np.sum(svf_template_image ** 2, axis=-1)).max() / 0.5))  # + 10
        long_svf_arr_T = def_utils.pole_ladder(long_svf_arr, svf_template_image, steps)

        # Down-scale SVF:
        # To have the same resolution as the original SVF
        new_vox_size = np.sqrt(np.sum(svf_atlas_v2r * svf_atlas_v2r, axis=0))[:-1]
        long_svf_arr_T, _ = fn_utils.rescale_voxel_size(long_svf_arr_T, proxyatlas.affine, new_vox_size,
                                                        not_aliasing=True)

        # The current SVF is on SyntMorph space (where the SVF to TEMPLATE is also defined).
        # If template!='SynthMorph', we need to specify the correct v2r to align it to the original TEMPLATE space
        # using the affine matrix MNI-SynthMorph.
        if args.template == 'MNI':
            M = np.load(MNI_to_ATLAS)
            svf_1mm_v2r = M @ proxyatlas.affine
        elif args.template == 'SynthMorph':
            svf_1mm_v2r = proxyatlas.affine
        else:
            svf_1mm_v2r = proxytemplate.affine

        svf_v2r = copy.deepcopy(svf_1mm_v2r)
        for c in range(3):
            svf_v2r[:-1, c] = svf_v2r[:-1, c] * args.factor
        svf_v2r[:-1, -1] -= svf_v2r[:-1, :-1] @ (0.5 * (np.array([1 / args.factor] * 3) - 1))

        print('saving.')
        proxy_long_svf_T = nib.Nifti1Image(long_svf_arr_T, svf_v2r)
        nib.save(proxy_long_svf_T, svf_tp_temp_fpath)
    
    




#####################
# Global parameters #
#####################
if __name__ == '__main__':

    parser = ArgumentParser(description='Computes the parallel transport of SVFs to a standard space.')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+')
    parser.add_argument('--template', default='MNI', choices=['MNI', 'Synthmorph'])
    parser.add_argument('--factor', default=2, type=int, help="Downsampling factor of the SVF fields.")
    parser.add_argument('--restrict_tp',
                        action='store_true',
                        help="If True, it restricts the computation on the FIRST and LAST timepoints.")
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--tm',
                        default='time_to_bl_days',
                        choices=['time_to_bl_days', 'age'],
                        help="Time marker. What metric of time to use (related to the temp_variance parameter.")

    args = parser.parse_args()

    print('\n\n\n\n\n')
    print('# ------------------------------------------- #')
    print('# SVF projected to TEMPLATE using Pole Ladder #')
    print('# ------------------------------------------- #')
    print('\n\n')

    if args.template == 'MNI':
        ref_template = MNI_TEMPLATE
        ref_template_seg = MNI_TEMPLATE_SEG
    elif args.template == 'SynthMorph':
        ref_template = synthmorph_utils.atlas_file
        ref_template_seg = synthmorph_utils.atlas_seg_file
    else:
        raise ValueError('Please, specify a valid template name.')

    proxyatlas = nib.load(synthmorph_utils.atlas_file)
    proxytemplate = nib.load(ref_template)
    proxytemplate_dseg = nib.load(ref_template_seg)
    template_shape = proxytemplate.shape

    print('Loading dataset ...\n')
    db_file = join(dirname(args.bids), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['uslr-nonlin'])
    bids_loader.add_derivatives(DIR_PIPELINES['uslr-lin'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    bids_loader.add_derivatives(DIR_PIPELINES['subject-mni'])
    subject_list = bids_loader.get_subjects() if args.subjects is None else args.subjects

    ###################
    # Run Pole-Ladder #
    ###################
    failed_subjects = []
    for it_subject, subject in enumerate(subject_list):
        print('* Subject: ' + subject + '. (' + str(it_subject) + '/' + str(len(subject_list)) + ').')
        t_init = time.time()
        # try:
        ms = process_subject(subject, bids_loader, args)
        print('  Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)) + ' seconds.')
        # except:
        #     ms = [subject]

        if ms is not None:
            failed_subjects.extend(ms)

    f = open(join(LOGS_DIR, 'register_template.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) +
          '. See ' + join(LOGS_DIR, 'register_template.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (USLR-NONLIN: register_template) --------- #')
    print('\n')

