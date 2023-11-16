from argparse import ArgumentParser
from os.path import join, exists, dirname
import time

import bids
from joblib import delayed, Parallel
import numpy as np

# project imports
from utils import seg_utils, io_utils
from setup import *



#####################
# Global parameters #
#####################
if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# -------------------------------------------- #')
    print('# Running the longitudinal segmentation script #')
    print('# -------------------------------------------- #')
    print('\n\n')

    # Parameters
    parser = ArgumentParser(description='Computes the prediction of certain models')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--field', default='usl-lin', choices=['usl-lin', 'usl-nonlin'],
                        help="specify the type of deformation field.")
    parser.add_argument('--subjects', default=None, nargs='+',
                        help="Subjets to segment. Set to None (default) to segment the whole BIDS_DIR.")
    parser.add_argument('--num_cores', default=1, type=int,
                        help="Number of cores used to segment in parallel multiple subjects.")
    parser.add_argument('--spatial_variance', default=[9], nargs='+', type=float,
                        help="Variance of the Gaussian kernel on the intensities. Set to inf for NAl")
    parser.add_argument('--temp_variance', default=[np.inf], nargs='+', type=float,
                        help="Variance of the Gaussian kernel on the time_to_bl_years. Set to inf for NA")
    parser.add_argument('--scope', default='synthseg', choices=['synthseg', 'freesurfer'],
                        help="Scope under derivatives to find the labels.")
    parser.add_argument('--tm', default='time_to_bl_days', choices=['time_to_bl_days', 'age'],
                        help="Time marker. What metric of time to use (related to the temp_variance parameter.")
    parser.add_argument('--force', action='store_true', help="Set to True to overwrite existin previous segmentations")
    parser.add_argument('--type_map', default='onehot_map', choices=[None, 'distance_map', 'onehot_map', 'gauss_map'],
                        help="Parameterise segmentation as distances, one_hot or via the posteriors (None).")
    parser.add_argument('--space', default='image', choices=['image', 'subject'])
    parser.add_argument('--all_labels', action='store_true', help="")
    parser.add_argument('--save_seg', action='store_true', help="")

    args = parser.parse_args()

    print('Loading dataset ...\n')
    db_file = join(dirname(args.bids), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False, database_path=db_file)


    io_utils.create_derivative_dir(join(DERIVATIVES_DIR, args.field + '-' + args.scope),
                                   'Longitudinal segmentation initialised using ' + args.scope + ' and using '
                                   + args.field + ' deformations.')
    bids_loader.add_derivatives(DIR_PIPELINES['uslr-nonlin'])
    bids_loader.add_derivatives(DIR_PIPELINES['uslr-lin'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    bids_loader.add_derivatives(DIR_PIPELINES[args.field + '-' + args.scope])
    subject_list = bids_loader.get_subjects() if args.subjects is None else args.subjects

    ##############
    # Processing #
    ##############

    segmenter = seg_utils.LabelFusion(bids_loader,
                                      def_scope=def_scope,
                                      seg_scope=scope,
                                      output_scope=seg_dirname,
                                      temp_variance=temp_variance,
                                      spatial_variance=spatial_variance,
                                      smooth=smooth,
                                      time_marker=time_marker,
                                      type_map=type_map,
                                      fusion_method=fusion,
                                      normalise=normalise_flag,
                                      all_labels_flag=all_labels_flag,
                                      save_seg=save_seg)

    if args.num_cores > 1:
        missing_subjects = []
        results = Parallel(n_jobs=args.num_cores)(
            delayed(segmenter.label_fusion)(subject, force_flag=args.force) for subject in subject_list)

    else:
        for it_subject, subject in enumerate(subject_list):
            print('* Subject: ' + subject + '. (' + str(it_subject) + '/' + str(len(subject_list)) + ').')
            t_init = time.time()
            segmenter.label_fusion(subject, force_flag=args.force)
            print('  Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)) + ' seconds.')
