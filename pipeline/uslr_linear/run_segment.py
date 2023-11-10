import os
import pdb
from argparse import ArgumentParser

import bids
from joblib import delayed, Parallel

# project imports
from src import bids_loader
from utils import seg_utils_bids as seg_utils
from setup import *

print('\n\n\n\n\n')
print('# -------------------------------------------- #')
print('# Running the longitudinal segmentation script #')
print('# -------------------------------------------- #')
print('\n\n')

#####################
# Global parameters #
#####################

# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--subjects', default=None, nargs='+',
                        help="Subjets to segment. Set to None to segment the whole BIDS_DIR.")
arg_parser.add_argument('--num_cores', default=1, type=int,
                        help="Number of cores used to segment in parallel multiple subjects.")
arg_parser.add_argument('--spatial_variance', default=['inf'], nargs='+',
                        help="Variance of the Gaussian kernel on the intensities. Set to inf for NAl")
arg_parser.add_argument('--temp_variance', default=['inf'], nargs='+',
                        help="Variance of the Gaussian kernel on the time_to_bl_years. Set to inf for NA")
arg_parser.add_argument('--scope', default='synthseg', choices=['synthseg', 'freesurfer', 'freesurfer-subfields'],
                        help="Scope under derivatives to find the labels.")
arg_parser.add_argument('--reg_method', default='slr', choices=['slr', 'pairwise'], help="Registration algorithm used to propagate the labels.")
arg_parser.add_argument('--tm', default='time_to_bl_days', choices=['time_to_bl_days', 'age'],
                        help="Time marker. What metric of time to use (related to the temp_variance parameter.")
arg_parser.add_argument('--force', action='store_true', help="Set to True to overwrite existin previous segmentations")
arg_parser.add_argument('--type_map', default='onehot_map', choices=[None, 'distance_map', 'onehot_map', 'gauss_map'],
                        help="Parameterise segmentation as distances, one_hot or via the posteriors (None).")
arg_parser.add_argument('--space', default='image', choices=['image', 'subject'])
arg_parser.add_argument('--fusion', default='post', choices=['post', 'seg'])
arg_parser.add_argument('--not_normalise', action='store_false', help="")
arg_parser.add_argument('--smooth', action='store_true', help="")
arg_parser.add_argument('--all_labels', action='store_true', help="")
arg_parser.add_argument('--save_seg', action='store_true', help="")

arguments = arg_parser.parse_args()
bidsdir = arguments.bids
initial_subject_list = arguments.subjects
num_cores = arguments.num_cores
spatial_variance = [float(v) for v in arguments.spatial_variance] #in grayscale
temp_variance = [float(v) for v in arguments.temp_variance] #in days^2
force_flag = arguments.force
scope = arguments.scope
reg_method = arguments.reg_method
time_marker = arguments.tm
smooth = arguments.smooth
type_map = arguments.type_map
space = arguments.space
fusion = arguments.fusion
normalise_flag = arguments.not_normalise
all_labels_flag = arguments.all_labels
save_seg = arguments.save_seg
print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = 'cpu'

##############
# Processing #
##############

if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
init_seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', scope)
slr_lin_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'slr-lin')

seg_dirname = 'slr-lin'  if reg_method == 'slr' else 'lin'
def_scope = 'slr-lin' if reg_method == 'slr' else 'lin'
if scope != 'synthseg':
    seg_dirname += '-' + scope

if space != 'image':
    seg_dirname += '-' + space

seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', seg_dirname)
if not os.path.exists(seg_dir):
    os.makedirs(seg_dir)
    data_descr_path = os.path.join(seg_dir, 'dataset_description.json')
    if not os.path.exists(data_descr_path):
        data_descr = {}
        data_descr['Name'] = seg_dirname
        data_descr['BIDSVersion'] = '1.0.2'
        data_descr['GeneratedBy'] = [{'Name': seg_dirname}]
        data_descr['Description'] = 'USLR segmentation label fusion'

        json_object = json.dumps(data_descr, indent=4)
        with open(data_descr_path, 'w') as outfile:
            outfile.write(json_object)

print('Loading dataset. \n')
db_file = os.path.join(os.path.dirname(BIDS_DIR), 'BIDS-raw.db')
if not os.path.exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False, database_path=db_file)

bids_loader.add_derivatives(init_seg_dir)
bids_loader.add_derivatives(slr_lin_dir)
if slr_lin_dir != seg_dir:
    bids_loader.add_derivatives(seg_dir)

subject_list = bids_loader.get_subjects() if initial_subject_list is None else initial_subject_list

print('[LONGITUDINAL SEGMENTATION] Start processing.')

if reg_method != 'slr':
    print('Still not implemented')
    segmenter = seg_utils.LabelFusionDirect()
    exit()

elif space == 'template':
    print('Still not implemented')
    segmenter = seg_utils.LabelFusionTemplate()
    exit()

else:
    segmenter = seg_utils.LabelFusion(bids_loader, def_scope=def_scope, seg_scope=scope, output_scope=seg_dirname,
                                      temp_variance=temp_variance, spatial_variance=spatial_variance, smooth=smooth,
                                      time_marker=time_marker, type_map=type_map, fusion_method=fusion,
                                      normalise=normalise_flag, all_labels_flag=all_labels_flag, save_seg=save_seg)

if num_cores > 1:
    missing_subjects = []
    results = Parallel(n_jobs=num_cores)(
        delayed(segmenter.label_fusion)(subject, force_flag=force_flag) for subject in subject_list)

else:
    for subject in subject_list:
        segmenter.label_fusion(subject, force_flag=force_flag)
