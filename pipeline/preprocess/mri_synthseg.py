import os
from os.path import exists, dirname, islink, join, basename
from os import makedirs, remove
from argparse import ArgumentParser
import pdb
import subprocess
import json
import shutil
import nibabel as nib
import csv
import numpy as np

from setup import *
import bids
from utils.labels import POST_AND_APARC_ARR

print('## [Preprocessing] Run SynthSeg segmentations  ##')

# parse arguments
parser = ArgumentParser(description="SynthSeg segmentation using freesurfer implementation. It includes  segmentation "
                                    "and parcellation volumes, a summary volfile (synthseg dir) and the resampled image "
                                    "(rawdata dir). No robust or QC flags are used.", epilog='\n')
# input/outputs
parser.add_argument("--bids", default=BIDS_DIR, help="Bids root directory, including rawdata")
parser.add_argument("--force", action='store_true', help="Force the script to overwriting existing segmentations in the derivatives/synthseg directory.")
parser.add_argument("--gpu", action='store_true', help="Try to run SynthSeg on gpu.")

args = parser.parse_args()
bidsdir = args.bids
gpu_flag = args.gpu
force_flag = args.force

extra_title = '-'.join(['']*len(bidsdir))
extra_blank_0 = ' '.join(['']*min(0, np.clip(len(bidsdir) - 37, 0, 1000000000)))
extra_blank = ' '.join(['']*len(extra_blank_0))
print('\n\n\n\n\n')
print('# ---------------------------------------------------------------------------' + extra_title + ' #')
print('# SynthSeg segmentation on BIDS dataset ' + bidsdir + extra_blank_0 + ' #')
print('#    - Robust segmentation                                                   ' + extra_blank + ' #')
print('#    - Compute volumes                                                       ' + extra_blank  + ' #')
if force_flag is True:
    print('#    - OVERWRITING existing files                                            ' + extra_blank + ' #')
else:
    print('#    - Running only on subjects/sessions where segmentation is missing       ' + extra_blank + ' #')
print('# ---------------------------------------------------------------------------' + extra_blank + ' #')
print('\n\n')

print('Loading dataset. \n')
db_file = join(dirname(BIDS_DIR), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False, database_path=db_file)

subject_list = bids_loader.get_subjects()

input_files = []
res_files = []
output_files = []
vol_files = []
discarded_files = []
for subject in subject_list:
    t1w_list = list(filter(lambda x: 'acq-1' not in x.filename, bids_loader.get(subject=subject, extension='nii.gz', suffix='T1w')))
    for t1w_i in t1w_list:
        raw_dirname = t1w_i.dirname
        if not exists(raw_dirname): makedirs(raw_dirname)

        synthseg_dirname = join(SYNTHSEG_DIR, 'sub-' + subject, 'ses-' + t1w_i.entities['session'], t1w_i.entities['datatype'])
        entities = {k: str(v) for k, v in t1w_i.entities.items() if k in filename_entities}
        entities['acquisition'] = '1'
        anat_res = basename(bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN))
        anat_seg = anat_res.replace('T1w', 'T1wdseg')
        anat_vols = anat_seg.replace('nii.gz', 'tsv')
        if not exists(join(synthseg_dirname, anat_seg)) or force_flag:
            try:
                proxy = nib.load(join(raw_dirname, t1w_i.filename))
                if len(proxy.shape) != 3:
                    with open(join(synthseg_dirname, 'excluded_file.txt'), 'w') as f:
                        f.write('File excluded due to wrong image dimensions.')

                    discarded_files += [join(raw_dirname, t1w_i.filename)]
                    continue

                if any([r > 7 for r in np.sum(np.sqrt(np.abs(proxy.affine * proxy.affine)), axis=0)[:3].tolist()]):
                    with open(join(synthseg_dirname, 'excluded_file.txt'), 'w') as f:
                        f.write('File excluded due to large resolution in some image dimension.')
                    discarded_files += [join(raw_dirname, t1w_i.filename)]
                    continue

                # if all(np.sum(np.abs(proxy.affine * proxy.affine), axis=0) > 0.01): continue
                input_files += [join(raw_dirname, t1w_i.filename)]
                res_files += [join(raw_dirname, anat_res)]
                output_files += [join(synthseg_dirname, anat_seg)]
                vol_files += [join(synthseg_dirname, anat_vols)]

            except:
                with open(join(synthseg_dirname, 'excluded_file.txt'), 'w') as f:
                    f.write('File excluded due to an error reading the file or computing image shape and resolution.')
                discarded_files += [join(raw_dirname, t1w_i.filename)]


# Segment image
with open('/tmp/discardedfiles.txt', 'w') as f:
    for i_f in discarded_files:
        f.write(i_f)
        f.write('\n')

with open('/tmp/inputfiles.txt', 'w') as f:
    for i_f in input_files:
        f.write(i_f)
        f.write('\n')

with open('/tmp/resfiles.txt', 'w') as f:
    for i_f in res_files:
        f.write(i_f)
        f.write('\n')

with open('/tmp/outputfiles.txt', 'w') as f:
    for i_f in output_files:
        f.write(i_f)
        f.write('\n')

with open('/tmp/volfiles.txt', 'w') as f:
    for i_f in vol_files:
        f.write(i_f)
        f.write('\n')


if len(output_files) >= 1:
    gpu_cmd = [''] if gpu_flag else ['--cp']
    subprocess.call(['mri_synthseg', '--i', '/tmp/inputfiles.txt', '--o', '/tmp/outputfiles.txt',
                     '--resample', '/tmp/resfiles.txt', '--vol', '/tmp/volfiles.txt', '--threads', '16','--robust',
                     '--parc'] + gpu_cmd)


for file in vol_files:
    fr = open(file, "r")
    fw = open('/tmp/vol.tsv', "w")

    reader = csv.reader(fr, delimiter=',')
    writer = csv.writer(fw, delimiter='\t')
    writer.writerows(reader)

    fr.close()
    fw.close()

    subprocess.call(['cp', '/tmp/vol.tsv', file])

for i, r in zip(input_files, res_files):
    if not exists(r):
        try:
            subprocess.call(['ln', '-s', i, r])
        except:
            subprocess.call(['cp', i, r])

    proxy = nib.load(r)
    aff = proxy.affine
    pixdim = str(np.sqrt(np.sum(aff * aff, axis=0))[:-1])
    im_json = {
               "Resolution": {
                   "R": str(pixdim[0]),
                   "A": str(pixdim[1]),
                   "S": str(pixdim[2])
               },
               "ImageShape": {
                   "X": str(proxy.shape[0]),
                   "Y": str(proxy.shape[1]),
                   "Z": str(proxy.shape[2])
               }}

    
    json_object = json.dumps(im_json, indent=4)
    with open(r.replace('nii.gz', 'json'), 'w') as outfile:
        outfile.write(json_object)


for seg_file in output_files:
    proxy = nib.load(seg_file)
    aff = proxy.affine
    pixdim = str(np.sqrt(np.sum(aff * aff, axis=0))[:-1])
    im_json = {"Manual": False,
               "Resolution": {
                   "R": str(pixdim[0]),
                   "A": str(pixdim[1]),
                   "S": str(pixdim[2])
               },
               "ImageShape": {
                   "X": str(proxy.shape[0]),
                   "Y": str(proxy.shape[1]),
                   "Z": str(proxy.shape[2])
               }}

    
    json_object = json.dumps(im_json, indent=4)
    with open(seg_file.replace('nii.gz', 'json'), 'w') as outfile:
        outfile.write(json_object)


print('******************')
print('****** DONE ******')
print('******************')

