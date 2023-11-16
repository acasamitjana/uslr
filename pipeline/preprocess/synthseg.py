import os
from os.path import exists, dirname, join, basename
from os import makedirs
from argparse import ArgumentParser
import subprocess
import nibabel as nib
import csv
import numpy as np
import bids

from setup import *
from utils.labels import SYNTHSEG_APARC_LUT
from utils.io_utils import write_json_derivatives

def segment(subject_list, force_flag=False):
    '''
   Get subjects to segment
   :param subject_list: initial list with subjects' id.
   :return: input_files, res_files, output_files, vol_files to be passed to mri_synthseg
   '''
    input_files, res_files, output_files, vol_files, discarded_files = [], [], [], [], []
    for subject in subject_list:
        timepoints = bids_loader.get_session(subject=subject)
        # t1w_list = bids_loader.get(subject=subject, extension='nii.gz', suffix='T1w', acquisition=None)
        # for t1w_i in t1w_list:
        for tp in timepoints:
            synthseg_dirname = join(DIR_PIPELINES['seg'], 'sub-' + subject, 'ses-' + tp, 'anat')

            # Select a single T1w image per session
            t1w_list = bids_loader.get(subject=subject, extension='nii.gz', suffix='T1w', session=tp)
            t1w_list = list(filter(lambda f: f.entities['acquisition'] != '1', t1w_list))
            if len(t1w_list) == 0:
                continue
            elif len(t1w_list) > 1:
                if any(['acquisition' not in f.entities.keys() for f in t1w_list]):
                    t1w_list_r = list(filter(lambda x: 'acquisition' not in x.entities.keys(), t1w_list))
                elif not all(['run' in f.entities.keys() for f in t1w_list]):
                    t1w_list_r = list(filter(lambda x: 'run' not in x.entities.keys(), t1w_list))
                elif all(['run' in f.entities.keys() for f in t1w_list]):
                    t1w_list_r = list(filter(lambda x: x.entities['run'] == '01', t1w_list))
                else:
                    t1w_list_r = t1w_list

                t1w_i = t1w_list_r[0]
                f = open(join(synthseg_dirname, t1w_i[0].filename.replace('nii.gz', 'txt')), 'w')
                f.write('Since there exists more than one T1w image for this session, we choose file to run over the '
                        'entire USLR pipeline with the corresponding segmentation. Refere to the rawdata to check '
                        'correspondence')
            else:
                t1w_i = t1w_list[0]


            raw_dirname = t1w_i.dirname
            entities = {k: str(v) for k, v in t1w_i.entities.items() if k in filename_entities}
            entities['acquisition'] = '1'
            anat_res = basename(bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN))
            anat_seg = anat_res.replace('T1w', 'T1wdseg')
            anat_vols = anat_seg.replace('nii.gz', 'tsv')
            if not exists(join(synthseg_dirname, anat_seg)) or force_flag:
                if len(t1w_list) > 1:
                    f = open(join(synthseg_dirname, anat_seg.replace('nii.gz', 'txt')), 'w')
                    f.write('This is the chosen segmentation file to run over the entire USLR pipeline.')

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
                    res_files += [join(TMP_DIR, anat_res)]
                    output_files += [join(synthseg_dirname, anat_seg)]
                    vol_files += [join(synthseg_dirname, anat_vols)]

                except:
                    with open(join(synthseg_dirname, 'excluded_file.txt'), 'w') as f:
                        f.write('File excluded due to an error reading the file or computing image shape and resolution.')
                    discarded_files += [join(raw_dirname, t1w_i.filename)]

    return input_files, res_files, output_files, vol_files, discarded_files

if __name__ == '__main__':

    # parse arguments
    parser = ArgumentParser(description="SynthSeg segmentation using freesurfer implementation. It includes  segmentation "
                                        "and parcellation volumes, a summary volfile (synthseg dir) and the resampled image "
                                        "(rawdata dir). No robust or QC flags are used.", epilog='\n')
    parser.add_argument('--subjects',
                        default=None,
                        nargs='+',
                        help="(optional) specify which subjects to process")
    parser.add_argument("--bids",
                        default=BIDS_DIR,
                        help="Bids root directory, including rawdata")
    parser.add_argument("--force",
                        action='store_true',
                        help="Force the script to overwriting existing segmentations in the derivatives/synthseg directory.")
    parser.add_argument("--gpu",
                        action='store_true',
                        help="Try to run SynthSeg on gpu.")

    args = parser.parse_args()
    bids_dir = args.bids
    gpu_flag = args.gpu
    init_subject_list = args.subjects
    force_flag = args.force

    extra_title = '-'.join(['']*len(bids_dir))
    extra_blank_0 = ' '.join(['']*min(0, np.clip(len(bids_dir) - 37, 0, 1000000000)))
    extra_blank = ' '.join(['']*len(extra_blank_0))

    print('\n\n\n\n\n')
    print('# ---------------------------------------------------------------------------' + extra_title + ' #')
    print('# SynthSeg segmentation on BIDS dataset ' + bids_dir + extra_blank_0 + ' #')
    print('#    - Robust segmentation                                                   ' + extra_blank + ' #')
    print('#    - Compute volumes                                                       ' + extra_blank  + ' #')
    if force_flag is True:
        print('#    - OVERWRITING existing files                                            ' + extra_blank + ' #')
    else:
        print('#    - Running only on subjects/sessions where segmentation is missing       ' + extra_blank + ' #')
    print('# ---------------------------------------------------------------------------' + extra_blank + ' #')
    print('\n\n')

    print('Loading dataset. \n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list

    input_files, res_files, output_files, vol_files, discarded_files = segment(subject_list, force_flag=force_flag)

    # Segment image
    with open('/tmp/discardedfiles_uslr.txt', 'w') as f:
        for i_f in discarded_files:
            f.write(i_f)
            f.write('\n')

    with open('/tmp/inputfiles_uslr.txt', 'w') as f:
        for i_f in input_files:
            f.write(i_f)
            f.write('\n')

    with open('/tmp/resfiles_uslr.txt', 'w') as f:
        for i_f in res_files:
            f.write(i_f)
            f.write('\n')

    with open('/tmp/outputfiles_uslr.txt', 'w') as f:
        for i_f in output_files:
            f.write(i_f)
            f.write('\n')

    with open('/tmp/volfiles_uslr.txt', 'w') as f:
        for i_f in vol_files:
            f.write(i_f)
            f.write('\n')

    if len(output_files) >= 1:
        gpu_cmd = [''] if gpu_flag else ['--cp']
        subprocess.call(['mri_synthseg', '--i', '/tmp/inputfiles.txt', '--o', '/tmp/outputfiles.txt',
                         '--resample', '/tmp/resfiles.txt', '--vol', '/tmp/volfiles.txt', '--threads', '16',
                         '--robust', '--parc'] + gpu_cmd)

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
        if exists(r.replace('nii.gz', 'json')): continue
        if not exists(r) and exists(i):
            rcode = subprocess.call(['ln', '-s', i, r], stderr=subprocess.PIPE)
            if rcode != 0:
                subprocess.call(['cp', i, r], stderr=subprocess.PIPE)

        proxy = nib.load(r)
        aff = proxy.affine
        pixdim = str(np.sqrt(np.sum(aff * aff, axis=0))[:-1])
        write_json_derivatives(pixdim, proxy.shape, r.replace('nii.gz', 'json'))


    for i_file, seg_file in zip(input_files, output_files):
        if not exists(seg_file): continue

        proxy = nib.load(seg_file)
        aff = proxy.affine
        pixdim = str(np.sqrt(np.sum(aff * aff, axis=0))[:-1])

        if 'rec' in i_file:
            sec = [e.split('-')[-1] for e in basename(i_file).split('_') if 'rec' in e][0]
            write_json_derivatives(pixdim, proxy.shape, seg_file.replace('nii.gz', 'json'),
                                   extra_kwargs={'SelectedSlice': sec})

        else:
            write_json_derivatives(pixdim, proxy.shape, seg_file.replace('nii.gz', 'json'))

    if not exists(join(DIR_PIPELINES['seg'], 'synthseg_lut.txt')):

        labels_abbr = {
            0: 'BG',
            2: 'L-Cerebral-WM',
            3: 'L-Cerebral-GM',
            4: 'L-Lat-Vent',
            5: 'L-Inf-Lat-Vent',
            7: 'L-Cerebell-WM',
            8: 'L-Cerebell-GM',
            10: 'L-TH',
            11: 'L-CAU',
            12: 'L-PU',
            13: 'L-PA',
            14: '3-Vent',
            15: '4-Vent',
            16: 'BS',
            17: 'L-HIPP',
            18: 'L-AM',
            26: 'L-ACC',
            28: 'L-VDC',
            41: 'R-Cerebral-WM',
            42: 'R-Cerebral-GM',
            43: 'R-Lat-Vent',
            44: 'R-Inf-Lat-Vent',
            46: 'R-Cerebell-WM',
            47: 'R-Cerebell-WM',
            49: 'R-TH',
            50: 'R-CAU',
            51: 'R-PU',
            52: 'R-PA',
            53: 'R-HIPP',
            54: 'R-AM',
            58: 'R-ACC',
            60: 'R-VDC',
        }

        fs_lut = {0: {'name': 'Background', 'R': 0, 'G': 0, 'B': 0}}
        with open(join(os.environ['FREESURFER_HOME'], 'FreeSurferColorLUT.txt'), 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                info = [r for r in row[None][0].split(' ') if r != '']
                if len(info) < 5: continue
                try:
                    name = info[1].lower().replace('-', ' ')
                    fs_lut[int(info[0])] = {'name': name, 'R': info[2], 'G': info[3], 'B': info[4]}
                except:
                    continue

        header = ['index', 'name', 'abbreviation', 'R', 'G', 'B', 'mapping']
        label_dict = [
            {'index': label, 'name': fs_lut[label]['name'],
             'abbreviation': labels_abbr[label] if label in labels_abbr else fs_lut[label]['name'],
             'R': fs_lut[label]['R'], 'G': fs_lut[label]['G'], 'B': fs_lut[label]['B'], 'mapping': it_label}
            for it_label, label in SYNTHSEG_APARC_LUT.items()
        ]

        with open(join(DIR_PIPELINES['seg'], 'synthseg_lut.txt'), 'w') as csvfile:
            csvreader = csv.DictWriter(csvfile, fieldnames=header, delimiter='\t')
            csvreader.writeheader()
            csvreader.writerows(label_dict)

    print('\n')
    print('# --------- FI (SynthSeg pipeline) --------- #')
    print('\n')

