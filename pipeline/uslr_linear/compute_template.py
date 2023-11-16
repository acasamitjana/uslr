import subprocess
from os.path import join, dirname
from argparse import ArgumentParser
from joblib import delayed, Parallel

import bids
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.morphology import ball

from setup import *
from src.uslr import *
from utils.fn_utils import one_hot_encoding
from utils.labels import SYNTHSEG_APARC_LUT
from utils.uslr_utils import create_template_space

def compute_subject_template(subject, uslr_lin_dir, verbose=True):
    dir_results = join(uslr_lin_dir, 'sub-' + subject)

    image_ent = {'scope': basename(DIR_PIPELINES['seg']), 'extension': 'nii.gz', 'subject': subject, 'suffix': 'T1w'}
    seg_ent = {'scope': basename(DIR_PIPELINES['seg']), 'extension': 'nii.gz', 'subject': subject, 'suffix': 'T1wdseg'}
    aff_ent= {'scope': basename(DIR_PIPELINES['uslr-lin']), 'desc': 'aff', 'extension': 'npy', 'suffix': 'T1w'}

    fname_template = 'sub-' + subject + '_desc-linTemplate_T1w'
    template_path = join(dir_results, fname_template + '.nii.gz')
    template_mask_path = join(dir_results, fname_template + 'mask.nii.gz')
    template_seg_path = join(dir_results, fname_template + 'dseg.nii.gz')

    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x: len(bids_loader.get(**{**seg_ent, 'session':x })) > 0, timepoints))

    if len(timepoints) == 0:
        print('[done] No modalities found. Skipping')
        return

    elif len(timepoints) == 1:
        im_res = bids_loader.get(scope=basename(DIR_PIPELINES['seg']), acquisition='orig', **image_ent)
        if len(im_res) != 1:
            print('[error] Search image command failed. Number of matched files: ' + str(len(im_res)) + '. Skipping.')
            return

        else:
            im_res = im_res[0]

        if not exists(template_path):
            rc = subprocess.call(['ln', '-s', im_res.path, template_path], stderr=subprocess.PIPE)
            if rc != 0:
                subprocess.call(['cp',  im_res.path, template_path], stderr=subprocess.PIPE)

        print('[done] It has only 1 modality. No registration is made.')
        return

    elif len(bids_loader.get(**aff_ent)) != len(timepoints):
        if verbose: print('[error] Could not find observations for all timepoints of subject ' + subject + '.')
        return subject


    if not exists(template_path) or force_flag:

        t_init = time.time()
        if verbose: print('   - Updating vox2ras0  ... ')

        image_dict = {}
        seg_dict = {}
        mask_dict = {}
        mask_reg_dict = {}
        headers_1mm = {}
        headers_orig = {}
        for it_tp, tp in enumerate(timepoints):
            image_tp_ent = {**image_ent, 'session': tp}
            affine_file = bids_loader.get(**{**aff_ent, 'session': tp})
            if len(affine_file) != 1:
                print('[error] Wrong affine file entities')
                # pdb.set_trace()
                return subject

            affine_matrix = np.load(affine_file[0])

            # Load image at the original resolution
            im_file = bids_loader.get(**image_tp_ent, acquisition=None)
            if len(im_file) != 1:
                print('[error] ' + str(len(im_file)) + ' image(s) found in the synthseg directory. Skipping.')
                return subject
            else:
                im_file = im_file[0]

            # Load mask at 1mm3
            seg_file = bids_loader.get(**image_tp_ent, acquisition=1, suffix='T1wdseg')
            if len(seg_file) != 1:
                if all(['run' in r.entities.keys() for r in seg_file]):
                    seg_file = list(filter(lambda x: x.entities['run'] == '01', seg_file))
                else:
                    seg_file = list(filter(lambda x: 'run' not in x.entities.keys(), seg_file))

                if len(seg_file) != 1:
                    print('[error] ' + str(len(seg_file)) + ' segmentation(s) found in the synthseg directory. Skipping')
                    return subject
                else:
                    seg_file = seg_file[0]
            else:
                seg_file = seg_file[0]

            mask_file = seg_file.replace('T1wdseg', 'T1wmask')


            proxyim = nib.load(im_file.path)
            proxymask = nib.load(mask_file.path)
            mask = np.array(proxymask.dataobj)
            mask_dilated = binary_dilation(mask, ball(3)).astype('uint8')

            image_dict[tp] = im_file
            seg_dict[tp] = seg_file
            mask_dict[tp] = mask_file
            mask_reg_dict[tp] = nib.Nifti1Image(mask_dilated, np.linalg.inv(affine_matrix) @ proxymask.affine)

            headers_1mm[tp] = np.linalg.inv(affine_matrix) @ proxymask.affine
            headers_orig[tp] = np.linalg.inv(affine_matrix) @ proxyim.affine

        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #

        if verbose: print('   - Creating subject space  ... ')

        rasMosaic, template_vox2ras0, template_size = create_template_space(list(mask_reg_dict.values()))
        proxytemplate = nib.Nifti1Image(np.zeros(template_size), template_vox2ras0)

        if verbose: print('   - Computing linear template ... ')
        mri_list = []
        mask_list = []
        aparc_aseg = np.array(list(SYNTHSEG_APARC_LUT.keys()))
        seg_list = np.zeros(template_size + (len(aparc_aseg),))
        for it_tp, tp in enumerate(timepoints):
            ent_im = {k: str(v) for k, v in image_dict[tp].entities.items() if k in filename_entities}
            ent_mask = {k: str(v) for k, v in mask_dict[tp].entities.items() if k in filename_entities}
            ent_im['space'] = 'SUBJECT'
            ent_mask['space'] = 'SUBJECT'
            ent_im['acquisition'] = '1'
            ent_mask['acquisition'] = '1'

            im_filename = bids_loader.build_path(ent_im, scope='uslr-lin', strict=False, validate=False,
                                                 path_patterns=BIDS_PATH_PATTERN,  absolute_paths=False)

            mask_filename = bids_loader.build_path(ent_mask, scope='uslr-lin',strict=False, validate=False,
                                                   path_patterns=BIDS_PATH_PATTERN, absolute_paths=False)

            # Anti-aliasing filter for original image
            proxyim = nib.load(image_dict[tp])
            pixdim = np.sqrt(np.sum(proxyim.affine * proxyim.affine, axis=0))[:-1]
            new_vox_size = np.array([1, 1, 1])
            factor = pixdim / new_vox_size
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            im_orig_array = np.array(image_dict[tp].dataobj)
            if len(im_orig_array.shape) > 3:
                im_orig_array = im_orig_array[..., 0]
            volume_filt = gaussian_filter(im_orig_array, sigmas)

            # Resample image
            im_mri = nib.Nifti1Image(volume_filt, headers_orig[tp])
            im_mri = vol_resample(proxytemplate, im_mri)
            nib.save(im_mri, join(uslr_lin_dir, im_filename))
            mri_list.append(np.array(im_mri.dataobj))

            # Resample mask
            im_mri = vol_resample(proxytemplate, mask_reg_dict[tp])
            nib.save(im_mri, join(uslr_lin_dir, mask_filename))
            mask_list.append(np.array(im_mri.dataobj))

            # Resample seg
            im_mri = nib.Nifti1Image(np.array(seg_dict[tp].dataobj), headers_1mm[tp])
            im_mri = vol_resample(proxytemplate, im_mri, mode='nearest')
            seg_list += one_hot_encoding(np.array(im_mri.dataobj), categories=aparc_aseg)

        template = np.median(mri_list, axis=0)
        template = template.astype('uint8')
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, template_path)

        template = np.sum(mask_list, axis=0)/len(mask_list) > 0.5
        template = template.astype('uint8')
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, template_mask_path)

        seg_hard = np.argmax(seg_list, axis=-1)
        template = np.zeros(template_size, dtype='int16')
        for it_l, l in enumerate(aparc_aseg): template[seg_hard == it_l] = l
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, template_seg_path)

        linear_etiv = template_mask_path.replace('T1wmask', 'etiv')
        linear_etiv = linear_etiv.replace('nii.gz', 'npy')
        np.save(linear_etiv, np.sum(template > 0))

        if verbose: print('   - Total Elapsed time: ' + str(time.time() - t_init) + '\n')


if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# ----------------------------- #')
    print('# Linear SLR: compute template  #')
    print('# ----------------------------- #')
    print('\n\n')

    parser = ArgumentParser(description='Computes the subject-specific linear template.')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--max_iter', type=int, default=20, help='LBFGS')
    parser.add_argument('--n_epochs', type=int, default=30, help='Mask dilation factor')
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--num_cores', default=1, type=int)
    parser.add_argument('--force', action='store_true')
    
    arguments = parser.parse_args()
    bids_dir = arguments.bids
    cost = arguments.cost
    lr = arguments.lr
    max_iter = arguments.max_iter
    n_epochs = arguments.n_epochs
    initial_subject_list = arguments.subjects
    num_cores = arguments.num_cores
    force_flag = arguments.force

    print('Loading dataset. \n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['uslr-lin'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if arguments.subjects is None else arguments.subjects


    failed_subjects = []
    if num_cores > 1:
        VERBOSE = False
        results = Parallel(n_jobs=num_cores)(
            delayed(compute_subject_template)(subject, DIR_PIPELINES['uslr-lin']) for subject in subject_list)
    else:
        VERBOSE = True
        for it_subject, subject in enumerate(subject_list):
            print('\nSubject: ' + subject)
            try:
                fs = compute_subject_template(subject, DIR_PIPELINES['uslr-lin'])
                if fs is not None:
                    failed_subjects.append(fs)
            except:
                failed_subjects.append(subject)


    f = open(join(LOGS_DIR, 'compute_template_linear.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) +
          '. See ' + join(LOGS_DIR, 'compute_template_linear.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (USLR-LIN: compute latent template) --------- #')
    print('\n')



