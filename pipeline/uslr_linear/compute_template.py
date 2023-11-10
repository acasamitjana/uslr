import pdb
from argparse import ArgumentParser
from joblib import delayed, Parallel
import bids

# project imports
from setup import *
# from src import bids_loader
from src.slr import *


print('\n\n\n\n\n')
print('# ----------------------------- #')
print('# Linear SLR: compute template  #')
print('# ----------------------------- #')
print('\n\n')

#####################
# Global parameters #
#####################

# Input parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
arg_parser.add_argument('--max_iter', type=int, default=20, help='LBFGS')
arg_parser.add_argument('--n_epochs', type=int, default=30, help='Mask dilation factor')
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--num_cores', default=1, type=int)
arg_parser.add_argument('--force', action='store_true')

arguments = arg_parser.parse_args()
bidsdir = arguments.bids
cost = arguments.cost
lr = arguments.lr
max_iter = arguments.max_iter
n_epochs = arguments.n_epochs
initial_subject_list = arguments.subjects
num_cores = arguments.num_cores
force_flag = arguments.force

if bidsdir[-1] == '/': bidsdir = bidsdir[:-1]
seg_dir = os.path.join(os.path.dirname(bidsdir), 'derivatives', 'synthseg')
slr_dir = '/media/biofisica/BIG_DATA/ADNI-T1' # os.path.dirname(bidsdir)
slr_lin_dir = os.path.join(slr_dir, 'derivatives', 'slr-lin')

##############
# Processing #
##############
print('Loading dataset. \n')
db_file = join(os.path.dirname(BIDS_DIR), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False, database_path=db_file)

bids_loader.add_derivatives(seg_dir)
bids_loader.add_derivatives(slr_lin_dir)
subject_list = bids_loader.get_subjects() if initial_subject_list is None else initial_subject_list

anat_modalities = ['T1w']

def compute_subject_template(subject, slr_lin_dir, verbose=True):
    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x: len(bids_loader.get(extension='nii.gz', subject=subject, session=x, suffix='T1w')) > 0, timepoints))
    timepoints = list(filter(lambda x: not exists(join(dirname(bids_loader.get(scope='synthseg',subject=subject, session=x, return_type='filename')[0]), 'excluded_file.txt')), timepoints))


    print('Subject: ' + subject)
    image_dict = {'scope': 'synthseg', 'extension': 'nii.gz', 'subject': subject}
    aff_dict = {'subject': subject, 'desc': 'aff', 'scope': 'slr-lin', 'extension': 'npy'}
    if 'T1w' in bids_loader.get(**{**aff_dict, 'return_type': 'id', 'target': 'suffix'}):
        aff_dict['suffix'] = 'T1w'

    seg_dict = {'subject': subject, 'scope': 'synthseg', 'extension': 'nii.gz'}
    if 'T1wdseg' in bids_loader.get(**{**seg_dict, 'return_type': 'id', 'target': 'suffix'}):
        seg_dict['suffix'] = 'T1wdseg'
    else:
        seg_dict['suffix'] = 'dseg'

    seg_files = bids_loader.get(**seg_dict)
    if len(seg_files) <= 1:
        print('  It has only 0/1 timepoints. No registration is made.')
        return None

    dir_results = join(slr_lin_dir, 'sub-' + subject)

    filename_template = 'sub-' + subject + '_desc-linTemplate_T1w'
    linear_template = join(dir_results, filename_template + '.nii.gz')
    linear_template_mask = join(dir_results, filename_template + 'mask.nii.gz')
    linear_template_seg = join(dir_results, filename_template + 'dseg.nii.gz')

    if len(timepoints) == 0:
        print('  Skipping. No modalities found.')
        return

    elif len(timepoints) == 1 and seg_files[0].entities['dataype'] == 'anat':
        ent_res = {k: str(v) for k, v in seg_files[0].entities.items() if k in filename_entities}
        im_mask = bids_loader.get(scope='synthseg', **ent_res)
        if len(im_mask) != 1: print('  Search mask command failed. Number of matched files: ' + str(len(im_mask)))
        else: im_mask = im_mask[0]

        ent_res['suffix'] = ent_res['suffix'].replace('mask')
        im_res = bids_loader.get(scope='synthseg', **ent_res)
        if len(im_res) != 1: print('  Search image command failed. Number of matched files: ' + str(len(im_res)))
        else: im_res = im_res[0]

        if not exists(join(dir_results, 'anat', im_res.filename)):
            subprocess.call(['ln', '-s', im_res.path, join(dir_results, 'anat', im_res.filename)])

        if not exists(join(dir_results, 'anat', im_mask.filename)):
            subprocess.call(['ln', '-s', im_mask.path, join(dir_results, 'anat', im_mask.filename)])

        if not exists(linear_template):
            subprocess.call(['ln', '-s', join(dir_results, 'anat', im_res.filename), linear_template])

        print('   It has only 1 modality. No registration is made.')
        return

    elif len(bids_loader.get(**aff_dict)) != len(seg_files):
        if verbose: print('   !!! WARNING -- No observations found for subject ' + subject + '.')
        return


    if not exists(linear_template) or force_flag:

        t_init = time.time()
        if verbose: print('   - [Deforming] Updating vox2ras0  ... ')

        headers = {}
        headers_orig = {}
        linear_mask_list = {}
        for it_tp, tp in enumerate(timepoints):
            image_tp_dict = {**image_dict, 'session': tp}
            affine_file = bids_loader.get(**{**aff_dict, 'session': tp})
            if len(affine_file) != 1:
                print('Wrong affine file entities')
                # pdb.set_trace()
                continue

            affine_matrix = np.load(affine_file[0])

            # Update image header
            im_file = [im for im in bids_loader.get(**{**image_tp_dict, 'suffix':'T1w'}) if 'acq-orig' in im.filename]
            if len(im_file) != 1:
                print('     !! WARNING: More than one image is found in the synthseg directory ', end=' ')
                print(str([m.filename for m in im_file]), flush=True)
                continue
            else:
                im_file = im_file[0]

            proxyim = nib.load(im_file.path)
            v2r_sbj_orig = np.linalg.inv(affine_matrix) @ proxyim.affine
            headers_orig[tp] = v2r_sbj_orig

            # Update mask res header
            mask_file = bids_loader.get(**{**image_tp_dict, 'suffix':'T1wmask'})
            if len(mask_file) != 1:
                print('     !! WARNING: More than one mask is found in the synthseg directory ', end=' ')
                print(str([m.filename for m in mask_file]), flush=True)
                continue
            else:
                mask_file = mask_file[0]

            proxymask = nib.load(mask_file.path)
            mask = np.array(proxymask.dataobj)
            mask_dilated = binary_dilation(mask, ball(3)).astype('uint8')
            img = nib.Nifti1Image(mask_dilated, np.linalg.inv(affine_matrix) @ proxymask.affine)
            linear_mask_list[tp] = img
            headers[tp] = np.linalg.inv(affine_matrix) @ proxymask.affine

        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #

        if verbose: print('   - [Deforming] Creating subject space  ... ')

        rasMosaic, template_vox2ras0, template_size = create_template_space(list(linear_mask_list.values()))
        proxytemplate = nib.Nifti1Image(np.zeros(template_size), template_vox2ras0)

        if verbose: print('   - [Deforming] Computing linear template ... ')
        mri_list = []
        mask_list = []
        aparc_aseg = np.concatenate((APARC_ARR, ASEG_ARR), axis=0)
        seg_list = np.zeros(template_size + (len(aparc_aseg),))
        for it_tp, tp in enumerate(timepoints):
            image_tp_dict = {**image_dict, 'session': tp}
            im_file = bids_loader.get(**{**image_tp_dict, 'suffix':'T1w'})
            im_file = [im for im in im_file if 'acq-orig' in im.filename or 'acq' not in im.filename]

            if len(im_file) != 1:
                print('     !! WARNING: More than one image is found in the synthseg directory ', end=' ')
                print(str([m.filename for m in im_file]), flush=True)
                continue
            else:
                im_file = im_file[0]

            mask_file = bids_loader.get(**{**image_tp_dict, 'suffix':'T1wmask'})
            if len(mask_file) != 1:
                # pdb.set_trace()
                print('     !! WARNING: More than one mask is found in the synthseg directory ', end=' ')
                print(str([m.filename for m in mask_file]), flush=True)
                continue
            else:
                mask_file = mask_file[0]

            seg_file = bids_loader.get(**{**image_tp_dict, 'suffix':'T1wdseg'})
            if len(seg_file) != 1:
                # pdb.set_trace()
                print('     !! WARNING: More than one segmentation is found in the synthseg directory ', end=' ')
                print(str([m.filename for m in seg_file]), flush=True)
                continue
            else:
                seg_file = seg_file[0]

            ent_im = {k: str(v) for k, v in im_file.entities.items() if k in filename_entities}
            ent_mask = {k: str(v) for k, v in mask_file.entities.items() if k in filename_entities}
            ent_im['space'] = 'SUBJECT'
            ent_mask['space'] = 'SUBJECT'
            ent_im['acquisition'] = '1'

            im_filename = bids_loader.build_path({**ent_im, 'extension': 'nii.gz'}, scope='slr-lin',
                                                  path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                  absolute_paths=False)

            mask_filename = bids_loader.build_path({**ent_mask, 'extension': 'nii.gz'}, scope='slr-lin',
                                                   path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                   absolute_paths=False)

            proxyim = nib.load(im_file.path)
            pixdim = np.sqrt(np.sum(proxyim.affine * proxyim.affine, axis=0))[:-1]
            new_vox_size = np.array([1, 1, 1])
            factor = pixdim / new_vox_size
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            im_orig_array = np.array(proxyim.dataobj)
            if len(im_orig_array.shape) > 3:
                im_orig_array = im_orig_array[..., 0]
            volume_filt = gaussian_filter(im_orig_array, sigmas)

            im_orig_mri = nib.Nifti1Image(volume_filt, headers_orig[tp])
            im_mri = vol_resample(proxytemplate, im_orig_mri)
            nib.save(im_mri, join(slr_lin_dir, im_filename))
            mri_list.append(np.array(im_mri.dataobj))

            mask = np.array(nib.load(mask_file.path).dataobj)
            proxyflo = nib.Nifti1Image(mask, headers[tp])
            im_mri = vol_resample(proxytemplate, proxyflo)
            nib.save(im_mri, join(slr_lin_dir, mask_filename))
            mask_list.append(np.array(im_mri.dataobj))

            seg = np.array(nib.load(seg_file.path).dataobj)
            proxyflo = nib.Nifti1Image(seg, headers[tp])
            im_mri = vol_resample(proxytemplate, proxyflo, mode='nearest')
            seg_list += one_hot_encoding(np.array(im_mri.dataobj), categories=aparc_aseg)

        template = np.median(mri_list, axis=0)
        template = template.astype('uint8')
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, linear_template)

        template = np.sum(mask_list, axis=0)/len(mask_list) > 0.5
        template = template.astype('uint8')
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, linear_template_mask)

        seg_hard = np.argmax(seg_list, axis=-1)
        template = np.zeros(template_size, dtype='int16')
        for it_l, l in enumerate(aparc_aseg): template[seg_hard == it_l] = l
        img = nib.Nifti1Image(template, template_vox2ras0)
        nib.save(img, linear_template_seg)

        linear_etiv = linear_template_mask.replace('T1wmask', 'etiv')
        linear_etiv = linear_etiv.replace('nii.gz', 'npy')
        np.save(linear_etiv, np.sum(template > 0))

        if verbose: print('   - [Deforming] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

if num_cores > 1:
    VERBOSE = False
    results = Parallel(n_jobs=num_cores)(delayed(compute_subject_template)(subject, verbose=True)
                                         for subject in subject_list)
else:
    VERBOSE = True
    for it_subject, subject in enumerate(subject_list):

        try:
            compute_subject_template(subject, slr_lin_dir, verbose=True)
        except:
            continue




