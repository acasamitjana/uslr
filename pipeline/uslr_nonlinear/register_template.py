import pdb

from setup import *

import time
import warnings
from os.path import exists, join, dirname
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
from tensorflow import keras
import bids

from utils.fn_utils import compute_centroids_ras
from utils import synthmorph_utils
from utils.synthmorph_utils import VxmDenseOriginalSynthmorph, instance_register, path_model_registration, \
    RescaleTransform, VecInt, fast_3D_interp_torch, fast_3D_interp_field_torch

warnings.filterwarnings("ignore")

def process_subject(subject, bids_loader, args, proxytemplate, proxyatlas, tmp_dir='/tmp'):
    uslr_nonlin_dir_sbj = join(DIR_PIPELINES['uslr-nonlin'], 'sub-' + subject)
    uslr_mni_dir_sbj = join(DIR_PIPELINES['subject-mni'], 'sub-' + subject)
    if not exists(uslr_mni_dir_sbj): os.makedirs(uslr_mni_dir_sbj)

    svf_fname_suffix = '_space-' + args.template + '_desc-field_svf.nii.gz'
    def_fname_suffix = '_space-' + args.template + '_desc-field_def.nii.gz'
    timepoints = bids_loader.get_session(subject=subject)

    if len(timepoints) == 1:
        print('[done] Subject has only one session. Skipping. ')
        return

    elif not exists(join(uslr_nonlin_dir_sbj, 'sub-' + subject + '_desc-atlas_aff.npy')):
        print('[error] Affine matrix to SynthMorph space not found. Please, compute it and re-run the script. Skipping')
        return subject

    template = join(uslr_nonlin_dir_sbj, 'sub-' + subject + '_desc-nonlinTemplate_T1w.nii.gz')
    template_mask = join(uslr_nonlin_dir_sbj, 'sub-' + subject + '_desc-nonlinTemplate_T1wmask.nii.gz')

    if not exists(template) or not exists(template_mask):
        print('[error] Latent non-linear template has not been computed for subject ' + subject + '. Skipping.')
        return subject

    svf_fpath = join(uslr_mni_dir_sbj, 'sub-' + subject + svf_fname_suffix)
    def_fpath = join(uslr_mni_dir_sbj, 'sub-' + subject + def_fname_suffix)
    reg_path = join(uslr_mni_dir_sbj, 'sub-' + subject + '_space-MNI_desc-nonlinTemplate_T1w.nii.gz')

    if not exists(svf_fpath) or not exists(def_fpath) or args.force:
        if args.template.lower() == 'SynthMorph':
            M_ref = np.eye(4)
            R_aff = np.eye(4)

            Msbj = np.load(join(uslr_nonlin_dir_sbj, 'sub-' + subject + '_desc-atlas_aff.npy'))
            Rlin = torch.tensor(np.array(proxyatlas.dataobj))
            Rlin = Rlin / torch.max(Rlin)
            F = np.array(nib.load(template).dataobj)
            Flin, F_aff, _ = synthmorph_utils.compute_atlas_alignment(template, template_mask, proxyatlas, Msbj)


        elif args.template == 'MNI':
            # Compute Msbj
            if not exists(MNI_to_ATLAS):
                centroid_sbj, ok = compute_centroids_ras(MNI_TEMPLATE_SEG, synthmorph_utils.labels_registration)
                centroid_atlas = np.load(synthmorph_utils.atlas_cog_file)
                M_ref = synthmorph_utils.getM(centroid_atlas[:, ok > 0], centroid_sbj[:, ok > 0], use_L1=False)
                np.save(MNI_to_ATLAS, M_ref)

            else:
                M_ref = np.load(MNI_to_ATLAS)

            Msbj = np.load(join(uslr_nonlin_dir_sbj, 'sub-' + subject + '_desc-atlas_aff.npy'))

            F = np.array(nib.load(template).dataobj)
            Rlin, R_aff, _ = synthmorph_utils.compute_atlas_alignment(ref_template, ref_template_mask, proxyatlas, M_ref)
            Flin, F_aff, _ = synthmorph_utils.compute_atlas_alignment(template, template_mask, proxyatlas, Msbj)

        else:
            print(args.template + " template still not implemented. Skipping")
            return [subject]

        print('  > Computing svf field; ', end='', flush=True)
        cnn = VxmDenseOriginalSynthmorph.load(path_model_registration)
        svf1 = cnn.register(Flin.detach().numpy()[np.newaxis, ..., np.newaxis],
                            Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])
        if args.nosym:
            svf = svf1
        else:
            svf2 = cnn.register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                Flin.detach().numpy()[np.newaxis, ..., np.newaxis])
            svf = 0.5 * svf1 - 0.5 * svf2

        if args.nepochs_refinement > 0:
            instance_model = instance_register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                               Flin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                               svf, inshape=proxyatlas.shape, epochs=args.nepochs_refinement)
            svf_refined = instance_model.references.flow_layer(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])
        else:
            svf_refined = svf

        upscaler = keras.Sequential([RescaleTransform(2)])
        svf_final = upscaler(svf_refined)

        img = nib.Nifti1Image(np.squeeze(svf_final), proxyatlas.affine)
        nib.save(img, svf_fpath)

        print('integrating and computing deformation field (in mm); ', end='', flush=True)
        integrator = keras.Sequential([VecInt(method='ss', int_steps=7)])
        f2r_field = torch.tensor(np.squeeze(integrator(svf_final)))

        II, JJ, KK = np.meshgrid(np.arange(proxytemplate.shape[0]), np.arange(proxytemplate.shape[1]),
                                 np.arange(proxytemplate.shape[2]), indexing='ij')
        II = torch.tensor(II, device='cpu')
        JJ = torch.tensor(JJ, device='cpu')
        KK = torch.tensor(KK, device='cpu')
        #
        if args.template != 'SynthMorph':
            affine = torch.tensor(np.matmul(np.linalg.inv(atlas_aff), np.matmul(np.linalg.inv(M_ref), R_aff)),
                                  device='cpu')
            II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
            JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
            KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
            FIELD = fast_3D_interp_field_torch(f2r_field, II2, JJ2, KK2)

        else:
            II2, JJ2, KK2 = II, JJ, KK
            FIELD = f2r_field
        #
        II3 = II2 + FIELD[:, :, :, 0]
        JJ3 = JJ2 + FIELD[:, :, :, 1]
        KK3 = KK2 + FIELD[:, :, :, 2]
        #
        affine = torch.tensor(Msbj @ atlas_aff, device='cpu')
        RAS_X = affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
        RAS_Y = affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
        RAS_Z = affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]
        #
        img = nib.Nifti1Image(torch.stack([RAS_X, RAS_Y, RAS_Z], axis=-1).numpy(), R_aff)
        nib.save(img, def_fpath)
        #
        if args.save_registration:
            print('deforming floating image; ', end='', flush=True)
            affine = torch.tensor(np.linalg.inv(F_aff), device='cpu')
            II4 = affine[0, 0] * RAS_X + affine[0, 1] * RAS_Y + affine[0, 2] * RAS_Z + affine[0, 3]
            JJ4 = affine[1, 0] * RAS_X + affine[1, 1] * RAS_Y + affine[1, 2] * RAS_Z + affine[1, 3]
            KK4 = affine[2, 0] * RAS_X + affine[2, 1] * RAS_Y + affine[2, 2] * RAS_Z + affine[2, 3]
            #
            registered = fast_3D_interp_torch(torch.tensor(F), II4, JJ4, KK4, 'linear')
            img = nib.Nifti1Image(np.squeeze(registered.numpy()), R_aff)
            nib.save(img, reg_path)
        print('done.')

    elif args.save_registration:
        print('deforming floating image; ', end='', flush=True)
        proxy = nib.load(def_fpath)
        RAS = np.array(proxy.dataobj)
        RAS_X, RAS_Y, RAS_Z = RAS[..., 0], RAS[..., 1], RAS[..., 2]

        F_proxy = nib.load(template)
        F_aff = F_proxy.affine
        F = np.array(F_proxy.dataobj)

        affine = torch.tensor(np.linalg.inv(F_aff), device='cpu')
        II4 = affine[0, 0] * RAS_X + affine[0, 1] * RAS_Y + affine[0, 2] * RAS_Z + affine[0, 3]
        JJ4 = affine[1, 0] * RAS_X + affine[1, 1] * RAS_Y + affine[1, 2] * RAS_Z + affine[1, 3]
        KK4 = affine[2, 0] * RAS_X + affine[2, 1] * RAS_Y + affine[2, 2] * RAS_Z + affine[2, 3]
        #
        registered = fast_3D_interp_torch(torch.tensor(F), II4, JJ4, KK4, 'linear')
        img = nib.Nifti1Image(np.squeeze(registered.numpy()), proxytemplate.affine)
        nib.save(img, reg_path)
        print('done.')
    else:
        print('[done] Deformation field already exists. Skipping.')

    return

if __name__ == '__main__':

    parser = ArgumentParser(description='Computes the registration of the subject-specific template to a '
                                        'standard space.')
    parser.add_argument('--bids',
                        default=BIDS_DIR,
                        help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects',
                        default=None,
                        nargs='+',
                        help="(optional, default=None) specify which subjects  to process")
    parser.add_argument('--template',
                        default='MNI',
                        choices=['MNI', 'SynthMorph'],
                        help="(optional, default=MNI) choose the template space")
    parser.add_argument('--nepochs_refinement',
                        default=0,
                        type=int,
                        help="(optional, default=0) number of epoch for pairwise registration refinement after "
                             "SynthMorph.")
    parser.add_argument('--save_registration',
                        action='store_true',
                        help="(optional, default=False) save the image registered to the template.")
    parser.add_argument('--force',
                        action='store_true',
                        help="(optional, default=False) force to compute registration even if it already exists.")
    parser.add_argument('--nosym',
                        action='store_true',
                        help="(optional, default=False) compute symmetric registration.")

    args = parser.parse_args()

    print('\n\n\n\n\n')
    print('# ------------------------------------------------' + '-' * len(args.template) + ' #')
    print('# Register Nonlinear Template to reference space: ' + args.template + ' #')
    print('# ------------------------------------------------' + '-' * len(args.template) + ' #')
    print('\n\n')

    if args.template == 'MNI':
        ref_template = MNI_TEMPLATE
        ref_template_mask = MNI_TEMPLATE_MASK

    elif args.template == 'SynthMorph':
        ref_template = synthmorph_utils.atlas_file
        ref_template_mask = synthmorph_utils.atlas_mask_file
        if not exists(ref_template_mask):
            proxy = nib.load(synthmorph_utils.atlas_seg_file)
            seg = np.array(proxy.dataobj)
            mask = seg > 0
            img = nib.Nifti1Image(mask.astype('uint8'), proxy.affine)
            nib.save(img, ref_template_mask)
    else:
        raise ValueError('[error] Please, specify a valid template name.')

    proxyatlas = nib.load(synthmorph_utils.atlas_file)
    atlas_aff = proxyatlas.affine
    proxytemplate = nib.load(ref_template)
    tmp_dir = '/tmp/uslr-register-template'
    if not exists(tmp_dir): os.makedirs(tmp_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = 'cpu'


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

    ####################
    # Run registration #
    ####################
    failed_subjects = []
    for it_subject, subject in enumerate(subject_list):
        print('* Subject: ' + subject + '. (' + str(it_subject) + '/' + str(len(subject_list)) + ').')
        t_init = time.time()
        # try:
        ms = process_subject(subject, bids_loader, args, proxytemplate, proxyatlas, tmp_dir=tmp_dir)
        print('  Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)) + ' seconds.')
        # except:
        #     ms = subject

        if ms is not None:
            failed_subjects.append(ms)

    f = open(join(LOGS_DIR, 'register_template.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) +
          '. See ' + join(LOGS_DIR, 'register_template.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (USLR-NONLIN: register_template) --------- #')
    print('\n')

