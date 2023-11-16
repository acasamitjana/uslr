from os.path import join, exists, basename
from os import  makedirs
import itertools
from datetime import date, datetime
import time
import subprocess
from joblib import delayed, Parallel

import json
import torch
from torch import nn
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linprog
import nibabel as nib

from setup import *
from src.callbacks import History, ModelCheckpoint, PrinterCallback
from src.models import InstanceRigidModelLOG
from utils.io_utils import write_affine_matrix
from utils.def_utils import vol_resample


# Read linear st2 graph
# Formulas extracted from: https://math.stackexchange.com/questions/3031999/proof-of-logarithm-map-formulae-from-so3-to-mathfrak-so3
def init_st2_lineal(timepoints, input_dir, eps=1e-6):
    nk = 0

    N = len(timepoints)
    K = int(N * (N - 1) / 2)

    phi_log = np.zeros((6, K))

    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
        if not isinstance(tp_ref, str):
            tid_ref, tid_flo = tp_ref.id, tp_flo.id
        else:
            tid_ref, tid_flo = tp_ref, tp_flo


        filename = str(tid_ref) + '_to_' + str(tid_flo)

        rigid_matrix = np.load(join(input_dir, filename + '.npy'))
        rotation_matrix, translation_vector = rigid_matrix[:3, :3], rigid_matrix[:3, 3]

        # Log(R) and Log(T)
        t_norm = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1 + eps, 1 - eps)) + eps
        W = 1 / (2 * np.sin(t_norm)) * (rotation_matrix - rotation_matrix.T) * t_norm
        Vinv = np.eye(3) - 0.5 * W + ((1 - (t_norm * np.cos(t_norm / 2)) / ( 2 * np.sin(t_norm / 2))) / t_norm ** 2) * W * W  # np.matmul(W, W)

        phi_log[0, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * t_norm
        phi_log[1, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * t_norm
        phi_log[2, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * t_norm

        phi_log[3:, nk] = np.matmul(Vinv, translation_vector)

        nk += 1

    return phi_log


# Read st2 graph
def init_st2(timepoints, input_dir, image_shape, factor=1, scope='uslr-lin', se=None, penalty=1):


    timepoints_dict = {
       t.id: it_t for it_t, t in enumerate(timepoints)
    } # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)

    nk = 0

    N = len(timepoints)
    K = int(N*(N-1)/2) +1

    w = np.zeros((K, N), dtype='int')
    obs_mask = np.zeros(image_shape + (K,))

    phi = np.zeros((3,) + image_shape + (K, ))
    for sl_ref, sl_flo in itertools.combinations(timepoints, 2):
        sid_ref, sid_flo = sl_ref.id, sl_flo.id

        t0 = timepoints_dict[sl_ref.id]
        t1 = timepoints_dict[sl_flo.id]
        filename = str(sid_ref) + '_to_' + str(sid_flo)

        svf_proxy = nib.load(join(input_dir, filename + '.svf.nii.gz'))
        field = np.asarray(svf_proxy.dataobj)

        if field.shape[0] != 3: field = np.transpose(field, axes=(3, 0, 1, 2))

        phi[..., nk] = factor*field

        # Masks
        if se is not None:
            cond_mask = {'suffix': 'mask', 'acquisition': '1', 'scope': scope, 'space': 'SUBJECT', 'extension': '.nii.gz'}

            mask_proxy = nib.load(join(sl_ref.data_dir[scope], sl_ref.get_files(**cond_mask)[1]))
            mask_proxy = vol_resample(svf_proxy, mask_proxy)

            mask = np.asarray(mask_proxy.dataobj) > 0
            mask = binary_dilation(mask, se)

        else:
            mask = np.ones(image_shape)


        obs_mask[..., nk] = mask
        w[nk, t0] = -1
        w[nk, t1] = 1
        nk += 1

    obs_mask[..., nk] = (np.sum(obs_mask[..., :nk-1]) > 0).astype('uint8')
    w[nk, :] = penalty
    nk += 1
    return phi, obs_mask, w, nk


# Optimization of rigid transforms
def st_linear(bids_loader, subject, cost, lr, max_iter, n_epochs, slr_lin_dir, force_flag=False, verbose=False):
    print('Subject: ' + str(subject))

    timepoints = bids_loader.get_session(subject=subject)
    timepoints = list(filter(lambda x: len(bids_loader.get(extension='nii.gz', subject=subject, session=x, suffix='T1w')) > 0, timepoints))

    if len(timepoints) <= 1:
        print('[done] It has only ' + str(len(timepoints)) + '. Skipping.')
        return

    dir_results = join(slr_lin_dir, 'sub-' + subject)
    if not exists(dir_results): makedirs(dir_results)

    date_start = date.today().strftime("%d/%m/%Y")
    time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    exp_dict = {
        'date': date_start,
        'time': time_start,
        'cost': cost,
        'lr': lr,
        'max_iter': max_iter,
        'n_epochs': n_epochs
    }

    filename_template = 'sub-' + subject + '_desc-linTemplate_anat'
    if not exists(join(dir_results, filename_template + '_T1w.json')):
        json_object = json.dumps(exp_dict, indent=4)
        with open(join(dir_results, filename_template + '_T1w.json'), "w") as outfile:
            outfile.write(json_object)

    deformations_dir = join(dir_results, 'deformations')

    # Deformations dir
    if not exists(join(deformations_dir, timepoints[-2] + '_to_' + timepoints[-1] + '.npy')):
        if verbose: print('[error] No observations found for subject ' + subject + ' and Procrustes Analysis. Skipping.')
        return [subject]

    # Check if multiple runs in this dataset.
    aff_dict = {'subject': subject, 'desc': 'aff', 'suffix': 'T1w'}
    if not len(bids_loader.get(subject=subject,  extension='npy', desc='aff', scope='uslr-lin')) == len(timepoints) or force_flag:

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#

        if verbose: print('  [' + str(subject) + ' - Building the graph] Reading transforms ...')
        t_init = time.time()

        graph_structure = init_st2_lineal(timepoints, deformations_dir)
        R_log = graph_structure

        if verbose: print('  [' + str(subject) + ' - Building the graph] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

        if verbose: print('  [' + str(subject) + ' - USLR] Computimg the graph ...')
        t_init = time.time()

        Tres = st2_lineal_pytorch(R_log, timepoints, n_epochs, cost, lr, dir_results, max_iter=max_iter, verbose=False)

        if verbose: print('  [' + str(subject) + ' - USLR] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#

        if verbose: print('  [' + str(subject) + ' - Integration] Computing the latent rigid transform ... ')
        t_init = time.time()
        for it_tp, tp in enumerate(timepoints):

            extra_kwargs = {'session': tp}
            filename_npy = bids_loader.build_path({**extra_kwargs, **aff_dict, 'extension': 'npy'}, scope='uslr-lin',
                                                  path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                  absolute_paths=False)
            filename_aff = bids_loader.build_path({**extra_kwargs, **aff_dict, 'extension': 'txt'}, scope='uslr-lin',
                                                  path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                  absolute_paths=False)

            affine_matrix = Tres[..., it_tp]

            dir_results_sess = join(os.path.dirname(dir_results), os.path.dirname(filename_npy))
            if not exists(dir_results_sess): makedirs(dir_results_sess)
            np.save(join(dir_results_sess, os.path.basename(filename_npy)), affine_matrix)
            write_affine_matrix(join(dir_results_sess, os.path.basename(filename_aff)), affine_matrix)

        if verbose: print('  [' + str(subject) + ' - Integration] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

    else:
        print('[done] Subject already processed. Skipping')

        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#


# Optimization of rigid transforms using pytorch
def st2_lineal_pytorch(logR, timepoints, n_epochs, cost, lr, results_dir_sbj, max_iter=5, patience=3,
                       device='cpu', verbose=True):

    if len(timepoints) > 2:
        log_keys = ['loss', 'time_duration (s)']
        logger = History(log_keys)
        model_checkpoint = ModelCheckpoint(join(results_dir_sbj, 'checkpoints'), -1)
        callbacks = [logger, model_checkpoint]
        if verbose: callbacks += [PrinterCallback()]

        model = InstanceRigidModelLOG(timepoints, cost=cost, device=device, reg_weight=0)
        optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

        min_loss = 1000
        iter_break = 0
        log_dict = {}
        logR = torch.FloatTensor(logR)
        for cb in callbacks:
            cb.on_train_init(model)

        for epoch in range(n_epochs):
            for cb in callbacks:
                cb.on_epoch_init(model, epoch)

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                loss = model(logR, timepoints)
                loss.backward()

                return loss

            optimizer.step(closure=closure)

            loss = model(logR, timepoints)

            if loss < min_loss + 1e-4:
                iter_break = 0
                min_loss = loss.item()

            else:
                iter_break += 1

            if iter_break > patience or loss.item() == 0.:
                break

            log_dict['loss'] = loss.item()

            for cb in callbacks:
                cb.on_step_fi(log_dict, model, epoch, iteration=1, N=1)

        T = model._compute_matrix().cpu().detach().numpy()

    else:
        logR = np.squeeze(logR.astype('float32'))
        model = InstanceRigidModelLOG(timepoints, cost=cost, device=device, reg_weight=0)
        model.angle = nn.Parameter(torch.tensor(np.array([[-logR[0]/2, logR[0]/2],
                                                          [-logR[1]/2, logR[1]/2],
                                                          [-logR[2]/2, logR[2]/2]])).float(), requires_grad=False)

        model.translation = nn.Parameter(torch.tensor(np.array([[-logR[3] / 2, logR[3] / 2],
                                                                [-logR[4] / 2, logR[4] / 2],
                                                                [-logR[5] / 2, logR[5] / 2]])).float(), requires_grad=False)
        T = model._compute_matrix().cpu().detach().numpy()

    return T


# Gaussian: l2-loss no masks
def st2_L2_global(phi, W, N):
    precision = 1e-6
    lambda_control = np.linalg.inv((W.T @ W) + precision * np.eye(N)) @ W.T
    Tres = lambda_control @ np.transpose(phi,[1, 2, 3, 4, 0])
    Tres = np.transpose(Tres, [4, 0, 1, 2, 3])

    return Tres

# Gaussian: l2-loss
def st2_L2(phi, obs_mask, w, N):

    image_shape = obs_mask.shape[:-1]

    #Initialize transforms
    Tres = np.zeros(phi.shape[:4] + (N,))

    print('    Computing weights and updating the transforms')
    precision = 1e-6
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0:
            print('       Row ' + str(it_control_row) + '/' + str(image_shape[0]))

        for it_control_col in range(image_shape[1]):
            # if np.mod(it_control_col, 10) == 0:
            #     print('           Col ' + str(it_control_col) + '/' + str(image_shape[1]))

            for it_control_depth in range(image_shape[2]):

                index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]
                if index_obs.shape[0] == 0:
                    Tres[:, it_control_row, it_control_col, it_control_depth] = 0
                else:
                    w_control = w[index_obs]
                    phi_control = phi[:,it_control_row, it_control_col, it_control_depth, index_obs]
                    lambda_control = np.linalg.inv(w_control.T @ (w_control + precision*np.eye(N))) @ w_control.T

                    for it_tf in range(N):
                        Tres[0, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[0].T
                        Tres[1, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[1].T
                        Tres[2, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[2].T

    return Tres


# Laplacian: l1-loss
def st2_L1(phi, obs_mask, w, N, chunk_id=None):

    if chunk_id is not None:
        print("Processing chunk " + str(chunk_id))

    image_shape = obs_mask.shape[:3]
    Tres = np.zeros(phi.shape[:4] + (N,))
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0 and chunk_id is None:
            print('    Row ' + str(it_control_row) + '/' + str(image_shape[0]))
        for it_control_col in range(image_shape[1]):
            for it_control_depth in range(image_shape[2]):
                index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]

                if index_obs.shape[0] > 0:
                    w_control = w[index_obs]
                    phi_control = phi[:, it_control_row, it_control_col, it_control_depth, index_obs]
                    n_control = len(index_obs)

                    for it_dim in range(3):
                        # Set objective
                        c_lp = np.concatenate((np.ones((n_control,)), np.zeros((N,))), axis=0)

                        # Set the inequality
                        A_lp = np.zeros((2 * n_control, n_control + N))
                        A_lp[:n_control, :n_control] = -np.eye(n_control)
                        A_lp[:n_control, n_control:] = -w_control
                        A_lp[n_control:, :n_control] = -np.eye(n_control)
                        A_lp[n_control:, n_control:] = w_control
                        # A_lp = sp.csr_matrix(A_lp)

                        reg = np.reshape(phi_control[it_dim], (n_control,))
                        b_lp = np.concatenate((-reg, reg), axis=0)

                        result = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=(None, None), method='highs-ds')
                        Tres[it_dim, it_control_row, it_control_col, it_control_depth] = result.x[n_control:]

    return Tres

def st2_L1_chunks(phi, obs_mask, w, N, num_cores=4):
    chunk_list = []
    nchunks = 2
    image_shape = obs_mask.shape[:3]
    chunk_size = [int(np.ceil(cs / nchunks)) for cs in image_shape]
    for x in range(nchunks):
        for y in range(nchunks):
            for z in range(nchunks):
                max_x = min((x + 1) * chunk_size[0], image_shape[0])
                max_y = min((y + 1) * chunk_size[1], image_shape[1])
                max_z = min((z + 1) * chunk_size[2], image_shape[2])
                chunk_list += [[[x * chunk_size[0], max_x],
                                [y * chunk_size[1], max_y],
                                [z * chunk_size[2], max_z]]]

    if num_cores == 1:
        Tres = st2_L1(phi, obs_mask, w, N)

    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(st2_L1)(
                phi[:, chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]],
                obs_mask[chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]],
                w, N, chunk_id=it_chunk) for it_chunk, chunk in enumerate(chunk_list))

        Tres = np.zeros(phi.shape[:4] + (N,))
        for it_chunk, chunk in enumerate(chunk_list):
            Tres[:, chunk[0][0]: chunk[0][1], chunk[1][0]: chunk[1][1], chunk[2][0]: chunk[2][1]] = results[it_chunk]

    return Tres