from setup import *

from os.path import dirname, join, exists
import time
from argparse import ArgumentParser
from joblib import delayed, Parallel

import nibabel as nib
import numpy as np
import bids

# project imports
from src.uslr import *



if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# --------------------------- #')
    print('# Linear USLR: compute graph  #')
    print('# --------------------------- #')
    print('\n\n')

    # Input parameters
    parser = ArgumentParser(description='Runs the linear longitudinal registration algorithm to compute the latent'
                                        ' transforms, i.e., solve the spanning tree')
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
        results = Parallel(n_jobs=num_cores)(
            delayed(st_linear)(bids_loader, subject, cost, lr, max_iter, n_epochs, DIR_PIPELINES['uslr-lin'],
                               force_flag=force_flag) for subject in subject_list
        )

    else:
        for it_subject, subject in enumerate(subject_list):
            t_init = time.time()
            try:
                fs = st_linear(bids_loader, subject, cost, lr, max_iter, n_epochs, DIR_PIPELINES['uslr-lin'],
                               force_flag=force_flag, verbose=VERBOSE)
                if fs is not None:
                    failed_subjects.append(subject)
            except:
                failed_subjects.append(subject)
            print('Total computation time: ' + str(np.round(time.time() - t_init, 2)) + '\n')


    f = open(join(LOGS_DIR, 'compute_graph_linear.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) +
          '. See ' + join(LOGS_DIR, 'compute_graph_linear.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (USLR-LIN: compute latent transforms) --------- #')
    print('\n')
