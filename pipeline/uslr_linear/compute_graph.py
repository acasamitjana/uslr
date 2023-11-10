from argparse import ArgumentParser
from joblib import delayed, Parallel
import bids

# project imports
from setup import *
from src import bids_loader
from src.slr import *


print('\n\n\n\n\n')
print('# -------------------------- #')
print('# Linear SLR: compute graph  #')
print('# -------------------------- #')
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
db_file = join(os.path.dirname(bidsdir), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=bidsdir, validate=False, database_path=db_file)

bids_loader.add_derivatives(seg_dir)
bids_loader.add_derivatives(slr_lin_dir)
subject_list = bids_loader.get_subjects() if initial_subject_list is None else initial_subject_list

if num_cores > 1:
    VERBOSE = False
    results = Parallel(n_jobs=num_cores)(
        delayed(st_linear_bids)(bids_loader, subject, cost, lr, max_iter, n_epochs,slr_lin_dir,
                                force_flag=force_flag) for subject in subject_list
    )

else:
    VERBOSE = True
    for it_subject, subject in enumerate(subject_list):
        st_linear_bids(bids_loader, subject, cost, lr, max_iter, n_epochs, slr_lin_dir,
                       force_flag=force_flag, verbose=VERBOSE)




