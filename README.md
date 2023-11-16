# USLR: an open-source tool for unbiased and smooth longitudinal registration of brain MRI

This repository performs linear and nonlinear registration between a set of points (e.g., timepoints in longitudinal studies) and a shared latent space. We use the log-space of transforms to infere the most probable deformations using Bayesian inference


### Requirements:
**Python** <br />
The code run on python v3.8 and several external libraries listed under _requirements.txt_

**BIDS protocol** <br />
The pipeline works with datasets following the BIDS protocol. 

**Freesurfer installed**<br />
We use Synthseg, a learning-based functionality of freesurfer, for MRI segmentation. Make sure that the version of freesurfer has SynthSeg on it and that it is properly sourced.

**GPU (optional)**<br />
If a GPU is available, non-linear stream of the pipeline will run faster.

**Data**<br />
Data needs to be organised following the BIDS protocol. Important! Make sure that
if multiple T1w images are available, the difference is not in the _acquisition_
entity (it can be in other, most often _run_, but also _desc_, _space_, etc. ). 
### Run the code
- **Set-up configuration files** 
  - _setup.py_: create this file following the setup_example.py and according to your local machine and data directories. It contains the absolute paths to input data and the all the generated output registration paths. Here, one needs to set the rawdata directory as environmental variable:
     - BIDS_DIR: your path to the 'rawdata' directory of the BIDS protocol
- **Run pre-processing**
   - _scripts/run_preprocessing.sh_: this script will run over all subjects available in $BIDS_DIR. It also accepts a list of arguments (SUBJECT_ID) to run it over a subset (1, ..., N) subjects. It performs anatomical segmentation using and intensity inhomogeneity correction. The output will be stored in $BIDS_DIR/../derivatives/synthseg
- **Run linear registration**
  - _scripts/run_linear_registration.sh_: this script will run over all subjects available in $BIDS_DIR/../derivatives/synthseg. It also accepts a list of arguments (SUBJECT_ID) to run it over a subset of (1, ..., N) subjects. The output will be stored in $BIDS_DIR/../derivatives/usrl-lin
- **Run non-linear registration**
  - _scripts/run_nonlinear_registration.sh_: this script will run over all subjects available in $BIDS_DIR/../derivatives/usrl-lin (subjects processed using the linear registration script). It also accepts a list of arguments (SUBJECT_ID) to run it over a subset (1, ..., N) subjects. The output will be stored in $BIDS_DIR/../derivatives/usrl-nonlin


## Code updates

10 November 2023:
- Initial commit and README file.



## Citation
TBC



