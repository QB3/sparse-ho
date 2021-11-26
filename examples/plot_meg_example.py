"""
=======================================
Finding MEG sources with weighted Lasso
=======================================

This example compares the Lasso and Weighted Lasso on real MEG data.
"""

import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates

from meg_utils import apply_solver

mne.viz.set_3d_backend("pyvistaqt")

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'
# condition = 'Right Auditory'
condition = 'Left Auditory'

if condition == 'Left Auditory':
    tmax = 0.18
else:
    tmax = 0.15

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)
# Handling average file
evoked = mne.read_evokeds(
    ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0.04, tmax=0.18)

evoked = evoked.pick_types(eeg=False, meg=True)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

loose, depth = 0., .8  # corresponds to free orientation
stc_wlasso, monitor = apply_solver(
    evoked, forward, noise_cov, loose, depth, model_name="wlasso")
print("Value of objectives:")
print(monitor.objs)
###########################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
plot_sparse_source_estimates(
    forward['src'], stc_wlasso, bgcolor=(1, 1, 1), opacity=0.1)
