"""
==================================
Weighted Lasso versus Lasso on MEG
==================================

This example compares the Lasso and the weighted Lasso on real MEG data.
While the bias of the Lasso leads to optimal coefficients with a lot of
sources in the brain, the weighted Lasso is able to recover 1 source per
hemisphere in the brain, as expected from a neuroscience point of view.
"""

# Authors: Mathurin Massias <mathurin.massas@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 4

import mne
from mne.viz import plot_sparse_source_estimates
from mne.datasets import sample

from meg_utils import apply_solver


data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'
condition = 'Left Auditory'

if condition == 'Left Auditory':
    tmax = 0.18
else:
    tmax = 0.15

# %%
# Read noise covariance matrix and evoked data
noise_cov = mne.read_cov(cov_fname)
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0.04, tmax=0.18)

# Crop data around the period of interest
evoked = evoked.pick_types(eeg=False, meg=True)

# %%
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

loose, depth = 0., .8  # corresponds to free orientation

# %%
# Run estimation with Lasso
stc = apply_solver(
    evoked, forward, noise_cov, loose, depth, model_name="lasso")[0]
# Plot glass brain
plot_sparse_source_estimates(
    forward['src'], stc, bgcolor=(1, 1, 1), opacity=0.1)

# %%
# Run estimation with Weighted Lasso
stc = apply_solver(
    evoked, forward, noise_cov, loose, depth, model_name="wlasso")[0]
# Plot glass brain
plot_sparse_source_estimates(
    forward['src'], stc, bgcolor=(1, 1, 1), opacity=0.1)
