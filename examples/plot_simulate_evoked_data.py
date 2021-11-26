"""
==============================
Generate simulated evoked data
==============================

Use :func:`~mne.simulation.simulate_sparse_stc` to simulate evoked data.
"""
# Author: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates
from mne.simulation import simulate_sparse_stc, simulate_evoked

from plot_meg_example import apply_solver

print(__doc__)

mne.viz.set_3d_backend("pyvistaqt")
###############################################################################
# Load real data as templates
data_path = sample.data_path()

raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = mne.read_proj(data_path + '/MEG/sample/sample_audvis_ecg-proj.fif')
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(ave_fname)

label_names = ['Aud-lh', 'Aud-rh', 'Vis-rh']
# label_names = ['Aud-lh', 'Aud-rh']
labels = [mne.read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
          for ln in label_names]

###############################################################################
# Generate source time courses from 2 dipoles and the correspond evoked data

times = np.arange(200, dtype=np.float) / raw.info['sfreq'] - 0.1

rng = np.random.RandomState(42)


def data_fun(times):

    """Function to generate random source time courses"""
    return (50e-9 * np.sin(30. * times) *
            np.exp(- (times - 0.15 + 0.05 * rng.randn(1)) ** 2 / 0.01))


stc = simulate_sparse_stc(fwd['src'], n_dipoles=len(labels), times=times,
                          random_state=rng, labels=labels, data_fun=data_fun)

###############################################################################
# Generate noisy evoked data
picks = mne.pick_types(raw.info, eeg=False, meg=True, exclude='bads')
iir_filter = None  # do not add autocorrelated noise
nave = 200  # simulate average of 100 epochs
evoked = simulate_evoked(fwd, stc, info, cov, nave=nave, use_cps=True,
                         iir_filter=iir_filter, random_state=66)

###############################################################################
# Plot

# plt.figure()
# plt.psd(evoked.data[0])
# plt.show(block=False)

# evoked.plot(time_unit='s')


stc_wlasso, monitor = apply_solver(
    evoked, fwd, cov, p_alpha0=0.5, model="wlasso")

stc_lasso, monitor_lasso = apply_solver(
    evoked, fwd, cov, p_alpha0=0.5, model="lasso")

print(evoked.data.sum())

plot_sparse_source_estimates(fwd['src'], stc_wlasso, bgcolor=(1, 1, 1),
                             opacity=0.1, high_resolution=True)

plot_sparse_source_estimates(fwd['src'], stc_lasso, bgcolor=(1, 1, 1),
                             opacity=0.1, high_resolution=True)

plot_sparse_source_estimates(fwd['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.1, high_resolution=True)
