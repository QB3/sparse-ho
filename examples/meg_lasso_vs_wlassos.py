import mne

from surfer import Brain
from mayavi import mlab

from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates

from plot_meg_example import apply_solver


def plot_blob(
        stc, subject="sample", surface="inflated", s=14, save_fname="",
        data_path=sample.data_path(), subject_name='/subjects',
        fig_dir="", figsize=(800, 800), event_id=1):

    subjects_dir = data_path + subject_name
    list_hemi = ["lh", "rh"]

    for i, hemi in enumerate(list_hemi):
        figure = mlab.figure(size=figsize)
        brain = Brain(
            subject, hemi, surface, subjects_dir=subjects_dir,
            offscreen=False, figure=figure)
        surf = brain.geo[hemi]
        sources_h = stc.vertices[i]  # 0 for lh, 1 for rh
        for sources in sources_h:
            mlab.points3d(
                surf.x[sources], surf.y[sources],
                surf.z[sources], color=(1, 0, 0),
                scale_factor=s, opacity=1., transparent=True)
        if save_fname:
            fname = fig_dir + hemi + save_fname
            print(fname)
            if event_id == 1 or event_id == 2:
                brain.save_montage(fname, order=['lat'])
            else:
                brain.save_montage(fname, order=['lat'])
                # brain.save_montage(fname, order=['ven'])

            # mlab.savefig(fname)
            figure = mlab.gcf()
            mlab.close(figure)


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
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0.04, tmax=0.18)

evoked = evoked.pick_types(eeg=False, meg=True)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

loose, depth = 0., .8  # corresponds to free orientation

models = ["lasso", "wlasso"]
for model in models:
    stc, monitor = apply_solver(
        evoked, forward, noise_cov, loose, depth, model=model)
    print("Value of objectives:")
    print(monitor.objs)

    fig_dir = "../../CD_SUGAR/tex/slides_qbe_long/prebuiltimages/"
    save_fname = "%s.png" % model

    plot_blob(stc, fig_dir=fig_dir, save_fname=save_fname)

    ###########################################################################
    # View in 2D and 3D ("glass" brain like 3D plot)
    # plot_sparse_source_estimates(
    #     forward['src'], stc, bgcolor=(1, 1, 1), opacity=0.1)
