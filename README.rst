sparse-ho
=========

|image0| |image1|

sparse-ho stands for "sparse hyperparameter optimization". This code aims to offer
a efficient solution to the problem of hyperparameter setting for sparse models such as Lasso.

This package implements the implicit forward differentiation algorithm, a fast algorithm to
compute the Jacobian of the Lasso as described in `this paper <https://arxiv.org/pdf/2002.08943.pdf>`_.

It also implements a large number of competitors, such as the forward differentiation, the backward differentiation, and the implicit differentiation.


Install
-------

To be able to run the experiments  you should install the conda environment:

::

    conda env create -f environment.yml
    conda activate sparse-ho-env

(you may need to open  fresh new terminal).

To be able to run the code you first need to run, in the folder that contains
the setup.py file (root folder):

::

    pip install -e .


You should now be able to run a first example which reproduces Figure 1:

::

    ipython -i examples/plot_time_to_compute_single_gradient.py

If you want to compare methods to solve the whole hyperparameter optimization
problem, you can run the example:

::

    ipython -i examples/plot_friendly_example.py

in particular you can play with the parameters of the bayesian solver in
the implicit_forward/bayesian.py file.

You should now be able to play with the methods and the estimators.


Reproduce all experiments
=========================

Scripts to reproduce all the experiments: Figure 2, 3, 4, 5.

Be careful, you may want to run these scripts on a server with multiple CPUs.

All the figures of the paper can be reproduced.

What is needed is in the implicit_forward/expes folder:

- Figure 2:

::

    run main_lasso_pred.py
    ipython -i plot_lasso_pred.py

- Figure 3:

::

    run main_lasso_est.py
    ipython -i plot_lasso_est.py

- Figure 4:

::

    run main_wLasso.py
    ipython -i plot_wLasso.py

- Figure 5 in Appendix:

::

    run main_mcp_pred.py
    ipython -i plot_mcp_pred.py



Cite
====

If you use this code, please cite:

::

    @article{bertrand2020implicit,
    title={Implicit differentiation of Lasso-type models for hyperparameter optimization},
    author={Bertrand, Quentin and Klopfenstein, Quentin and Blondel, Mathieu and Vaiter, Samuel and Gramfort, Alexandre and Salmon, Joseph},
    journal={arXiv preprint arXiv:2002.08943},
    year={2020}
    }


ArXiv links:

- https://arxiv.org/pdf/2002.08943.pdf

.. |image0| image:: https://travis-ci.org/QB3/sparse-ho.svg?branch=master
   :target: https://travis-ci.org/QB3/sparse-ho/
.. |image1| image:: https://codecov.io/gh/QB3/sparse-ho/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mathurinm/QB3/sparse-ho
