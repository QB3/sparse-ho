sparse-ho
=========

|image0| |image1|

`sparse-ho` stands for "sparse hyperparameter optimization".
This package implements efficient hyperparameter tuning for sparse machine learning models.
It supports models such as the Lasso, the Weighted Lasso, multiclass sparse Logistic regression, SVM, etc.

Relying on a first order algorithm for bilevel optimization, ``sparse-ho``'s performances scales gracefully with the number of hyperparameters to tune.

In order to benchmark performances, the package also implements alternatives such as forward or backward differentiation.

Documentation
-------------

Please visit `https://qb3.github.io/sparse-ho`_ for the latest version of the documentation.


Install
-------


To run the code you first need to clone the repository, and then run, in the folder containing
the ``setup.py`` file (root folder):

::

    pip install -e .


Cite
====

If you use this code, please cite:

::

    @inproceedings{bertrand2020implicit,
        title={Implicit differentiation of Lasso-type models for hyperparameter optimization},
        author={Bertrand, Q. and Klopfenstein, Q. and Blondel, M. and Vaiter, S. and Gramfort, A. and Salmon, J.},
        booktitle={ICML},
        year={2020},
    }

    @article{bertrand2021implicit,
        title={Implicit differentiation for fast hyperparameter selection in non-smooth convex learning},
        author={Bertrand, Q. and Klopfenstein, Q. and Massias, M. and Blondel, M. and Vaiter, S. and Gramfort, A. and Salmon, S.},
        journal={arXiv preprint arXiv:2105.01637},
        year={2021}
    }


Code to reproduce the figures of the paper is in the ``expes`` folder.


ArXiv link:

- https://arxiv.org/pdf/2002.08943.pdf
- https://arxiv.org/pdf/2105.01637.pdf

.. |image0| image:: https://github.com/QB3/sparse-ho/workflows/build/badge.svg?branch=master
   :target: https://github.com/QB3/sparse-ho/actions?query=workflow%3Abuild
.. |image1| image:: https://codecov.io/gh/QB3/sparse-ho/branch/master/graphs/badge.svg?branch=master
   :target: https://app.codecov.io/gh/qb3/sparse-ho
