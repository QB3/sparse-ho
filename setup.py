#! /usr/bin/env python

import os
from setuptools import setup, find_packages

descr = 'Implicit forward differentiation for Lasso-type problems'

version = None
with open(os.path.join('sparse_ho', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


DISTNAME = 'sparse_ho'
DESCRIPTION = descr
AUTHOR = ('Q. Bertrand', 'Q. Klopfenstein')
AUTHOR_EMAIL = 'quentin.bertrand@inria.fr'
LICENSE = 'BSD'
DOWNLOAD_URL = 'https://github.com/QB3/sparse-ho.git'
VERSION = version
URL = 'https://github.com/QB3/sparse-ho'


if __name__ == "__main__":
    setup(name=DISTNAME,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=find_packages(),
          )
