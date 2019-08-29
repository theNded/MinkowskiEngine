import codecs
import os
import re
import time
import argparse
from subprocess import Popen, PIPE, STDOUT
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

from distutils.sysconfig import get_python_inc

here = os.path.abspath(os.path.dirname(__file__))

# Python interface

# parser = argparse.ArgumentParser()
# parser.add_argument('--version', type=str, required=True,
#                     help='version of MinkowskiEngine, can be found in __init__.py')
# parser.add_argument('--project_dir', type=str, required=True,
#                     help='root dir for the MinkowskiEngine')
# args = parser.parse_args()


print('setup...')
setup(
    name='MinkowskiEngine',
    version='0.2.6',
    install_requires=['torch'],
    packages=[
        'MinkowskiEngine', 'MinkowskiEngine.utils', 'MinkowskiEngine.modules'
    ],
    package_dir={'MinkowskiEngine': '../MinkowskiEngine'},
    ext_modules=[
        CUDAExtension(
            name='MinkowskiEngineBackend',
            include_dirs=['..', get_python_inc() + "/.."],
            sources=[
                '../pybind/minkowski.cpp',
            ],
            libraries=['minkowski', 'openblas'],
            library_dirs=['../build'],
            # extra_compile_args=['-g']
            # Uncomment the following for CPU_ONLY build
            # extra_compile_args=['-DCPU_ONLY']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Christopher B. Choy',
    author_email='chrischoy@ai.stanford.edu',
    description='Minkowski Engine, a Sparse Tensor Library for Neural Networks',
    keywords='Minkowski Engine Sparse Tensor Library Convolutional Neural Networks',
    url='https://github.com/StanfordVL/MinkowskiEngine',
    zip_safe=False,
)
