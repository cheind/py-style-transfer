# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='styletransfer',
    version=open('style/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='A neural style transfer library.',    
    author='Christoph Heindl',
    url='https://github.com/cheind/py-style-transfer',
    license='MIT',
    install_requires=required,
    packages=['style'],
    include_package_data=True,
    keywords='style-transfer neural deep dream artistic'
)