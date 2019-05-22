#!/usr/bin/env python
from distutils.core import setup

setup(
    name='OptimalGroups',
    version='0.1dev',
    packages=['optimalgroups',],
    license='Blue Oak Model License Version 1.0.0',
    long_description=open('README.md').read(),
    scripts=['bin/og.py'],
    install_requires=[
        'argparse',
        'pandas',
        'numpy',
        'pulp',
    ],
)
