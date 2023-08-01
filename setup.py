#! /usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="DupuitLEM",
    version="0.1",
    description="Groundwater landscape evolution with landlab",
    author="David Litwin",
    author_email="dlitwin3@jhu.edu",
    url='https://github.com/DavidLitwin/DupuitLEM/',
    packages=find_packages(),
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'landlab',
        'tqdm',
    ],
)
