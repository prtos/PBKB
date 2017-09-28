#!/usr/bin/env python
#-*- coding:utf-8 -*-

# **********************************************************
# To install:
# python setup.py install
# **********************************************************

import os, numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


def from_pwd(*args):
    return [os.path.join(os.path.dirname(__file__), *args)]


ext_modules = [
    Extension("kernels._gskernel", from_pwd("kernels", "_gskernel.pyx"),
              include_dirs=[numpy.get_include()]),

    Extension("inference.BB.search.bound_calculator",
              from_pwd("inference", "BB", "search", "bound_calculator.pyx"),
              include_dirs=[numpy.get_include()]),
    Extension("inference.BB.search.node",
              from_pwd("inference", "BB", "search", "node.pyx"),
              include_dirs=[numpy.get_include()]),
    Extension("inference.BB.search.node_creator",
              from_pwd("inference", "BB", "search", "node_creator.pyx"),
              include_dirs=[numpy.get_include()]),
    Extension("inference.BB.features.gs_similarity_weights",
              from_pwd("inference", "BB",  "features", "gs_similarity_weights.pyx"),
              include_dirs=[numpy.get_include()]),
    Extension("inference.BB.search._branch_and_bound",
              from_pwd("inference", "BB", "search", "_branch_and_bound.pyx"),
              include_dirs=[numpy.get_include()]),
    Extension("inference.BB.search.search_stats",
              from_pwd("inference", "BB", "search", "search_stats.pyx"),
              include_dirs=[numpy.get_include()]),

    Extension("inference.graph_based.preimage_fast",
              from_pwd("inference", "graph_based", "preimage_fast.pyx"),
              include_dirs=[numpy.get_include()]),
]
setup(
    name="PBKB",
    version="1.2",
    author="Prudencio Tossou, Sébastien Giguère, Amélie Roland",
    author_email="prudencio.tossou.1@ulaval.ca",
    description='Tools to learn peptide\'s bioactivity predictor and find the peptides that maximize the output of that predictor.',
    license="GPL",
    keywords="Preimage gs-kernel peptides",
    # Dependencies
    install_requires=['numpy>=1.6.2'],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)