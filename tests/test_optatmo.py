# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

from __future__ import print_function
import numpy as np
import piff
import os
import yaml
import subprocess

from piff_test_helper import get_script_name, timer

from time import time

@timer
def test_init():
    print('testing initialization')

@timer
def test_fit():
    print('test that fitting procedure works')

@timer
def test_yaml():
    print('test reading in yaml and executing')

@timer
def test_opt_init():
    print('testing initialization of optical psf')

@timer
def test_opt_fit():
    print('test fitting procedure for optical psf works')

if __name__ == '__main__':
    test_init()
    test_fit()
    test_yaml()
