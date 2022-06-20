# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:36:28 2022

@author: Salvo
"""

import numpy

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))