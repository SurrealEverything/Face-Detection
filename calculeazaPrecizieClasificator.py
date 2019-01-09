    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 03:11:20 2019

@author: gabriel
"""
import numpy as np


def calculeazaPrecizieClasificator(rec, prec):
    """
    functie inspirata din 2010 Pascal VOC development kit
    """
    mrec = np.concatenate(([0], rec, [1]), axis=0)
    mpre = np.concatenate(([0], prec, [0]), axis=0)

    for i in range(mpre.size-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1

    ap = sum((mrec[i] - mrec[i-1]) * mpre[i])

    return ap
