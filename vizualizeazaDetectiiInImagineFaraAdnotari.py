#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:07:49 2019

@author: gabriel
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pathlib
import os


def vizualizeazaDetectiiInImagineFaraAdnotari(
        detectii, scoruriDetectii, imageIdx, numeDirectorExempleTest):
    """
    'detectii' = matrice Nx4, unde
              N este numarul de detectii
              detectii[i, :] = [x_min, y_min, x_max, y_max]
    'scoruriDetectii' = matrice N. scoruriDetectii[i] este scorul detectiei i
    'imageIdx' = tablou de celule N. imageIdx[i] este imaginea in care
    apare detectia i
    """
    test_files_unsorted = pathlib.Path(numeDirectorExempleTest).glob('*.jpg')
    test_files = sorted([x for x in test_files_unsorted])
    test_files = test_files
    num_test_images = len(test_files)

    for i in range(num_test_images):

        numeImg = os.path.basename(os.path.normpath(test_files[i]))
        cur_test_image = cv2.imread(str(test_files[i]), 0)
        cur_detections = numeImg == imageIdx
        cur_detectii = detectii[cur_detections, :]
        cur_scoruriDetectii = scoruriDetectii[cur_detections]

        fig, ax = plt.subplots()
        # imshow(cur_test_image);
        plt.imshow(cur_test_image, cmap='gray')

        num_detections = sum(cur_detections)

        for j in range(num_detections):
            bb = cur_detectii[j, :]
            plt.plot([bb[0], bb[2], bb[2], bb[0], bb[0]],
                     [bb[1], bb[1], bb[3], bb[3], bb[1]], 'g:', linewidth=2)

        # axis image
        # axis off
        plt.title('Imaginea: "' + numeImg + '" verde=detectie')
        # ,'interpreter','none')

        # set(4, 'Color', [.988, .988, .988])

        # print('Apasati orice tasta pentru a continua cu urmatoarea
        # imagine \n')

        # pause
        plt.show()
