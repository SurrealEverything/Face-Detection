#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:49:57 2019

@author: gabriel
"""
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import os
import cv2


def vizualizeazaDetectiiInImagineCuAdnotari(
        detectii, scoruriDetectii, imageIdx, tp, fp, numeDirectorExempleTest,
        numeDirectorAdnotariTest):
    """
    'detectii' = matrice N, unde
              N este numarul de detectii
              detectii[i,:] = [x_min, y_min, x_max, y_max]
    'scoruriDetectii' = matrice N. scoruriDetectii[i] este scorul detectiei i
    'imageIdx' = tablou de celule N. imageIdx[i] este imaginea in care apare
    detectia i
    tp - true positives
    fp - false positives
    """

    test_files_unsorted = pathlib.Path(numeDirectorExempleTest).glob('*.jpg')
    test_files = sorted([x for x in test_files_unsorted])
    test_files = test_files
    num_test_images = len(test_files)

    with open(numeDirectorAdnotariTest, 'r') as fid:
            data = fid.readlines()

    gt_ids = []
    gt_detectii = []

    for line in data:
        words = line.split()
        gt_ids.append(words[0])
        detectie = list(map(int, words[1:]))
        gt_detectii.append(tuple(detectie))

    gt_ids = np.array(gt_ids)
    gt_detectii = np.array(gt_detectii)

    # gt_file_list = np.unique(gt_ids)

    # num_test_images = gt_file_list.shape[0]

    for i in range(num_test_images):

        numeImg = os.path.basename(os.path.normpath(test_files[i]))
        cur_test_image = cv2.imread(str(test_files[i]), 1)

        cur_gt_detections = numeImg == gt_ids
        cur_gt_detectii = gt_detectii[cur_gt_detections, :]

        cur_detections = numeImg == imageIdx
        cur_detectii = detectii[cur_detections, :]
        # cur_scoruriDetectii = scoruriDetectii[cur_detections]

        cur_tp = tp[cur_detections]
        cur_fp = fp[cur_detections]

        fig, ax = plt.subplots()
        # imshow(cur_test_image);
        # hold on;
        plt.imshow(cur_test_image)  # , cmap='gray')

        num_detections = sum(cur_detections)

        for j in range(num_detections):
            bb = cur_detectii[j, :]

            if(cur_tp[j]):  # detectie corecta
                plt.plot([bb[0], bb[2], bb[2], bb[0], bb[0]],
                         [bb[1], bb[1], bb[3], bb[3], bb[1]], 'g-',
                         linewidth=2)
            elif(cur_fp[j]):
                plt.plot([bb[0], bb[2], bb[2], bb[0], bb[0]],
                         [bb[1], bb[1], bb[3], bb[3], bb[1]], 'r-',
                         linewidth=2)
            else:
                print('detectia nu e nici adevarat pozitiva nici fals'
                      + ' pozitiva')

        num_gt_detectii = cur_gt_detectii.shape[0]

        for j in range(num_gt_detectii):

            bbgt = cur_gt_detectii[j, :]

            plt.plot([bbgt[0], bbgt[2], bbgt[2], bbgt[0], bbgt[0]],
                     [bbgt[1], bbgt[1], bbgt[3], bbgt[3], bbgt[1]], 'y-',
                     linewidth=2)

        # hold off;
        # axis image;
        # axis off;

        plt.title('Imaginea: "' + numeImg + '" (verde=detectie adevarata'
                  + ', rosu=detectie falsa, galben=ground-truth adnotat)'
                  + str(sum(cur_tp)) + '/' + str(cur_gt_detectii.shape[0]))

        # set(4, 'Color', [.988, .988, .988])
        # pause(0.1) %p
        # detection_image = frame2im(getframe(4));
        # imwrite(detection_image, sprintf(
        # '../data/salveazaFisiere/vizualizari/detectii_%s.png'
        # , gt_file_list{i}))

        # print('Apasati orice tasta pentru a continua cu urmatoarea
        # imagine \n');
        plt.show()
