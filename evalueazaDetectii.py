#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 23:50:11 2019

@author: gabriel
"""
import numpy as np
# from time import time
from matplotlib import pyplot as plt

from calculeazaPrecizieClasificator import calculeazaPrecizieClasificator


def evalueazaDetectii(detectii, scoruriDetectii, imageIdx,
                      numeDirectorAdnotariTest):
    """
     'detectii' = matrice Nx4, unde
               N este numarul de detectii
               detectii[i,:] = [x_min, y_min, x_max, y_max]
     'scoruriDetectii' = matrice N. scoruriDetectii[i] este scorul detectiei i
     'imageIdx' = tablou de celule N. imageIdx[i] este imaginea
     in care apare detectia i
    """
    draw = 1

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

    # numar total de adevarat pozitive
    npos = gt_ids.shape[0]
    gt_existaDetectie = np.zeros((npos,), np.bool_)

    # sorteaza detectiile dupa scorul lor
    ind = np.argsort(-scoruriDetectii).ravel()
    scoruriDetectii = scoruriDetectii[ind]
    imageIdx = imageIdx[ind]
    detectii = detectii[ind, :]

    # asigneaza detectii obiectelor ground-truth adnotate
    nd = scoruriDetectii.shape[0]
    tp = np.zeros((nd,), np.int_)
    fp = np.zeros((nd,), np.int_)
    detectii_duplicat = np.zeros((nd,), np.int_)

    # start = time()
    # stop = time() - start

    for d in range(nd):

        cur_gt_ids = imageIdx[d] == gt_ids
        bb = detectii[d, :]
        ovmax = np.NINF

        for j in np.nonzero(cur_gt_ids)[0]:
            bbgt = gt_detectii[j, :]

            bi = (max(bb[0], bbgt[0]),
                  max(bb[1], bbgt[1]),
                  min(bb[2], bbgt[2]),
                  min(bb[3], bbgt[3]))

            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1

            if iw > 0 and ih > 0:
                # calculeaza overlap ca intersectie / reuniune
                ua = ((bb[2] - bb[0] + 1)
                      * (bb[3] - bb[1] + 1)
                      + (bbgt[2] - bbgt[0] + 1)
                      * (bbgt[3] - bbgt[1] + 1)
                      - iw * ih)

                ov = iw * ih / ua

                if ov > ovmax:
                    ovmax = ov
                    jmax = j

        # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
        if ovmax >= 0.3:
            if gt_existaDetectie[jmax]:
                # detectie fals pozitiva (detectie multipla)
                fp[d] = 1
                detectii_duplicat[d] = 1
            else:
                # detectie adevarat pozitiva
                tp[d] = 1
                gt_existaDetectie[jmax] = True

        else:
            # detectie fals pozitiva
            fp[d] = 1

    # calculeaza graficul precizie/recall
    cum_fp = np.cumsum(fp)
    cum_tp = np.cumsum(tp)
    rec = cum_tp / npos
    prec = cum_tp / (cum_fp + cum_tp)

    ap = calculeazaPrecizieClasificator(rec, prec)

    if draw:
        # ploteaza graficul precizie/recall
        fig, ax = plt.subplots()
        plt.plot(rec, prec, '-')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.grid()
        plt.xlabel('recall')
        plt.ylabel('precizie')
        plt.title('Precizie medie = ' + str(ap))
        # set(3, 'Color', [.988, .988, .988])

        # pause(0.1)
        # average_precision_image = frame2im(getframe(3))
        # imwrite(average_precision_image,
        #         '../data/salveazaFisiere/vizualizari/precizie_medie.png')

        # figure(4)
        # plot(cum_fp,rec,'-')
        # axis([0 300 0 1])
        # grid
        # xlabel 'Exemple fals pozitive'
        # ylabel 'Numar detectii corecte (recall)'

        plt.show()

    reverse_map = np.empty((nd,), dtype=int)
    reverse_map[ind] = np.arange(nd)
    tp = tp[reverse_map]
    fp = fp[reverse_map]
    detectii_duplicat = detectii_duplicat[reverse_map]

    return gt_ids, gt_detectii, gt_existaDetectie, tp, fp, detectii_duplicat
