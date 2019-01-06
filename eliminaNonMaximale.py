#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:56:29 2019

@author: gabriel
"""
import numpy as np


def eliminaNonMaximale(detectii, scoruriDetectii, dimensiuneImg):
    """
    Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea
    dar au scor mai mic.
    Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
    fi in interiorul celeilalte detectii.

    'detectii' = matrice Nx4, unde
              N este numarul de detectii
              detectii(i,:) = [x_min, y_min, x_max, y_max]
    'scoruriDetectii' = matrice N. scoruriDetectii(i) este scorul detectiei i
    'dimensiuneImg' este =  dimensiunile [y,x] ale imaginii

    esteMaxim = matrice N (1 daca detectia este maxim, 0 altfel)
    """
    h, w = dimensiuneImg

    # trunchiaza detectiile la dimensiunile imaginii
    # xmax > dimensiunea x
    x_out_of_bounds = np.argwhere(detectii[:, 2] > w)
    # ymax > dimensiunea y
    y_out_of_bounds = np.argwhere(detectii[:, 3] > h)

    detectii[x_out_of_bounds, 2] = w
    detectii[y_out_of_bounds, 3] = h

    numarDetectii = scoruriDetectii.shape[0]

    # ordonam detectiile in functie de scorul lor
    ind = np.argsort(-scoruriDetectii).ravel()
    scoruriDetectii = scoruriDetectii[ind]
    detectii = detectii[ind, :]

    # indicator for whether each bbox will be accepted or suppressed
    esteMaxim = np.zeros((numarDetectii,), np.bool_)

    for i in range(numarDetectii):

        detectiaCurenta = detectii[i, :].ravel()
        detectiaCurenta_esteMaxim = True

        for j in np.nonzero(esteMaxim)[0]:
            # calculeaza suprapunerea(overlap) cu
            # fiecare detectia confirmata ca fiind maxim

            detectieAnterioara = detectii[j, :].ravel()

            bi = (max(detectiaCurenta[0], detectieAnterioara[0]),
                  max(detectiaCurenta[1], detectieAnterioara[1]),
                  min(detectiaCurenta[2], detectieAnterioara[2]),
                  min(detectiaCurenta[3], detectieAnterioara[3]))

            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1

            if iw > 0 and ih > 0:
                # calculam suprapunere ca fiind intersectie/reuniune
                ua = ((detectiaCurenta[2] - detectiaCurenta[0] + 1)
                      * (detectiaCurenta[3] - detectiaCurenta[1] + 1)
                      + (detectieAnterioara[2] - detectieAnterioara[0] + 1)
                      * (detectieAnterioara[3] - detectieAnterioara[1] + 1)
                      - iw * ih)

                ov = iw * ih / ua
                # daca detectia cu scor mai mic
                # se suprapune prea mult (>0.3) cu detectia anterioara
                if ov > 0.3:
                    detectiaCurenta_esteMaxim = False

                # caz special -- centrul detectiei curente este in interiorul
                # detectiei anterioare
                center_coord = ((detectiaCurenta[0] + detectiaCurenta[2])/2,
                                (detectiaCurenta[1] + detectiaCurenta[3])/2)

                if (
                        center_coord[0] > detectieAnterioara[0]
                        and center_coord[0] < detectieAnterioara[2]
                        and center_coord[1] > detectieAnterioara[1]
                        and center_coord[1] < detectieAnterioara[3]
                   ):
                    detectiaCurenta_esteMaxim = False

        esteMaxim[i] = detectiaCurenta_esteMaxim

    # intoarce-te la ordinea initiala a detectiilor
    reverse_map = np.empty((numarDetectii,), dtype=int)
    reverse_map[ind] = np.arange(numarDetectii)
    esteMaxim = esteMaxim[reverse_map]

    print(' Elimina non-maxime: ' + str(numarDetectii) + ' detectii -> ' +
          str(sum(esteMaxim)) + ' detectii finale\n')

    return esteMaxim
