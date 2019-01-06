#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:21:19 2018

@author: gabriel
"""
import numpy as np
import cv2
import pathlib
import os
from itertools import product

from eliminaNonMaximale import eliminaNonMaximale


def ruleazaDetectorFacial(parametri):
    """
    'detectii' = matrice Nx4, unde
              N este numarul de detectii
              detectii[i,:] = [x_min, y_min, x_max, y_max]
    'scoruriDetectii' = matrice N. scoruriDetectii[i] este scorul detectiei i
    'imageIdx' = tablou de celule N. imageIdx{i} este imaginea
    in care apare detectia i (nu punem intregul path,
    ci doar numele imaginii: 'albert.jpg')

    Aceasta functie returneaza toate detectiile ( = ferestre) pentru
    toate imaginile din parametri.numeDirectorExempleTest
    Directorul cu numele parametri.numeDirectorExempleTest contine imagini ce
    pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete
    atat pe setul de date MIT+CMU dar si pentru alte imagini
    (imaginile realizate cu voi la curs+laborator).
    Functia 'suprimeazaNonMaximele' suprimeaza detectii care se suprapun
    (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
    Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.

    Functia voastra ar trebui sa calculeze pentru fiecare imagine
    descriptorul HOG asociat. Apoi glisati o fereastra de dimeniune
    paremtri.dimensiuneFereastra x  paremtri.dimensiuneFereastra
    (implicit 36x36) si folositi clasificatorul liniar (w,b) invatat
    poentru a obtine un scor. Daca acest scor este deasupra unui prag
    (threshold) pastrati detectia iar apoi procesati toate detectiile
    prin suprimarea non maximelor.
    pentru detectarea fetelor de diverse marimi folosit un detector multiscale
    """

    imgFilesUnsorted = pathlib.Path(
            parametri.numeDirectorExempleTest).glob('*.jpg')
    imgFiles = sorted([x for x in imgFilesUnsorted])
    imgFiles = imgFiles

    numarImagini = len(imgFiles)

    # initializare variabile de returnat
    detectii = []
    scoruriDetectii = []
    imageIdx = []

    # marimea fiecarei imagini
    imageSizes = np.empty((numarImagini,), dtype=object)
    # intervalul fiecarei imagini
    imageIntervals = np.empty((numarImagini,), dtype=object)
    count = 0

    scalingFactorPos = np.linspace(1, parametri.redimensionareMaxima,
                                   num=parametri.numarRedimensionari)
    scalingFactorNeg = np.flip(1 / scalingFactorPos, axis=0)
    scalingFactor = np.concatenate((scalingFactorNeg, scalingFactorPos[1:]))

    hog = cv2.HOGDescriptor(parametri.dimensiuneFereastra,
                            parametri.dimensiuneBloc,
                            parametri.pasBloc,
                            parametri.dimensiuneCelulaHOG,
                            parametri.orientari)

    for fileIdx in range(numarImagini):

        start = count

        numeImg = os.path.basename(os.path.normpath(imgFiles[fileIdx]))
        print('Rulam detectorul facial pe imaginea ' + numeImg)

        img = cv2.imread(str(imgFiles[fileIdx]), 0)

        for resIdx in range(parametri.numarRedimensionari*2-1):

            iH, iW = img.shape

            sF = scalingFactor[resIdx]

            resImg = cv2.resize(img, None, fx=sF, fy=sF)

            print('in rezolutia ' + str(resImg.shape)
                  + ' (scaling=' + str(sF) + ')')

            h, w = resImg.shape

            # limita inceputului unei ferestre
            limH = h - parametri.dimensiuneFereastra[0] + 1
            limW = w - parametri.dimensiuneFereastra[0] + 1

            # [(i, j) for i in range(limH) for j in range(limW)]

            for i, j in product(range(0, limH, parametri.pasGlisare),
                                range(0, limW, parametri.pasGlisare)):

                x_min = j
                x_max = j + parametri.dimensiuneFereastra[0]
                y_min = i
                y_max = i + parametri.dimensiuneFereastra[0]

                fereastra = resImg[y_min:y_max, x_min:x_max]

                desc = np.ravel(hog.compute(fereastra))

                scor = np.sum(parametri.w * desc) + parametri.b

                if scor > parametri.threshold:
                    detectii.append((x_min, y_min, x_max, y_max))
                    scoruriDetectii.append(scor)
                    imageIdx.append(numeImg)
                    count += 1

        stop = count

        imageSizes[fileIdx] = img.shape
        imageIntervals[fileIdx] = (start, stop)

    detectii = np.array(detectii)
    scoruriDetectii = np.array(scoruriDetectii).ravel()
    imageIdx = np.array(imageIdx)
    # imageSizes = np.array(imageSizes)
    # imageIntervals = np.array(imageIntervals)

    # numarImagini = imageIntervals.shape[0]

    esteMaximFinal = np.array([])

    for idx in range(numarImagini):

        start, stop = imageIntervals[idx]

        esteMaxim = eliminaNonMaximale(detectii[start:stop],
                                       scoruriDetectii[start:stop],
                                       imageSizes[idx])

        esteMaximFinal = np.concatenate((esteMaximFinal, esteMaxim), axis=0)

    maxId = np.nonzero(esteMaximFinal)[0]
    return detectii[maxId], scoruriDetectii[maxId], imageIdx[maxId]
