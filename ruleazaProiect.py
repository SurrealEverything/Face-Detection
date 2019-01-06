#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 00:41:32 2018

@author: gabriel

Structura codului:
ruleazaProiect.m                               <--(optional) completati pasul 3
 + obtineDescriptoriExemplePozitive.m          <--trb sa completati aceasta fct
 + obtineDescriptoriExempleNegative.m          <--trb sa completati aceasta fct
 + antreneazaClasificator.m                    <--functie scrisa in intregime
   + calculeazaAcurateteClasificator.m         <--functie scrisa in intregime
 + vizualizeazaTemplateHOG.m                   <--functie scrisa in intregime
 + calculeazaPrecizieClasificator.m            <--functie scrisa in intregime
 + ruleazaDetectorFacial.m                     <--trb sa completati aceasta fct
   + eliminaNonMaximele.m                      <--functie scrisa in intregime
 + evalueazaDetectii.m                         <--functie scrisa in intregime
   + calculeazaPrecizieClasificator.m          <--functie scrisa in intregime
 + vizualizeazaDetectiiInImagineCuAdnotari.m   <--functie scrisa in intregime
 + vizualizeazaDetectiiInImagineFaraAdnotari.m <--functie scrisa in intregime
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

from obtineDescriptoriExemplePozitive import obtineDescriptoriExemplePozitive
from obtineDescriptoriExempleNegative import obtineDescriptoriExempleNegative
from antreneazaClasificator import antreneazaClasificator
# from vizualizeazaTemplateHOG import vizualizeazaTemplateHOG
from ruleazaDetectorFacial import ruleazaDetectorFacial
from evalueazaDetectii import evalueazaDetectii
from vizualizeazaDetectiiInImagineFaraAdnotari \
    import vizualizeazaDetectiiInImagineFaraAdnotari


# Pasul 0 - initializam parametri
class parametri:

    # seteaza path-urile pentru seturile de date: antrenare, test
    numeDirectorSetDate = '/home/gabriel/Spyder Projects/VA/Tema5/data/'
    # exemple pozitive de antrenare: 36x36 fete cropate
    numeDirectorExemplePozitive = numeDirectorSetDate + 'exemplePozitive'
    # exemple negative de antrenare:
    # imagini din care trebuie sa selectati ferestre 36x36
    numeDirectorExempleNegative = numeDirectorSetDate + 'exempleNegative'
    # exemple test din dataset-ul CMU+MIT
    numeDirectorExempleTest = numeDirectorSetDate + 'exempleTest/CMU+MIT'
    # exemple test realizate la laborator si curs
    # numeDirectorExempleTest = numeDirectorSetDate + 'exempleTest/CursVA'
    # fisierul cu adnotari pentru exemplele test din dataset-ul CMU+MIT
    numeDirectorAdnotariTest = (
            numeDirectorSetDate
            + 'exempleTest/CMU+MIT_adnotari/ground_truth_bboxes.txt')
    existaAdnotari = 1
    numeDirectorSalveazaFisiere = numeDirectorSetDate + 'salveazaFisiere/'
    if not os.path.exists(numeDirectorSalveazaFisiere):
        os.makedirs(numeDirectorSalveazaFisiere)

    # seteaza valori pentru diferiti parametri
    # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
    dimensiuneFereastra = (36, 36)
    # dimensiunea blocului
    dimensiuneBloc = (12, 12)
    # pas bloc
    pasBloc = (6, 6)
    # dimensiunea celulei
    dimensiuneCelulaHOG = (6, 6)
    # nr de orientari ale histogramelor de gradienti
    orientari = 9
    # cat de mult trebuie sa se suprapuna doua detectii
    # pentru a o elimina pe cea cu scorul mai mic
    overlap = 0.3
    # (optional)antrenare cu exemple puternic negative
    antrenareCuExemplePuternicNegative = 0
    # numarul exemplelor pozitive
    numarExemplePozitive = 6713
    # numarul exemplelor negative
    numarExempleNegative = 10000
    # toate ferestrele cu scorul > threshold si maxime locale devin detectii
    threshold = 0.1
    # vizualizeaza template HOG
    # vizualizareTemplateHOG = 0
    # kernel folosit:
    # 'liniar'
    # Radial basis: 'rbf' (nu este suportat)
    kernel = 'liniar'
    # weights and bias
    w = None
    b = None
    model = None
    # nr de mariri/micsorari ale imaginilor
    # pe care este rulat detectorul facial
    # 1 = no scaling
    numarRedimensionari = 1
    # factorul de redimensionare maxim
    redimensionareMaxima = 1
    # pasul uc care glisam fereastra
    pasGlisare = 1


# Pasul 1. Incarcam exemplele pozitive (cropate) si exemple negative generate

# exemple pozitive
numeFisierDescriptoriExemplePozitive = (
            parametri.numeDirectorSalveazaFisiere
            + 'descriptoriExemplePozitive_'
            + str(parametri.dimensiuneCelulaHOG[0])
            + '_'
            + str(parametri.numarExemplePozitive)
            + '.csv')

fisierDescriptoriExemplePozitive = Path(numeFisierDescriptoriExemplePozitive)

if fisierDescriptoriExemplePozitive.is_file():
    descriptoriExemplePozitive = pd.read_csv(
            numeFisierDescriptoriExemplePozitive)
    descriptoriExemplePozitive = descriptoriExemplePozitive.drop(
            ["Unnamed: 0"], axis=1)
    descriptoriExemplePozitive = descriptoriExemplePozitive.values
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive: ')
    descriptoriExemplePozitive = obtineDescriptoriExemplePozitive(parametri)
    pdDescExPoz = pd.DataFrame(descriptoriExemplePozitive)
    pdDescExPoz.to_csv(numeFisierDescriptoriExemplePozitive)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul '
          + numeFisierDescriptoriExemplePozitive)

# exemple negative
numeFisierDescriptoriExempleNegative = (
        parametri.numeDirectorSalveazaFisiere
        + 'descriptoriExempleNegative_'
        + str(parametri.dimensiuneCelulaHOG[0])
        + '_'
        + str(parametri.numarExempleNegative)
        + '.csv')

fisierDescriptoriExempleNegative = Path(numeFisierDescriptoriExempleNegative)

if fisierDescriptoriExempleNegative.is_file():
    descriptoriExempleNegative = pd.read_csv(
            numeFisierDescriptoriExempleNegative)
    descriptoriExempleNegative = descriptoriExempleNegative.drop(
            ["Unnamed: 0"], axis=1)
    descriptoriExempleNegative = descriptoriExempleNegative.values
    print('Am incarcat descriptorii pentru exemplele negative')

else:
    print('Construim descriptorii pentru exemplele negative: ')
    descriptoriExempleNegative = obtineDescriptoriExempleNegative(parametri)
    pdDescExNeg = pd.DataFrame(descriptoriExempleNegative)
    pdDescExNeg.to_csv(numeFisierDescriptoriExempleNegative)
    print('Am salvat descriptorii pentru exemplele negative in fisierul '
          + numeFisierDescriptoriExempleNegative)

print('Am obtinut exemplele de antrenare')

# Pasul 2. Invatam clasificatorul liniar
X_train = np.concatenate((descriptoriExemplePozitive,
                          descriptoriExempleNegative), axis=0)

y_train = np.concatenate(
        (np.ones(parametri.numarExemplePozitive),
         np.negative(np.ones(parametri.numarExempleNegative))), axis=0)


w, b, model = antreneazaClasificator(X_train, y_train, parametri.kernel)
parametri.model = model

parametri.w = w
parametri.b = b

"""
# vizualizare model invatat HOG
if parametri.vizualizareTemplateHOG:
    vizualizeazaTemplateHOG(parametri)
"""

"""
# Pasul 3. (optional) Antrenare cu exemple puternic negative
# (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia
# ruleazaDetectorFacial.m astfel incat sa va returneze descriptorii
# detectiilor cu scor >0 din cele 274 imagini negative
"""

# Pasul 4. Ruleaza detectorul facial pe imaginile de test.
detectii, scoruriDetectii, imageIdx = ruleazaDetectorFacial(parametri)

# Pasul 5. Evalueaza si vizualizeaza detectiile
# Pentru imagini pentru care exista adnotari (cele din setul de date  CMU+MIT)
# folositi functia vizualizeazaDetectiiInImagineCuAdnotari.m,
# pentru imagini fara adnotari (cele realizate la curs si laborator)
# folositi functia vizualizeazaDetectiiInImagineFaraAdnotari.m,

if parametri.existaAdnotari:
    gt_ids, gt_detectii, gt_existaDetectie, tp, fp, detectii_duplicat = \
        evalueazaDetectii(detectii, scoruriDetectii, imageIdx,
                          parametri.numeDirectorAdnotariTest)
    """
    vizualizeazaDetectiiInImagineCuAdnotari(
            detectii, scoruriDetectii, imageIdx, tp, fp,
            parametri.numeDirectorExempleTest,
            parametri.numeDirectorAdnotariTest)
    """

else:
    vizualizeazaDetectiiInImagineFaraAdnotari(
        detectii, scoruriDetectii, imageIdx, parametri.numeDirectorExempleTest)
