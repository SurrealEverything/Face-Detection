#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:32:43 2018

@author: gabriel
"""
# import numpy as np
# import math


def vizualizeazaTemplateHOG(parametri):
    """Functie care vizualizeaza descriptorii hog
    preferati de SVM pentru a clasifica fete

    TODO: convert the following matlab code to python
    https://github.com/rbgirshick/voc-dpm/blob/master/vis/HOGpicture.m
    https://github.com/rbgirshick/voc-dpm/blob/master/vis/visualizeHOG.m
    """
    w = parametri.w
    b = parametri.b

    """
    # construct a "glyph" for each orientation
    bim1 = np.zeros(bs, bs);
    bim1(:,round(bs/2):round(bs/2)+1) = 1;
    bim = zeros([size(bim1) 9]);
    bim(:,:,1) = bim1;
    for i = 2:9,
      bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
    end

    % make pictures of positive weights bs adding up weighted glyphs
    s = size(w);
    w(w < 0) = 0;
    im = zeros(bs*s(1), bs*s(2));
    for i = 1:s(1),
      iis = (i-1)*bs+1:i*bs;
      for j = 1:s(2),
        jjs = (j-1)*bs+1:j*bs;
        for k = 1:9,
          im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k);
        end
      end
    end
    """
