#!/bin/bash

for BAND in {g,r,i,z,y}; do
  echo $BAND
  apply-photo-transform \
    models/phot-transformation/DES_${BAND}-PS1_g--r--i--z--y.onnx \
    data/ps1_stars.parquet
done
