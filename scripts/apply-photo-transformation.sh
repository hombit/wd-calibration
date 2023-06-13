#!/bin/bash

for BAND in {g,r,i,z,y}; do
  echo $BAND
  train-photo-transform \
    data/ps1_des-grizy.parquet \
    --input-bands g r i z y \
    --modeldir=models/phot-transformation \
    --figdir=figures/phot-transformation/des-ps1 \
    --fig-support-color=r,i \
    --input-survey=PS1 \
    --output-survey=DES \
    --output-band=$BAND
done