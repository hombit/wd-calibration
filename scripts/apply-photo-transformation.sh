#!/bin/bash

for BAND in {g,r,i,z,y}; do
  echo $BAND
#  RMSE=$(jq '.["rmse"]' models/phot-transformation/DES_${BAND}-PS1_g--r--i--z--y.json)
  apply-photo-transform \
    models/phot-transformation/DES_${BAND}-PS1_g--r--i--z--y.onnx \
    data/ps1_stars.parquet \
    --batch-size=1_000_000 \
    --var-model=models/phot-transformation/var_DES_${BAND}-PS1_g--r--i--z--y.onnx
#    --estimate-errors --sigma=${RMSE}
done
