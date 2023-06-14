#!/bin/bash

# rm data/stars.parquet

rm data/stars.parquet


rm -rf tmp
mkdir tmp


# Phot corrections from Axelrod et al 2023
# https://ui.adsabs.harvard.edu/abs/2023arXiv230507563A/abstract
# Figs 5 (DES) and 9 (PS1), no y-band correction for PS1

clickhouse local --path=tmp --progress --multiline --multiquery "
CREATE TABLE stars
ENGINE = MergeTree()
ORDER BY (ifNull(des_id, 0), ifNull(ps1_id, 0))
AS SELECT * FROM file('data/ps1_des-grizy.parquet', Parquet)
;


SELECT count() FROM stars;


INSERT INTO stars (
  des_ra,
  des_dec,
  des_id,
  des_mag_g,
  des_magerr_g,
  des_mag_r,
  des_magerr_r,
  des_mag_i,
  des_magerr_i,
  des_mag_z,
  des_magerr_z,
  des_mag_y,
  des_magerr_y
)
SELECT
  des_ra,
  des_dec,
  des_id,
  des_mag_g,
  des_magerr_g,
  des_mag_r,
  des_magerr_r,
  des_mag_i,
  des_magerr_i,
  des_mag_z,
  des_magerr_z,
  des_mag_y,
  des_magerr_y
FROM file('data/des_stars.parquet', Parquet)
WHERE des_id NOT IN (SELECT des_id FROM stars);


SELECT count() FROM stars;


INSERT INTO stars (
  ps1_ra,
  ps1_dec,
  ps1_id,
  ps1_mag_g,
  ps1_magerr_g,
  ps1_mag_r,
  ps1_magerr_r,
  ps1_mag_i,
  ps1_magerr_i,
  ps1_mag_z,
  ps1_magerr_z,
  ps1_mag_y,
  ps1_magerr_y,
  des_mag_g,
  des_magerr_g,
  des_mag_r,
  des_magerr_r,
  des_mag_i,
  des_magerr_i,
  des_mag_z,
  des_magerr_z,
  des_mag_y,
  des_magerr_y
)
SELECT
  base.ps1_ra,
  base.ps1_dec,
  base.ps1_id AS ps1_id,
  base.ps1_mag_g,
  base.ps1_magerr_g,
  base.ps1_mag_r,
  base.ps1_magerr_r,
  base.ps1_mag_i,
  base.ps1_magerr_i,
  base.ps1_mag_z,
  base.ps1_magerr_z,
  base.ps1_mag_y,
  base.ps1_magerr_y,
  des_g.des_mag_g,
  des_g.des_magerr_g,
  des_r.des_mag_r,
  des_r.des_magerr_r,
  des_i.des_mag_i,
  des_i.des_magerr_i,
  des_z.des_mag_z,
  des_z.des_magerr_z,
  des_y.des_mag_y,
  des_y.des_magerr_y
FROM (SELECT * FROM file('data/ps1_stars.parquet', Parquet)) AS base
  INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_g.parquet')) AS des_g ON (base.ps1_id = des_g.ps1_id)
  INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_r.parquet')) AS des_r ON (base.ps1_id = des_r.ps1_id)
  INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_i.parquet')) AS des_i ON (base.ps1_id = des_i.ps1_id)
  INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_z.parquet')) AS des_z ON (base.ps1_id = des_z.ps1_id)
  INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_y.parquet')) AS des_y ON (base.ps1_id = des_y.ps1_id)
WHERE ps1_id NOT IN (SELECT ps1_id FROM stars);


SELECT count() FROM stars;


ALTER TABLE stars ADD COLUMN des_corr_mag_g Nullable(Float32) DEFAULT des_mag_g - 0.011 AFTER des_magerr_g;
ALTER TABLE stars ADD COLUMN des_corr_magerr_g Nullable(Float32) DEFAULT hypot(des_magerr_g, 0.007) AFTER des_corr_mag_g;

ALTER TABLE stars ADD COLUMN des_corr_mag_r Nullable(Float32) DEFAULT des_mag_r + 0.005 AFTER des_magerr_r;
ALTER TABLE stars ADD COLUMN des_corr_magerr_r Nullable(Float32) DEFAULT hypot(des_magerr_r, 0.009) AFTER des_corr_mag_r;

ALTER TABLE stars ADD COLUMN des_corr_mag_i Nullable(Float32) DEFAULT des_mag_i + 0.005 AFTER des_magerr_i;
ALTER TABLE stars ADD COLUMN des_corr_magerr_i Nullable(Float32) DEFAULT hypot(des_magerr_i, 0.008) AFTER des_corr_mag_i;

ALTER TABLE stars ADD COLUMN des_corr_mag_z Nullable(Float32) DEFAULT des_mag_z + 0.015 AFTER des_magerr_z;
ALTER TABLE stars ADD COLUMN des_corr_magerr_z Nullable(Float32) DEFAULT hypot(des_magerr_z, 0.006) AFTER des_corr_mag_z;

ALTER TABLE stars ADD COLUMN des_corr_mag_y Nullable(Float32) DEFAULT des_mag_y - 0.006 AFTER des_magerr_y;
ALTER TABLE stars ADD COLUMN des_corr_magerr_y Nullable(Float32) DEFAULT hypot(des_magerr_y, 0.026) AFTER des_corr_mag_y;


ALTER TABLE stars ADD COLUMN ps1_corr_mag_g Nullable(Float32) DEFAULT ps1_mag_g - 0.001 AFTER ps1_magerr_g;
ALTER TABLE stars ADD COLUMN ps1_corr_magerr_g Nullable(Float32) DEFAULT hypot(ps1_magerr_g, 0.014) AFTER ps1_corr_mag_g;

ALTER TABLE stars ADD COLUMN ps1_corr_mag_r Nullable(Float32) DEFAULT ps1_mag_r - 0.015 AFTER ps1_magerr_r;
ALTER TABLE stars ADD COLUMN ps1_corr_magerr_r Nullable(Float32) DEFAULT hypot(ps1_magerr_r, 0.018) AFTER ps1_corr_mag_r;

ALTER TABLE stars ADD COLUMN ps1_corr_mag_i Nullable(Float32) DEFAULT ps1_mag_i - 0.028 AFTER ps1_magerr_i;
ALTER TABLE stars ADD COLUMN ps1_corr_magerr_i Nullable(Float32) DEFAULT hypot(ps1_magerr_i, 0.018) AFTER ps1_corr_mag_i;

ALTER TABLE stars ADD COLUMN ps1_corr_mag_z Nullable(Float32) DEFAULT ps1_mag_z - 0.086 AFTER ps1_magerr_z;
ALTER TABLE stars ADD COLUMN ps1_corr_magerr_z Nullable(Float32) DEFAULT hypot(ps1_magerr_z, 0.029) AFTER ps1_corr_mag_z;


SELECT * FROM stars INTO OUTFILE 'data/stars.parquet' FORMAT Parquet;


DROP TABLE stars;
"

#clickhouse local --progress --multiline --query "
#SELECT
#  d.des_id AS des_id,
#  d.des_ra AS des_ra,
#  d.des_dec AS des_dec,
#
#  ifNull(d.des_mag_g, p.des_mag_g) AS des_mag_g,
#  des_mag_g - 0.011 AS des_corr_mag_g,
#  ifNull(d.des_magerr_g, p.des_magerr_g) AS des_magerr_g,
#  hypot(des_magerr_g, 0.007) AS des_corr_magerr_g,
#
#  ifNull(d.des_mag_r, p.des_mag_r) AS des_mag_r,
#  des_mag_r + 0.005 AS des_corr_mag_r,
#  ifNull(d.des_magerr_r, p.des_magerr_r) AS des_magerr_r,
#  hypot(des_magerr_r, 0.009) AS des_corr_magerr_r,
#
#  ifNull(d.des_mag_i, p.des_mag_i) AS des_mag_i,
#  des_mag_i + 0.005 AS des_corr_mag_i,
#  ifNull(d.des_magerr_i, p.des_magerr_i) AS des_magerr_i,
#  hypot(des_magerr_i, 0.008) AS des_corr_magerr_i,
#
#  ifNull(d.des_mag_z, p.des_mag_z) AS des_mag_z,
#  des_mag_z + 0.015 AS des_corr_mag_z,
#  ifNull(d.des_magerr_z, p.des_magerr_z) AS des_magerr_z,
#  hypot(des_magerr_z, 0.006) AS des_corr_magerr_z,
#
#  ifNull(d.des_mag_y, p.des_mag_y) AS des_mag_y,
#  des_mag_y - 0.006 AS des_corr_mag_y,
#  ifNull(d.des_magerr_y, p.des_magerr_y) AS des_magerr_y,
#  hypot(des_magerr_y, 0.026) AS des_corr_magerr_y,
#
#  p.ps1_id AS ps1_id,
#  p.ps1_ra AS ps1_ra,
#  p.ps1_dec AS ps1_dec,
#
#  p.ps1_mag_g AS ps1_mag_g,
#  ps1_mag_g - 0.001 AS ps1_corr_mag_g,
#  p.ps1_magerr_g AS ps1_magerr_g,
#  hypot(ps1_magerr_g, 0.014) AS ps1_corr_magerr_g,
#
#  p.ps1_mag_r AS ps1_mag_r,
#  p.ps1_mag_r - 0.015 AS ps1_corr_mag_r,
#  p.ps1_magerr_r AS ps1_magerr_r,
#  hypot(ps1_magerr_r, 0.018) AS ps1_corr_magerr_r,
#
#  p.ps1_mag_i AS ps1_mag_i,
#  ps1_mag_i - 0.028 AS ps1_corr_mag_i,
#  p.ps1_magerr_i AS ps1_magerr_i,
#  hypot(ps1_magerr_i, 0.018) AS ps1_corr_magerr_i,
#
#  p.ps1_mag_z AS ps1_mag_z,
#  ps1_mag_z - 0.096 AS ps1_corr_mag_z,
#  p.ps1_magerr_z AS ps1_magerr_z,
#  hypot(ps1_magerr_z, 0.029) AS ps1_corr_magerr_z,
#
#  p.ps1_mag_y AS ps1_mag_y,
#  p.ps1_magerr_y AS ps1_magerr_y
#FROM file('data/ps1_des-grizy.parquet', Parquet) AS pd
#
#RIGHT OUTER JOIN (
#    SELECT * FROM file('data/des_stars.parquet', Parquet)
#  ) AS d
#  ON (pd.des_id = d.des_id)
#
#RIGHT OUTER JOIN (
#    SELECT
#      base.ps1_id AS ps1_id,
#      ps1_ra,
#      ps1_dec,
#      ps1_mag_g,
#      ps1_magerr_g,
#      ps1_mag_r,
#      ps1_magerr_r,
#      ps1_mag_i,
#      ps1_magerr_i,
#      ps1_mag_z,
#      ps1_magerr_z,
#      ps1_mag_y,
#      ps1_magerr_y,
#      des_mag_g,
#      des_magerr_g,
#      des_mag_r,
#      des_magerr_r,
#      des_mag_i,
#      des_magerr_i,
#      des_mag_z,
#      des_magerr_z,
#      des_mag_y,
#      des_magerr_y
#    FROM file('data/ps1_stars.parquet', Parquet) AS base
#      INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_g.parquet')) AS des_g ON (base.ps1_id = des_g.ps1_id)
#      INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_r.parquet')) AS des_r ON (base.ps1_id = des_r.ps1_id)
#      INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_i.parquet')) AS des_i ON (base.ps1_id = des_i.ps1_id)
#      INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_z.parquet')) AS des_z ON (base.ps1_id = des_z.ps1_id)
#      INNER JOIN (SELECT * FROM file('data/ps1_stars--DES_y.parquet')) AS des_y ON (base.ps1_id = des_y.ps1_id)
#  ) AS p
#  ON (pd.ps1_id = p.ps1_id)
#
#  INTO OUTFILE 'data/stars.parquet' FORMAT Parquet
#--  LIMIT 10 FORMAT Vertical
#"
