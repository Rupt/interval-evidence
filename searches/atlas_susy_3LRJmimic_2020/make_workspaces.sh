#!/bin/bash
# usage:
# ./searches/atlas_susy_3LRJmimic_2020/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.91127.v2/r3

tar -f likelihoods_ANA-SUSY-2018-06_3L-RJ-mimic.tar.gz -x BkgOnly.json

python ../../discohist/specgz.py BkgOnly.json bkg.json.gz

rm likelihoods_ANA-SUSY-2018-06_3L-RJ-mimic.tar.gz BkgOnly.json
