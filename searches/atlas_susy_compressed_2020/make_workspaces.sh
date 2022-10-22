#!/bin/bash
# usage:
# ./searches/atlas_susy_compressed_2020/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.91374.v5/r6

tar -f statistical_models.tar.gz -x statistical_models/EWKinos_bkgonly.json statistical_models/Sleptons_bkgonly.json

python ../../discohisto/specgz.py statistical_models/EWKinos_bkgonly.json ewkinos_bkg.json.gz
python ../../discohisto/specgz.py statistical_models/Sleptons_bkgonly.json sleptons_bkg.json.gz

rm -r statistical_models.tar.gz statistical_models
