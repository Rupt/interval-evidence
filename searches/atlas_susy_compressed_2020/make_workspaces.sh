#!/bin/bash
# usage:
# ./searches/ins1767649/make_workspaces.sh
cd $(dirname $0)

# download hepdata to prepare workspaces for this search
curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.91374.v4/r6

tar -f statistical_models.tar.gz -x EWKinos_bkgonly.json Sleptons_bkgonly.json

python ../../discohist/specgz.py EWKinos_bkgonly.json ewkinos_bkg.json.gz
python ../../discohist/specgz.py Sleptons_bkgonly.json sleptons_bkg.json.gz

rm statistical_models.tar.gz EWKinos_bkgonly.json Sleptons_bkgonly.json
