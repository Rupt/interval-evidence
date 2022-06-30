#!/bin/bash
# usage:
# ./searches/ins1750597/make_workspaces.sh
cd $(dirname $0)

# download hepdata to prepare workspaces for this search
curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.89413.v4/r5

tar -f likelihoods.tar.gz -x bkgonly.json

python ../../pyhf_stuff/specgz.py bkgonly.json bkg.json.gz

rm likelihoods.tar.gz bkgonly.json
