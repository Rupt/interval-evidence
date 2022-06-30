#!/bin/bash
# usage:
# ./searches/ins1755298/make_workspaces.sh
cd $(dirname $0)

# download hepdata to prepare workspaces for this search
curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.90607.v4/r3

tar -f 1Lbb-likelihoods-hepdata.tar.gz -x BkgOnly.json

python ../../pyhf_stuff/specgz.py BkgOnly.json bkg.json.gz

rm 1Lbb-likelihoods-hepdata.tar.gz BkgOnly.json
