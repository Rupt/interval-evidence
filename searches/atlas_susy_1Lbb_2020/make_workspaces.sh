#!/bin/bash
# usage:
# ./searches/atlas_susy_1Lbb_2020/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.90607.v4/r3

tar -f 1Lbb-likelihoods-hepdata.tar.gz -x BkgOnly.json

python ../../discohist/specgz.py BkgOnly.json bkg.json.gz

rm 1Lbb-likelihoods-hepdata.tar.gz BkgOnly.json
