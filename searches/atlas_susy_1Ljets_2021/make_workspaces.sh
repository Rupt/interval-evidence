#!/bin/bash
# usage:
# ./searches/atlas_susy_1Ljets_2021/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.97041.v2/r3

tar -f likelihoods.tar.gz -x likelihoods/BkgOnly.json

python ../../discohist/specgz.py likelihoods/BkgOnly.json bkg.json.gz

rm -r likelihoods.tar.gz likelihoods
