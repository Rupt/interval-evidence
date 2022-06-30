#!/bin/bash
# usage:
# ./searches/atlas_susy_trilepton_2020/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.99806.v2/r2

tar -f FullLikelihoods_sm.tar.gz -x inclusive_bkgonly.json

python ../../discohist/specgz.py inclusive_bkgonly.json bkg.json.gz

rm -r FullLikelihoods_sm.tar.gz inclusive_bkgonly.json

