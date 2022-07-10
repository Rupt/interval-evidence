#!/bin/bash
# usage:
# ./searches/atlas_susy_2L0J_2019/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.89413.v4/r5

tar -f likelihoods.tar.gz -x bkgonly.json

python ../../discohisto/specgz.py bkgonly.json bkg.json.gz

rm likelihoods.tar.gz bkgonly.json
