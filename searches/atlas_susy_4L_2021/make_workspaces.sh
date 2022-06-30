#!/bin/bash
# usage:
# ./searches/atlas_susy_4L_2021/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.103062.v1/r2

tar -f statistical_models.tar.gz -x statistical_models/bgOnly_likelihood.json

python ../../discohist/specgz.py statistical_models/bgOnly_likelihood.json bkg.json.gz

rm -r statistical_models.tar.gz statistical_models
