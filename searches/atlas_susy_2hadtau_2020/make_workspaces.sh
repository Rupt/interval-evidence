#!/bin/bash
# usage:
# ./searches/atlas_susy_2hadtau_2020/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.92006.v2/r2

tar -f SUSY-2018-04_likelihoods.tar.gz -x Region-highMass/BkgOnly.json Region-lowMass/BkgOnly.json

python ../../discohist/specgz.py Region-highMass/BkgOnly.json high_mass_bkg.json.gz
python ../../discohist/specgz.py Region-lowMass/BkgOnly.json low_mass_bkg.json.gz

rm -r SUSY-2018-04_likelihoods.tar.gz Region-lowMass Region-highMass
