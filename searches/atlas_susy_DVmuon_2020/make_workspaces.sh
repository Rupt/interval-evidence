#!/bin/bash
# usage:
# ./searches/atlas_susy_DVmuon_2020/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/zip" https://doi.org/10.17182/hepdata.91760.v2/r1

unzip Likelihoods.zip Likelihoods/DVPlusMuConfig/SRMET_bkgonly.json Likelihoods/DVPlusMuConfig/SRMU_bkgonly.json

python ../../discohist/specgz.py Likelihoods/DVPlusMuConfig/SRMET_bkgonly.json srmet_bkg.json.gz
python ../../discohist/specgz.py Likelihoods/DVPlusMuConfig/SRMU_bkgonly.json srmu_bkg.json.gz

rm -r Likelihoods.zip Likelihoods
