#!/bin/bash
# usage:
# ./searches/atlas_susy_hb_2019/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.89408.v3/r2

tar -f HEPData_workspaces.tar.gz -x RegionA/BkgOnly.json RegionB/BkgOnly.json RegionC/BkgOnly.json

python ../../discohisto/specgz.py RegionA/BkgOnly.json a_bkg.json.gz
python ../../discohisto/specgz.py RegionB/BkgOnly.json b_bkg.json.gz
python ../../discohisto/specgz.py RegionC/BkgOnly.json c_bkg.json.gz

rm -r HEPData_workspaces.tar.gz RegionA RegionB RegionC
