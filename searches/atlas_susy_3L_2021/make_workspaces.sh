#!/bin/bash
# usage:
# ./searches/atlas_susy_3L_2021/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.95751.v2/r3

tar -f statistical_models.tar.gz -x bkg_offshell.json bkg_onshell.json

python ../../discohisto/specgz.py bkg_offshell.json offshell_bkg.json.gz
python ../../discohisto/specgz.py bkg_onshell.json onshell_bkg.json.gz

rm -r statistical_models.tar.gz bkg_onshell.json bkg_offshell.json
