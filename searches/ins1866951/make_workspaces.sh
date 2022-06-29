#!/bin/bash
# usage:
# ./searches/ins1866951/make_workspaces.sh
cd $(dirname $0)

# download hepdata to prepare workspaces for this search
curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.95751.v1/r3

tar -f statistical_models.tar.gz -x statistical_models/bkg_offshell.json statistical_models/bkg_onshell.json

python ../../pyhf_stuff/specgz.py statistical_models/bkg_offshell.json offshell_bkg.json.gz
python ../../pyhf_stuff/specgz.py statistical_models/bkg_onshell.json onshell_bkg.json.gz

rm -r statistical_models.tar.gz statistical_models
