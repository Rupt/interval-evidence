#!/bin/bash
# usage:
# ./searches/atlas_susy_jets_2021/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.95664.v2/r9

JSON_GZS_BDT='
BDT-GGd1_bkgonly.json
BDT-GGd2_bkgonly.json
BDT-GGd3_bkgonly.json
BDT-GGd4_bkgonly.json
BDT-GGo1_bkgonly.json
BDT-GGo2_bkgonly.json
BDT-GGo3_bkgonly.json
BDT-GGo4_bkgonly.json
'

JSON_GZS_SR='
SR2j-1600_bkgonly.json
SR2j-2200_bkgonly.json
SR2j-2800_bkgonly.json
SR4j-1000_bkgonly.json
SR4j-2200_bkgonly.json
SR4j-3400_bkgonly.json
SR5j-1600_bkgonly.json
SR6j-1000_bkgonly.json
SR6j-2200_bkgonly.json
SR6j-3400_bkgonly.json
'

tar -f Likelihoods.tar.gz -x ${JSON_GZS_BDT//BDT/Likelihoods\/BDT} ${JSON_GZS_SR//SR/Likelihoods\/SR}

for JSON_GZ in ${JSON_GZS_BDT} ${JSON_GZS_SR}
do
    python ../../discohisto/specgz.py Likelihoods/${JSON_GZ} ${JSON_GZ/%_bkgonly.json/_bkg.json.gz}
done

rm -r Likelihoods.tar.gz Likelihoods
