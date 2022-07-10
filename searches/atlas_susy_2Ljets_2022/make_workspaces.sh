#!/bin/bash
# usage:
# ./searches/atlas_susy_2Ljets_2022/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.116034.v1/r34

JSON_TGZS='
ewk_discovery_bkgonly.json.tgz
RJR_SR2L_ISR_discovery_bkgonly.json.tgz
RJR_SR2L_LOW_discovery_bkgonly.json.tgz
STR-SRC_12_31_discovery_bkgonly.json.tgz
STR-SRC_12_61_discovery_bkgonly.json.tgz
STR-SRC_31_81_discovery_bkgonly.json.tgz
STR-SRC_81__discovery_bkgonly.json.tgz
STR-SRHigh_12_301_discovery_bkgonly.json.tgz
STR-SRHigh_301__discovery_bkgonly.json.tgz
STR-SRLow_101_201_discovery_bkgonly.json.tgz
STR-SRLow_101_301_discovery_bkgonly.json.tgz
STR-SRLow_12_81_discovery_bkgonly.json.tgz
STR-SRLow_301__discovery_bkgonly.json.tgz
STR-SRMed_101__discovery_bkgonly.json.tgz
STR-SRMed_12_101_discovery_bkgonly.json.tgz
STR-SRZHigh___discovery_bkgonly.json.tgz
STR-SRZLow___discovery_bkgonly.json.tgz
STR-SRZMed___discovery_bkgonly.json.tgz
'

tar -f llh_jsons.tar -x ${JSON_TGZS} ewk_discovery_patchset.json.tgz

for JSON_TGZ in ${JSON_TGZS}
do
    tar -f ${JSON_TGZ} -x ${JSON_TGZ/%.tgz/}
done
tar -f ewk_discovery_patchset.json.tgz -x ewk_discovery_patchset.json


pyhf patchset apply --name DR_High_EWK --output-file ewk_high_bkg.json \
ewk_discovery_bkgonly.json ewk_discovery_patchset.json

pyhf patchset apply --name DR_llbb_EWK --output-file ewk_llbb_bkg.json \
ewk_discovery_bkgonly.json ewk_discovery_patchset.json

pyhf patchset apply --name DR_Int_EWK --output-file ewk_int_bkg.json \
ewk_discovery_bkgonly.json ewk_discovery_patchset.json

pyhf patchset apply --name DR_Low_EWK --output-file ewk_low_bkg.json \
ewk_discovery_bkgonly.json ewk_discovery_patchset.json

pyhf patchset apply --name DR_OffShell_EWK --output-file ewk_offshell_bkg.json \
ewk_discovery_bkgonly.json ewk_discovery_patchset.json


python ../../discohisto/specgz.py ewk_high_bkg.json ewk_high_bkg.json.gz
python ../../discohisto/specgz.py ewk_llbb_bkg.json ewk_llbb_bkg.json.gz
python ../../discohisto/specgz.py ewk_int_bkg.json ewk_int_bkg.json.gz
python ../../discohisto/specgz.py ewk_low_bkg.json ewk_low_bkg.json.gz
python ../../discohisto/specgz.py ewk_offshell_bkg.json ewk_offshell_bkg.json.gz


python ../../discohisto/specgz.py RJR_SR2L_ISR_discovery_bkgonly.json rjr_sr2l_isr_bkg.json.gz
python ../../discohisto/specgz.py RJR_SR2L_LOW_discovery_bkgonly.json rjr_sr2l_low_bkg.json.gz


python ../../discohisto/specgz.py STR-SRC_12_31_discovery_bkgonly.json str_src_12_31_bkg.json.gz
python ../../discohisto/specgz.py STR-SRC_12_61_discovery_bkgonly.json str_src_12_61_bkg.json.gz
python ../../discohisto/specgz.py STR-SRC_31_81_discovery_bkgonly.json str_src_31_81_bkg.json.gz
python ../../discohisto/specgz.py STR-SRC_81__discovery_bkgonly.json str_src_81_bkg.json.gz

python ../../discohisto/specgz.py STR-SRHigh_12_301_discovery_bkgonly.json str_srhigh_12_301_bkg.json.gz
python ../../discohisto/specgz.py STR-SRHigh_301__discovery_bkgonly.json str_srhigh_301_bkg.json.gz

python ../../discohisto/specgz.py STR-SRLow_101_201_discovery_bkgonly.json str_srlow_101_201_bkg.json.gz
python ../../discohisto/specgz.py STR-SRLow_101_301_discovery_bkgonly.json str_srlow_101_301_bkg.json.gz
python ../../discohisto/specgz.py STR-SRLow_12_81_discovery_bkgonly.json str_srlow_12_81_bkg.json.gz
python ../../discohisto/specgz.py STR-SRLow_301__discovery_bkgonly.json str_srlow_301_bkg.json.gz

python ../../discohisto/specgz.py STR-SRMed_12_101_discovery_bkgonly.json str_srmed_12_101_bkg.json.gz
python ../../discohisto/specgz.py STR-SRMed_101__discovery_bkgonly.json str_srmed_101_bkg.json.gz

python ../../discohisto/specgz.py STR-SRZHigh___discovery_bkgonly.json str_srzhigh_bkg.json.gz
python ../../discohisto/specgz.py STR-SRZLow___discovery_bkgonly.json str_srzlow_bkg.json.gz
python ../../discohisto/specgz.py STR-SRZMed___discovery_bkgonly.json str_srzmed_bkg.json.gz


rm -r llh_jsons.tar ${JSON_TGZS} ${JSON_TGZS//.tgz/} \
ewk_discovery_patchset.json.tgz \
ewk_discovery_patchset.json \
ewk_high_bkg.json \
ewk_llbb_bkg.json \
ewk_int_bkg.json \
ewk_low_bkg.json \
ewk_offshell_bkg.json
