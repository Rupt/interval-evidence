#!/bin/bash
# usage:
# ./searches/atlas_susy_3Lss_2019/make_workspaces.sh
cd $(dirname $0)

curl -OJLH "Accept: application/x-tar" https://doi.org/10.17182/hepdata.91214.v4/r6
#
tar -f InclSS3L_likelihoods.tgz -x \
Rpc2L0b_BkgOnly.json \
Rpc2L1b_BkgOnly.json \
Rpc2L2b_BkgOnly.json \
Rpc3LSS1b_BkgOnly.json \
Rpv2L_BkgOnly.json

python ../../discohist/specgz.py Rpc2L0b_BkgOnly.json rpc2l0b_bkg.json.gz
python ../../discohist/specgz.py Rpc2L1b_BkgOnly.json rpc2l1b_bkg.json.gz
python ../../discohist/specgz.py Rpc2L2b_BkgOnly.json rpc2l2b_bkg.json.gz
python ../../discohist/specgz.py Rpc3LSS1b_BkgOnly.json rpc3lss1b_bkg.json.gz
python ../../discohist/specgz.py Rpv2L_BkgOnly.json rpv2l_bkg.json.gz

rm -r InclSS3L_likelihoods.tgz \
Rpc2L0b_BkgOnly.json \
Rpc2L1b_BkgOnly.json \
Rpc2L2b_BkgOnly.json \
Rpc3LSS1b_BkgOnly.json \
Rpv2L_BkgOnly.json

