#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m' # No Color
for DS in "apps" "humaneval" ; do
    echo -e "Running ${GREEN}${DS}${NC} analysis and plotting script."
    cd data/$DS
    bash run-analysis-make-figs.sh
    cd -
done
echo -e "${GREEN}Done.${NC}"
