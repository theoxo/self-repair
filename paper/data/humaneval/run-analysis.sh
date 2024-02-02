#!/bin/bash
# Run all analysis scripts

if [ -z "$(find . -maxdepth 1 -name '*.jsonl' -print -quit)" ]; then
    # Extract the tarball
    tar -xjf humaneval-data.tar.bz2
else
    echo "JSONL files appear to already be present, skipping tarball extraction."
fi

mkdir -p ./processed
FULL_NP="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50"
for SEQMODE in "batch" "sequential-dfs" ; do
# gpt-3.5 baseline and gpt-3.5+gpt-3.5 "full kn"
python3 ../../analysis/sample-and-estimate.py --input "chatgpt-humaneval-results-tokens.jsonl" --output "processed/gpt3.5-full-kn-$SEQMODE.json" --with-replacement --mode $SEQMODE &
# gpt-3.5 baseline and gpt-3.5+gpt-3.5 "(n,1)"
python3 ../../analysis/sample-and-estimate.py --input "chatgpt-humaneval-results-tokens.jsonl" --output "processed/gpt3.5-nfr1-$SEQMODE.json" --with-replacement --np $FULL_NP --nfr 1 --mode $SEQMODE &

# gpt-4 baseline and gpt-4+gpt-4 "full kn"
python3 ../../analysis/sample-and-estimate.py --input "gpt4-humaneval-results-tokens.jsonl" --output "processed/gpt4-full-kn-$SEQMODE.json" --with-replacement --mode $SEQMODE &
# gpt-4 baseline and gpt-4+gpt-4 "(n,1)"
python3 ../../analysis/sample-and-estimate.py --input "gpt4-humaneval-results-tokens.jsonl" --output "processed/gpt4-nfr1-$SEQMODE.json" --with-replacement --np $FULL_NP --nfr 1 --mode $SEQMODE &

# codellama baseline and codellama+codellama "full kn"
python3 ../../analysis/sample-and-estimate.py --input "codellama-humaneval-results-tokens.jsonl" --output "processed/codellama-full-kn-$SEQMODE.json" --with-replacement --mode $SEQMODE &
# codellama baseline and codellama+codellama "(n,1)"
python3 ../../analysis/sample-and-estimate.py --input "codellama-humaneval-results-tokens.jsonl" --output "processed/codellama-nfr1-$SEQMODE.json" --with-replacement --np $FULL_NP --nfr 1 --mode $SEQMODE &

# codellama + gpt-3.5 "full kn"
python3 ../../analysis/sample-and-estimate.py --input "codellama+chatgpt-tokens.jsonl" --output "processed/codellama+gpt3.5-full-kn-$SEQMODE.json" --with-replacement --mode $SEQMODE &
# codellama + gpt-3.5 "(n,1)"
python3 ../../analysis/sample-and-estimate.py --input "codellama+chatgpt-tokens.jsonl" --output "processed/codellama+gpt3.5-nfr1-$SEQMODE.json" --with-replacement --np $FULL_NP --nfr 1 --mode $SEQMODE &

# codellama + gpt-4 "full kn"
python3 ../../analysis/sample-and-estimate.py --input "codellama+gpt4-tokens.jsonl" --output "processed/codellama+gpt4-full-kn-$SEQMODE.json" --with-replacement --mode $SEQMODE &
# codellama + gpt-3.5 "(n,1)"
python3 ../../analysis/sample-and-estimate.py --input "codellama+gpt4-tokens.jsonl" --output "processed/codellama+gpt4-nfr1-$SEQMODE.json" --with-replacement --np $FULL_NP --nfr 1 --mode $SEQMODE &


done
echo "Launched analysis scripts, waiting for them to finish..."
wait
echo "Done analyzing."
