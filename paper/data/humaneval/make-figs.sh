#!/bin/bash
# Run all analysis scripts

if [ -z "$(find ./processed -maxdepth 1 -name '*.json' -print -quit)" ]; then
    # Extract the tarball
    echo "Could not make plots; no processed results found. Did you run the analysis script first?"
    exit
fi

mkdir -p ../../figures/humaneval

# Generate all figures
echo "Generating figures..."
for SEQMODE in "batch" "sequential-dfs" ; do
for FT in "pdf" ; do
for MODE in "pass@t" "pass@k-simple" ; do
for MODEL in "gpt3.5" "gpt4" "codellama" ; do
    python3 ../../plotting/scatterplot.py --input processed/$MODEL-full-kn-$SEQMODE.json --output ../../figures/humaneval/$MODEL-scatter-${MODE}-${SEQMODE}.$FT --mode $MODE &
    python3 ../../plotting/heatmap.py --input processed/$MODEL-full-kn-$SEQMODE.json --output ../../figures/humaneval/$MODEL-heatmap-${MODE}-${SEQMODE}.$FT --mode $MODE &
done
python3 ../../plotting/models-plot-var.py --mode $MODE --inputs processed/gpt3.5-nfr1-$SEQMODE.json processed/codellama-nfr1-$SEQMODE.json processed/codellama+gpt3.5-nfr1-$SEQMODE.json processed/codellama+gpt4-nfr1-$SEQMODE.json --labels "GPT-3.5" "Code Llama" "Code Llama+GPT-3.5" "Code Llama+GPT-4" --output ../../figures/humaneval/nfr1-models-plot-var-${MODE}-${SEQMODE}.$FT --legend 'none' &
done
done
done
wait
echo "Done."
