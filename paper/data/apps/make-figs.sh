#!/bin/bash
# Run all analysis scripts

if [ -z "$(find ./processed -maxdepth 1 -name '*.json' -print -quit)" ]; then
    # Extract the tarball
    echo "Could not make plots; no processed results found. Did you run the analysis script first?"
    exit
fi

mkdir -p ../../figures/apps

# Generate all figures
for SEQMODE in "batch" "sequential-dfs" ; do
for FT in "pdf" ; do
for MODE in "pass@t" "pass@k-simple" ; do
for MODEL in "gpt3.5" "gpt4" "codellama" "codellama+gpt3.5" "gpt3.5+gpt4" ; do
    python3 ../../plotting/scatterplot.py --input processed/$MODEL-full-kn-$SEQMODE.json --output ../../figures/apps/$MODEL-scatter-${MODE}-${SEQMODE}.$FT --mode $MODE &
    python3 ../../plotting/heatmap.py --input processed/$MODEL-full-kn-$SEQMODE.json --output ../../figures/apps/$MODEL-heatmap-${MODE}-${SEQMODE}.$FT --mode $MODE &
    for DIFF in "introductory" "interview" "competition" ; do
        python3 ../../plotting/scatterplot.py --input processed/$MODEL-full-kn-$SEQMODE-$DIFF.json --output ../../figures/apps/$MODEL-scatter-${MODE}-${SEQMODE}-$DIFF.$FT --mode $MODE &
        python3 ../../plotting/heatmap.py --input processed/$MODEL-full-kn-$SEQMODE-$DIFF.json --output ../../figures/apps/$MODEL-heatmap-${MODE}-${SEQMODE}-$DIFF.$FT --mode $MODE &
    done
done
python3 ../../plotting/models-plot-var.py --legend 'full' --mode $MODE --inputs processed/gpt4-nfr1-$SEQMODE.json processed/gpt3.5-nfr1-$SEQMODE.json processed/codellama-nfr1-$SEQMODE.json processed/gpt3.5+gpt4-nfr1-$SEQMODE.json processed/codellama+gpt3.5-nfr1-$SEQMODE.json --labels "GPT-4" "GPT-3.5" "Code Llama" "GPT-3.5+GPT-4" "Code Llama+GPT-3.5" --output ../../figures/apps/nfr1-models-plot-var-${MODE}-${SEQMODE}.$FT &
for DIFF in "introductory" "interview" "competition" ; do
    python3 ../../plotting/models-plot-var.py --mode $MODE --inputs processed/gpt4-nfr1-$SEQMODE-$DIFF.json processed/gpt3.5-nfr1-$SEQMODE-$DIFF.json processed/codellama-nfr1-$SEQMODE-$DIFF.json processed/gpt3.5+gpt4-nfr1-$SEQMODE-$DIFF.json processed/codellama+gpt3.5-nfr1-$SEQMODE-$DIFF.json --labels "GPT-4" "GPT-3.5" "Code Llama" "GPT-3.5+GPT-4" "Code Llama+GPT-3.5" --output ../../figures/apps/nfr1-models-plot-var-${MODE}-${SEQMODE}-${DIFF}.$FT --legend 'partial' &
done
done
done
done
wait
echo "Done."
