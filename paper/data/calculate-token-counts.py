import argparse
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from tiktoken import get_encoding

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, type=str,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--arch",
        required=True,
        type=str,
        choices=['gpt', 'llama']
    )
    parser.add_argument(
        "--input-mode",
        required=False,
        type=str,
        default='line-by-line',
        choices=['line-by-line', 'full-json'],
    )
    return parser

args = build_args().parse_args()

if args.arch == 'gpt':
    encoder = get_encoding('cl100k_base')
elif args.arch == 'llama':
    MODEL_NAME = "codellama/CodeLlama-13b-Instruct-hf"
    encoder = AutoTokenizer.from_pretrained(MODEL_NAME)

with open(args.input, "r") as f:
    if args.input_mode == 'line-by-line':
        for line in tqdm(f):
            data = json.loads(line)
            for c in data['completions']:
                c.pop('avg_tokens_per_completion', None)
                c.pop('avg_tokens_per_repair', None)  # legacy and inaccurate
                c['tokens_generated'] = len(encoder.encode(c['executed_completion']))
                if c['binary']:
                    _v = c.get('repairs')
                    assert _v is None or _v == []
                    continue 
                if len(c['repairs']) == 1:
                    # legacy way of denoting passing on re-execution
                    c['binary'] = True
                    c['repairs'] = None
                    continue
                for r in (c['repairs'] or []):
                    explanation = r['explanation']
                    if r['repaired_program'] and r['repaired_program'] in explanation:
                        explanation = explanation.split(r['repaired_program'])[0]
                    r['tokens_generated'] = len(encoder.encode(r['repaired_program']))\
                        + len(encoder.encode(explanation))
                    
            with open(args.output, "a") as f:
                f.write(json.dumps(data) + '\n')
    else:
        data = json.load(f)
        for task in tqdm(sorted(data.keys(), key=lambda x: int(x.split('/')[-1]))):
            for c in data[task]['completions']:
                c['tokens_generated'] = len(encoder.encode(c['executed_completion']))
                if c['binary']:
                    assert c.get('repairs') is None
                    continue 
                if len(c['repairs']) == 1:
                    # legacy way of denoting passing on re-execution
                    c['binary'] = True
                    c['repairs'] = None
                    continue
                for r in (c['repairs'] or []):
                    explanation = r['explanation']
                    repaired_program = r.get('repaired_program', r.pop('program'))
                    r['repaired_program'] = repaired_program
                    if r['repaired_program'] and r['repaired_program'] in explanation:
                        explanation = explanation.split(r['repaired_program'])[0]
                    r['tokens_generated'] = len(encoder.encode(r['repaired_program']))\
                        + len(encoder.encode(explanation))
            with open(args.output, "a") as f:
                f.write(json.dumps(data[task]) + '\n')



