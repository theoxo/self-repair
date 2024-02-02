import argparse
from collections import defaultdict
import json
import random
from numpy import mean, std, array
import sys
import os

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='The file containing the repair results')
    parser.add_argument('--output', required=False, default=None, type=str, help='The file to write the results to.')
    parser.add_argument('--input-mode', required=False, type=str, default='all-chatgpt', choices=['line-by-line', 'full-json', 'all-chatgpt'])
    parser.add_argument('--repair-key', required=False, type=str, default='repairs')
    parser.add_argument('--no-baseline', default=False, action='store_true', help='Whether to calculate pass rates without repair')
    parser.add_argument('--np', required=False, nargs="+", type=int, default=[1, 2, 5, 10, 25])
    parser.add_argument('--nfr', required=False, nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument('--seed', required=False, type=int, default=2023)
    parser.add_argument('--split-by-difficulty', default=False, action='store_true', help='Whether to split the results by difficulty')
    parser.add_argument('--num-samples', required=False, type=int, default=1000, help='The number of samples to use for each np, nfr')
    parser.add_argument('--count-single-repair-as-passing', default=False, action='store_true', help='Whether to count a single "repair" as passing')
    parser.add_argument('--with-replacement', default=False, action='store_true', help='Whether to sample with replacement')
    parser.add_argument('--verbose', default=False, action='store_true', help='Whether to print verbose output')
    parser.add_argument('--mode', type=str, default='batch', choices=['batch', 'sequential-bfs', 'sequential-dfs'], help='Whether to sample a batch of base programs and repairs or sample them sequentially')
    return parser

def get_difficulty(prob_path):
    with open(os.path.join(os.environ['APPS_DIR'], prob_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
        return metadata["difficulty"]

def search(args, progs, suffix=None):

    def _print(s, **kwargs):
        if args.verbose:
            print(s, **kwargs)

    if args.with_replacement:
        sampler = random.choices
    else:
        sampler = random.sample

    avg_pass_rates = {}
    avg_tokens = {}
    avg_samples = {}
    num_samples = args.num_samples
    params = [] if args.no_baseline else [(i, 0) for i in range(1, 51)]
    params.extend([(np, nfr) for np in args.np for nfr in args.nfr])
    _print('Beginning grid search')
    for np, nfr in params:
        _print('-' * 80)
        _print(f'np={np}, nfr={nfr}')
        _print('-' * 80)
        pass_rates = []
        tokens = []
        samples = []

        for _ in range(num_samples):

            num_passed = 0
            tokens_generated = 0
            samples_drawn = 0

            for task, base_progs in progs.items():

                base_progs = base_progs['completions']

                # first sample np base programs
                bps = sampler(base_progs, k=np)

                if args.mode == 'sequential-dfs':
                    # if we're doing dfs, we want to sample the first base program first
                    _loop_outer = True
                    for bp in bps:
                        if not _loop_outer: break
                        tokens_generated += bp['tokens_generated']
                        samples_drawn += 1
                        if bp['binary'] or bp['fault'] == 'passed':
                            num_passed += 1
                            break
                        repairs = sampler(bp[args.repair_key], k=nfr)
                        for r in repairs:
                            tokens_generated += r['tokens_generated']
                            samples_drawn += 1
                            if r['binary']:
                                num_passed += 1
                                _loop_outer = False
                                break
                else:
                    # record the number of tokens generated
                    if args.mode == 'batch':
                        tokens_generated += sum([bp['tokens_generated'] for bp in bps])
                        samples_drawn += np
                    elif args.mode == 'sequential-bfs':
                        # sum up tokens generated for each base program up to and including the first passing one
                        first_passing_idx = next((i for i, bp in enumerate(bps) if bp['binary'] or bp['fault'] == 'passed'), -1)
                        if first_passing_idx >= 0:
                            tokens_generated += sum([bp['tokens_generated'] for bp in bps[:first_passing_idx+1]])
                            samples_drawn += first_passing_idx + 1
                        else:
                            tokens_generated += sum([bp['tokens_generated'] for bp in bps])
                            samples_drawn += np

                    if any([bp['binary'] or bp['fault'] == 'passed' for bp in bps]):
                        # if any base program passes, we consider this a success and move on
                        num_passed += 1
                    elif args.count_single_repair_as_passing and any([len(bp[args.repair_key]) == 1 for bp in bps]):
                        num_passed += 1
                    elif nfr > 0:
                        
                        # get np * nfr repair candidates
                        repairs = (array([sampler(bp[args.repair_key], k=nfr) for bp in bps])).flatten()
                        assert len(repairs) == np * nfr, f'{len(repairs)} {nfr}'

                        if args.mode == 'batch':
                            # record the number of tokens generated
                            tokens_generated += sum([r['tokens_generated'] for r in repairs])
                            samples_drawn += np * nfr
                        else:
                            assert args.mode == 'sequential-bfs'
                            # sum up tokens generated for each repair up to and including the first passing one
                            first_passing_idx = next((i for i, r in enumerate(repairs) if r['binary']), -1)
                            if first_passing_idx >= 0:
                                tokens_generated += sum([r['tokens_generated'] for r in repairs[:first_passing_idx+1]])
                                samples_drawn += first_passing_idx + 1
                            else:
                                tokens_generated += sum([r['tokens_generated'] for r in repairs])
                                samples_drawn += np * nfr

                        # record whether any of them passed
                        num_passed += any([r['binary'] for r in repairs])

            # average over the tasks
            pass_rate = num_passed / len(progs)
            avg_tokens_generated = tokens_generated / len(progs)
            avg_num_samples =  samples_drawn / len(progs)

            pass_rates.append(pass_rate)
            tokens.append(avg_tokens_generated)
            samples.append(avg_num_samples)

        # average over the seeds
        avg_pass_rates[(np, nfr)] = (mean(pass_rates), std(pass_rates))
        avg_tokens[(np, nfr)] = (mean(tokens), std(tokens))
        avg_samples[(np, nfr)] = (mean(samples), std(samples))

    if args.output:
        if suffix:
            output = args.output.replace('.json', f'-{suffix}.json')
        else:
            output = args.output
        with open(output, 'w') as f:
            json.dump({str((np, nfr)): (avg_pass_rates[(np, nfr)], avg_tokens[(np, nfr)], avg_samples[(np, nfr)]) for (np, nfr) in avg_pass_rates}, f, indent=2, sort_keys=True)


    if args.verbose:
        from prettytable import PrettyTable
        pt = PrettyTable(['np', 'nfr', 'pass rate', 'tokens generated'])
        for (np, nfr) in sorted(avg_pass_rates, key=lambda x: avg_tokens[x], reverse=False):
            pt.add_row([np, nfr,
                        f"{avg_pass_rates[(np, nfr)][0]:.4f} +- {avg_pass_rates[(np, nfr)][1]:.4f} ({avg_pass_rates[(np, nfr)][1]/avg_pass_rates[((np, nfr))][0]:.4f}%)",
                        f"{avg_tokens[(np, nfr)][0]:.4f} +- {avg_tokens[(np, nfr)][1]:.4f} ({avg_tokens[(np, nfr)][1]/avg_tokens[((np, nfr))][0]:.4f}%)",
            ])
        print(pt)


def main(args):

    random.seed(args.seed)

    progs = {}
    if args.input_mode == 'line-by-line':
        with open(args.input, 'r') as f:
            for line in f.readlines():
                prog = json.loads(line)
                assert len(prog) == 1
                task = list(prog.keys())[0]
                assert task not in progs, f'Already have data for {task}'
                progs[task] = prog[task]
    elif args.input_mode == 'all-chatgpt':
        with open(args.input, 'r') as f:
            for line in f.readlines():
                prog = json.loads(line)
                task = prog.get('prob_path', prog.get('task_id'))
                assert task not in progs, f'Already have data for {task}'
                progs[task] = prog
    else:
        assert args.input_mode == 'full-json'
        with open(args.input, 'r') as f:
            progs = json.load(f)


    if args.split_by_difficulty:
        if args.verbose:
            print('Splitting by difficulty')
        by_diff = defaultdict(dict)
        for prog in progs:
            by_diff[get_difficulty(prog)][prog] = progs[prog]
        
        for diff in by_diff:
            search(args, by_diff[diff], suffix=diff)
    else:
        search(args, progs)



if __name__ == '__main__':
    main(build_args().parse_args())
