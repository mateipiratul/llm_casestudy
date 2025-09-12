import os
import re
import json
import argparse
from typing import Dict, List


RESULTS_DIR = 'results'


def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def expected_total_items(models_path='models.json', questions_path='questions.json', system_prompts_path='system_prompts.json') -> int:
    models = load_json(models_path).get('models', [])
    questions = load_json(questions_path).get('questions', [])
    sys_prompts = load_json(system_prompts_path).get('system_prompts', [])

    # Count prompt types per language (yesno + scale)
    prompts_per_lang: Dict[str, int] = {}
    for sp in sys_prompts:
        lang = sp.get('language')
        ptype = (sp.get('prompt_type') or '').lower()
        if ptype not in ('yesno', 'scale'):
            continue
        prompts_per_lang[lang] = prompts_per_lang.get(lang, 0) + 1

    # Sum prompts over the questions list, respecting their language
    total_qp = 0
    for q in questions:
        lang = q.get('language')
        total_qp += prompts_per_lang.get(lang, 0)

    # For each model we expect one result per (question, prompt) combo
    return len(models) * total_qp


def find_run_files(runs: List[int]) -> List[str]:
    files = []
    if not os.path.isdir(RESULTS_DIR):
        return files
    pat = re.compile(r'^(\d+)_test_results\.json$')
    for name in os.listdir(RESULTS_DIR):
        m = pat.match(name)
        if not m:
            continue
        run_id = int(m.group(1))
        if run_id in runs:
            files.append(os.path.join(RESULTS_DIR, name))
    return sorted(files, key=lambda p: int(re.search(r'/(\d+)_test_results\.json$', p).group(1)))


def main():
    ap = argparse.ArgumentParser(description='Check completeness of results (counts only).')
    ap.add_argument('--runs', default='1,2,3,4,5', help='Comma-separated run IDs to check (default: 1,2,3,4,5)')
    args = ap.parse_args()

    runs = [int(x) for x in args.runs.split(',') if x.strip().isdigit()]
    exp = expected_total_items()
    files = find_run_files(runs)

    if not files:
        print('No matching results files found to check.')
        return

    print(f'Expected items per run: {exp}')
    ok_all = True
    for p in files:
        try:
            data = load_json(p)
            res = data.get('results', [])
            n = len(res)
        except Exception as e:
            print(f'- {os.path.basename(p)}: ERROR reading file: {e}')
            ok_all = False
            continue
        status = 'OK' if n == exp else 'MISMATCH'
        if status != 'OK':
            ok_all = False
        print(f'- {os.path.basename(p)}: found {n}/{exp} -> {status}')

    if ok_all:
        print('All specified runs have complete counts.')
    else:
        print('One or more runs have mismatched counts.')


if __name__ == '__main__':
    main()

