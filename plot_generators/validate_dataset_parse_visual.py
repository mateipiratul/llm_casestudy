import os
import json
from collections import Counter
from typing import Dict, List, Optional
import re

import pandas as pd

OUTDIR = 'analysis_reports'
CSV_OUTDIR = os.path.join(OUTDIR, 'csv')

RUN_FILES = [
    'results/1_test_results.json',
    'results/2_test_results.json',
    'results/3_test_results.json',
    'results/4_test_results.json',
    'results/5_test_results.json',
]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# --- "parse_visual"-style parsing helpers ---
YES_WORDS = {
    'en': ['yes'], 'ro': ['da'], 'hu': ['igen'], 'ru': ['да']
}
NO_WORDS = {
    'en': ['no'], 'ro': ['nu'], 'hu': ['nem'], 'ru': ['нет']
}


def clean_response_parse_visual(response_text: Optional[str]) -> Optional[str]:
    """Mimic parse_visual cleaning but more robust:
    - Remove any <think> ... </think> blocks anywhere (also if unclosed)
    - Remove any [think] ... [/think] blocks (also if unclosed)
    - Remove fenced code blocks ```...```
    - Trim whitespace
    """
    if not isinstance(response_text, str):
        return response_text
    s = response_text
    # Remove fenced code blocks
    s = re.sub(r"```[\s\S]*?```", " ", s)
    # Remove <think> blocks, closed or open-ended
    s = re.sub(r"(?is)<think>.*?(?:</think>|$)", " ", s)
    # Remove [think] blocks, closed or open-ended
    s = re.sub(r"(?is)\[think\].*?(?:\[/think\]|$)", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_yesno_text_parse_visual(resp: Optional[str], lang: Optional[str]) -> Optional[bool]:
    if not isinstance(resp, str) or lang not in YES_WORDS:
        return None
    cleaned_response = resp
    # mimic parse_visual: strip punctuation .,!? and lowercase
    cleaned_response = ''.join(ch for ch in cleaned_response if ch not in '.,!?').lower().strip()
    if cleaned_response in YES_WORDS[lang]:
        return True
    if cleaned_response in NO_WORDS.get(lang, []):
        return False
    return None


def infer_prompt_type(sys_id: Optional[str]) -> Optional[str]:
    if not isinstance(sys_id, str):
        return None
    s = sys_id.lower()
    if 'yesno' in s:
        return 'yesno'
    if 'scale' in s:
        return 'scale'
    return None


def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    ensure_dir(OUTDIR)
    ensure_dir(CSV_OUTDIR)

    models_cfg = load_json('models.json')
    questions_cfg = load_json('questions.json')
    models = models_cfg.get('models', [])
    questions = questions_cfg.get('questions', [])

    # Expected (model, qid, lang, prompt_type) combinations per run
    q_lang_items = [(q['qid'], q['language']) for q in questions]
    expected_keys = set()
    for m in models:
        for (qid, lang) in q_lang_items:
            for p in ('yesno', 'scale'):
                expected_keys.add((m, qid, lang, p))

    summary_rows: List[Dict] = []
    missing_rows: List[Dict] = []
    dup_rows: List[Dict] = []
    parse_rows: List[Dict] = []
    parse_fail_list: List[Dict] = []

    run_coverage_sets = {}

    for run_idx, path in enumerate(RUN_FILES, start=1):
        if not os.path.exists(path):
            summary_rows.append({'run': run_idx, 'file': path, 'status': 'missing_file'})
            continue
        data = load_json(path)
        results = data.get('results', [])

        seen = Counter()
        covered = set()
        unparsable_yes = 0
        unparsable_scale = 0
        total_yes = 0
        total_scale = 0

        for r in results:
            model = r.get('model')
            qid = r.get('question_id')
            lang = r.get('question_language')
            sys_id = r.get('system_prompt_id')
            ptype = infer_prompt_type(sys_id)
            if model is None or qid is None or lang is None or ptype not in ('yesno','scale'):
                continue

            key = (model, qid, lang, ptype)
            seen[key] += 1
            covered.add(key)

            resp_raw = r.get('response')
            resp = clean_response_parse_visual(resp_raw)

            if ptype == 'yesno':
                total_yes += 1
                parsed = parse_yesno_text_parse_visual(resp, lang)
                if parsed is None:
                    unparsable_yes += 1
                    row = {
                        'run': run_idx, 'model': model, 'qid': qid, 'lang': lang,
                        'ptype': ptype, 'system_prompt_id': sys_id,
                        'response': (resp_raw or ''),
                        'response_cleaned': (resp or '')
                    }
                    parse_rows.append(row)
                    parse_fail_list.append(row)
            else:
                total_scale += 1
                # parse_visual numeric parsing uses pandas to_numeric on cleaned response
                try:
                    val = pd.to_numeric(resp, errors='coerce')
                except Exception:
                    val = None
                ok = val is not None and not pd.isna(val) and 1 <= float(val) <= 10
                if not ok:
                    unparsable_scale += 1
                    row = {
                        'run': run_idx, 'model': model, 'qid': qid, 'lang': lang,
                        'ptype': ptype, 'system_prompt_id': sys_id,
                        'response': (resp_raw or ''),
                        'response_cleaned': (resp or '')
                    }
                    parse_rows.append(row)
                    parse_fail_list.append(row)

        run_coverage_sets[run_idx] = covered

        # Missing and duplicates
        missing = sorted(list(expected_keys - covered))
        for (m, q, l, p) in missing:
            missing_rows.append({'run': run_idx, 'model': m, 'qid': q, 'lang': l, 'ptype': p})

        dups = [k for k, c in seen.items() if c > 1]
        for (m, q, l, p) in dups:
            dup_rows.append({'run': run_idx, 'model': m, 'qid': q, 'lang': l, 'ptype': p, 'count': seen[(m,q,l,p)]})

        summary_rows.append({
            'run': run_idx,
            'file': path,
            'items_expected': len(expected_keys),
            'items_found_unique': len(covered),
            'missing_items': len(missing),
            'duplicate_keys': len(dups),
            'unparsable_yesno': unparsable_yes,
            'unparsable_scale': unparsable_scale,
            'total_yesno_rows': total_yes,
            'total_scale_rows': total_scale,
        })

    # Intersection coverage for runs 1–4
    full14 = set.intersection(*(run_coverage_sets.get(i, set()) for i in [1,2,3,4] if i in run_coverage_sets)) if all(i in run_coverage_sets for i in [1,2,3,4]) else set()
    missing_all14 = sorted(list(expected_keys - full14))
    full14_rows = [{'model': m, 'qid': q, 'lang': l, 'ptype': p} for (m,q,l,p) in missing_all14]

    # Save CSVs (with _pv suffix to distinguish from the other validator)
    pd.DataFrame(summary_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_pv_summary_by_run.csv'), index=False)
    pd.DataFrame(missing_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_pv_missing_by_run.csv'), index=False)
    pd.DataFrame(dup_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_pv_duplicates_by_run.csv'), index=False)
    pd.DataFrame(parse_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_pv_parse_failures.csv'), index=False)
    pd.DataFrame(full14_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_pv_missing_in_all_first4_runs.csv'), index=False)

    # Text summary and printed details
    lines = []
    lines.append('Dataset Validation Summary (parse_visual rules)')
    lines.append('--------------------------------')
    for row in summary_rows:
        lines.append(
            f"Run {row['run']}: found {row.get('items_found_unique',0)}/{row.get('items_expected',0)} unique items | "
            f"missing={row.get('missing_items',0)}, dups={row.get('duplicate_keys',0)}, "
            f"unparsable_yesno={row.get('unparsable_yesno',0)}/{row.get('total_yesno_rows',0)}, "
            f"unparsable_scale={row.get('unparsable_scale',0)}/{row.get('total_scale_rows',0)}"
        )
    if full14_rows:
        lines.append('')
        lines.append(f"Missing in the intersection of runs 1–4: {len(full14_rows)} items (see CSV)")
    else:
        lines.append('')
        lines.append('All expected items are present in all of runs 1–4 (intersection check).')

    def _truncate(s: str, n: int = 180) -> str:
        s = (s or '').replace('\n', ' ').replace('\r', ' ')
        return s if len(s) <= n else s[:n] + ' …'

    lines.append('')
    lines.append('Unparsable items (parse_visual rules):')
    if parse_fail_list:
        dfp = pd.DataFrame(parse_fail_list)
        for ptype in ['yesno','scale']:
            sub = dfp[dfp['ptype'] == ptype]
            if sub.empty:
                continue
            lines.append(f"-- {ptype.upper()} parse failures --")
            for run_id, grp in sub.groupby('run'):
                lines.append(f"Run {run_id}: {len(grp)} failures")
                for _, row in grp.sort_values(['model','qid','lang']).iterrows():
                    lines.append(
                        f"  run={row['run']} | model={row['model']} | qid={row['qid']} | lang={row['lang']} | sys={row.get('system_prompt_id','')}\n"
                        f"    cleaned:  {_truncate(row.get('response_cleaned',''))}\n"
                        f"    raw:      {_truncate(row.get('response',''))}"
                    )
    else:
        lines.append('None')

    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUTDIR, 'dataset_validation_parse_visual.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)

    print('Validation (parse_visual) complete. See analysis_reports/dataset_validation_parse_visual.txt and CSVs in analysis_reports/csv/.')


if __name__ == '__main__':
    main()
