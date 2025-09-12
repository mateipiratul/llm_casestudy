import os
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
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


def infer_prompt_type(sys_id: Optional[str]) -> Optional[str]:
    if not isinstance(sys_id, str):
        return None
    s = sys_id.lower()
    if 'yesno' in s:
        return 'yesno'
    if 'scale' in s:
        return 'scale'
    return None


def _clean_text(resp: str) -> str:
    if not isinstance(resp, str):
        return ''
    s = resp
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"(?is)<think>.*?(?:</think>|$)", " ", s)
    s = re.sub(r"(?is)\[think\].*?(?:\[/think\]|$)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.search(r"(?is)(?:final\s*answer|answer|final|prediction|output)\s*[:\-]\s*(.+)$", s)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in re.split(r"[\n\r.!?]", s) if p.strip()]
    if parts:
        return parts[-1]
    return s


def _normalize_ru_homoglyphs(text: str) -> str:
    trans = str.maketrans({
        'A': '\u0410', 'a': '\u0430',
        'E': '\u0415', 'e': '\u0435',
        'T': '\u0422', 't': '\u0442',
        'H': '\u041D', 'h': '\u043D',
        'O': '\u041E', 'o': '\u043E',
        'P': '\u0420', 'p': '\u0440',
        'C': '\u0421', 'c': '\u0441',
        'Y': '\u0423', 'y': '\u0443',
        'X': '\u0425', 'x': '\u0445',
    })
    return text.translate(trans)


def parse_yesno(resp: str, lang: str) -> Optional[bool]:
    YES = {"en": ["yes"], "ro": ["da"], "hu": ["igen"], "ru": ["да"]}
    NO = {"en": ["no"], "ro": ["nu"], "hu": ["nem"], "ru": ["нет"]}
    s = _clean_text(resp)
    if not s:
        return None
    ys = YES.get(lang, [])
    ns = NO.get(lang, [])
    if not ys and not ns:
        return None
    if lang == 'ru':
        s = _normalize_ru_homoglyphs(s)
    patt = r"(?i)(?P<tok>" + "|".join(map(re.escape, ys + ns)) + r")"
    last = None
    for m in re.finditer(patt, s):
        last = m.group('tok').lower()
    if last is None:
        return None
    if last in [t.lower() for t in ys]:
        return True
    if last in [t.lower() for t in ns]:
        return False
    return None


def parse_scale(resp: str) -> Optional[int]:
    s = _clean_text(resp)
    if not s:
        return None
    s = re.sub(r"\b\d\s*[\-–]\s*\d{1,2}\b", " ", s)
    matches = list(re.finditer(r"\b(10|[1-9])\b", s))
    if not matches:
        return None
    v = int(matches[-1].group(1))
    return v if 1 <= v <= 10 else None


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

    # Expected combinations per run
    q_lang_items = [(q['qid'], q['language']) for q in questions]
    # For each (qid,lang) we expect 2 prompt types: yesno + scale
    expected_keys = set()
    for m in models:
        for (qid, lang) in q_lang_items:
            for p in ('yesno', 'scale'):
                expected_keys.add((m, qid, lang, p))

    # Per-run stats
    summary_rows = []
    missing_rows = []
    dup_rows = []
    parse_rows = []

    run_coverage_sets = {}
    # Keep detailed lists per type for console printing at the end
    parse_fail_list: List[Dict] = []

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
            ptype = infer_prompt_type(r.get('system_prompt_id'))
            if model is None or qid is None or lang is None or ptype not in ('yesno','scale'):
                continue
            key = (model, qid, lang, ptype)
            seen[key] += 1
            covered.add(key)
            resp = (r.get('response') or '').strip()
            if ptype == 'yesno':
                total_yes += 1
                if parse_yesno(resp, lang) is None:
                    unparsable_yes += 1
                    row = {
                        'run': run_idx, 'model': model, 'qid': qid, 'lang': lang,
                        'ptype': ptype, 'system_prompt_id': r.get('system_prompt_id'),
                        'response': resp
                    }
                    parse_rows.append(row)
                    parse_fail_list.append(row)
            else:
                total_scale += 1
                if parse_scale(resp) is None:
                    unparsable_scale += 1
                    row = {
                        'run': run_idx, 'model': model, 'qid': qid, 'lang': lang,
                        'ptype': ptype, 'system_prompt_id': r.get('system_prompt_id'),
                        'response': resp
                    }
                    parse_rows.append(row)
                    parse_fail_list.append(row)

        run_coverage_sets[run_idx] = covered

        # Missing and duplicates
        exp_for_run = expected_keys
        missing = sorted(list(exp_for_run - covered))
        for (m, q, l, p) in missing:
            missing_rows.append({'run': run_idx, 'model': m, 'qid': q, 'lang': l, 'ptype': p})

        dups = [k for k, c in seen.items() if c > 1]
        for (m, q, l, p) in dups:
            dup_rows.append({'run': run_idx, 'model': m, 'qid': q, 'lang': l, 'ptype': p, 'count': seen[(m,q,l,p)]})

        summary_rows.append({
            'run': run_idx,
            'file': path,
            'items_expected': len(exp_for_run),
            'items_found_unique': len(covered),
            'missing_items': len(missing),
            'duplicate_keys': len(dups),
            'unparsable_yesno': unparsable_yes,
            'unparsable_scale': unparsable_scale,
            'total_yesno_rows': total_yes,
            'total_scale_rows': total_scale,
        })

    # Cross-run coverage: do all runs 1–4 have the same full set?
    full14 = set.intersection(*(run_coverage_sets.get(i, set()) for i in [1,2,3,4] if i in run_coverage_sets)) if all(i in run_coverage_sets for i in [1,2,3,4]) else set()
    missing_all14 = sorted(list(expected_keys - full14))
    full14_rows = [{'model': m, 'qid': q, 'lang': l, 'ptype': p} for (m,q,l,p) in missing_all14]

    # Save CSVs and a human-readable summary
    pd.DataFrame(summary_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_summary_by_run.csv'), index=False)
    pd.DataFrame(missing_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_missing_by_run.csv'), index=False)
    pd.DataFrame(dup_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_duplicates_by_run.csv'), index=False)
    pd.DataFrame(parse_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_parse_failures.csv'), index=False)
    pd.DataFrame(full14_rows).to_csv(os.path.join(CSV_OUTDIR, 'validation_missing_in_all_first4_runs.csv'), index=False)

    # Text summary
    lines = []
    lines.append('Dataset Validation Summary')
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

    # Print details of parse failures to console and append to summary file
    def _truncate(s: str, n: int = 180) -> str:
        s = (s or '').replace('\n', ' ').replace('\r', ' ')
        return s if len(s) <= n else s[:n] + ' …'

    lines.append('')
    lines.append('Unparsable items (printed below):')
    # Group by ptype then run for readability
    pf_df = pd.DataFrame(parse_fail_list)
    if not pf_df.empty:
        for ptype in ['yesno','scale']:
            sub_p = pf_df[pf_df['ptype'] == ptype]
            if sub_p.empty:
                continue
            lines.append(f"-- {ptype.upper()} parse failures --")
            for run_id, sub_r in sub_p.groupby('run'):
                lines.append(f"Run {run_id}: {len(sub_r)} failures")
                for _, row in sub_r.sort_values(['model','qid','lang']).iterrows():
                    lines.append(
                        f"  run={row['run']} | model={row['model']} | qid={row['qid']} | lang={row['lang']} | sys={row.get('system_prompt_id','')}\n"
                        f"    response: {_truncate(row.get('response',''))}"
                    )
    else:
        lines.append('None')

    text_out = '\n'.join(lines) + '\n'
    with open(os.path.join(OUTDIR, 'dataset_validation.txt'), 'w', encoding='utf-8') as f:
        f.write(text_out)
    # Also print to stdout so the user can see immediately
    print(text_out)

    print('Validation complete. See analysis_reports/dataset_validation.txt and CSVs in analysis_reports/csv/.')


if __name__ == '__main__':
    main()
