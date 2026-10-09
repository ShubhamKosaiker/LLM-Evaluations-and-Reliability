"""Compare prefix removal with the frozen pilot retrieval baseline."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from eval_retriever import GOLDENS_PATH, TOP_K, load_and_validate, score_results

BASELINE_PATH = Path('data/eval/retrieval_baseline.json')
OUTPUT_PATH = Path('data/eval/retrieval_query_wording.json')
METRICS = {'Hit@1': 'hit_at_1', 'Hit@3': 'hit_at_3',
           'Hit@5': 'hit_at_5', 'MRR@5': 'reciprocal_rank_at_5'}


def search_query(query):
    # Only strip the introductory clause; retain all clinical question wording.
    return re.sub(r'^According to the [^,]+,\s*', '', query, count=1)


def preview():
    cases = json.loads(GOLDENS_PATH.read_text(encoding='utf-8'))['cases']
    changed = 0
    for case in cases:
        query = search_query(case['query'])
        changed += query != case['query']
        print(f"{case['id']}: {query}")
    print(f'Preview: {changed}/{len(cases)} queries changed; no retrieval run')


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f'Report already exists: {OUTPUT_PATH}')
    goldens, chunks, golden_hash = load_and_validate()
    baseline_bytes = BASELINE_PATH.read_bytes()
    baseline = json.loads(baseline_bytes)
    if (baseline['goldens_sha256'] != golden_hash or
            baseline['corpus_text_sha256'] != goldens['corpus_text_sha256'] or
            baseline['top_k'] != TOP_K):
        raise ValueError('Dataset, corpus or top_k differs from saved baseline')
    retriever_hash = hashlib.sha256(Path('src/retriever.py').read_bytes()).hexdigest()
    if retriever_hash != baseline['retriever_source_sha256']:
        raise ValueError('Retriever source differs from saved baseline')
    packages = {name: version(name) for name in baseline['package_versions']}
    if packages != baseline['package_versions']:
        raise ValueError('Package versions differ from saved baseline')
    old_cases = {case['id']: case for case in baseline['cases']}
    if set(old_cases) != {case['id'] for case in goldens['cases']}:
        raise ValueError('Baseline case IDs differ from goldens')
    if not Path('data/chroma_db/chroma.sqlite3').is_file():
        raise FileNotFoundError('Existing Chroma database not found')

    from retriever import collection, retrieve
    stored = collection.get(include=['documents'])
    if dict(zip(stored['ids'], stored['documents'])) != chunks:
        raise ValueError('Index differs from the reviewed corpus')

    rows = []
    for case in goldens['cases']:
        old = old_cases[case['id']]
        if (old['query'] != case['query'] or
                old['relevant_chunk_ids'] != case['relevant_chunk_ids']):
            raise ValueError(f"Baseline query or labels changed: {case['id']}")
        # Reproduce the original ranking before comparing the modified query.
        original_ids = retrieve(case['query'], top_k=TOP_K)['ids'][0]
        if original_ids != old['retrieved_ids']:
            raise ValueError(f"Original ranking no longer reproduces: {case['id']}")
        query = search_query(case['query'])
        ids = retrieve(query, top_k=TOP_K)['ids'][0]
        if len(ids) != TOP_K or any(cid not in chunks for cid in ids):
            raise ValueError(f"Invalid retrieval result: {case['id']}")
        scores = score_results(ids, case['relevant_chunk_ids'])
        old_scores = score_results(original_ids, case['relevant_chunk_ids'])
        delta = scores['reciprocal_rank_at_5'] - old_scores['reciprocal_rank_at_5']
        outcome = 'improved' if delta > 0 else 'worsened' if delta < 0 else 'unchanged'
        rows.append({'id': case['id'], 'topic': case['topic'],
                     'original_query': case['query'], 'search_query': query,
                     'query_changed': query != case['query'],
                     'relevant_chunk_ids': case['relevant_chunk_ids'],
                     'baseline_retrieved_ids': original_ids, 'retrieved_ids': ids,
                     'baseline_scores': old_scores, **scores,
                     'reciprocal_rank_delta': delta, 'outcome': outcome})
        print(f"{case['id']}: {old_scores['first_relevant_rank_at_5']} -> "
              f"{scores['first_relevant_rank_at_5']} ({outcome})")
    summary = {metric: sum(row[key] for row in rows) / len(rows)
               for metric, key in METRICS.items()}
    counts = {outcome: sum(row['outcome'] == outcome for row in rows)
              for outcome in ('improved', 'worsened', 'unchanged')}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'experiment': 'Remove leading According to clause only',
              'dataset_status': goldens['status'], 'goldens_sha256': golden_hash,
              'corpus_text_sha256': goldens['corpus_text_sha256'],
              'baseline_report_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
              'retriever_source_sha256': retriever_hash,
              'experiment_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'package_versions': packages, 'original_rankings_reproduced': True,
              'top_k': TOP_K, 'question_count': len(rows),
              'changed_query_count': sum(row['query_changed'] for row in rows),
              'baseline_summary': baseline['summary'], 'summary': summary,
              'rank_changes': counts, 'cases': rows}
    with OUTPUT_PATH.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print('\nBASELINE -> QUERY WORDING EXPERIMENT')
    for metric, value in summary.items():
        print(f"{metric}: {baseline['summary'][metric]:.4f} -> {value:.4f}")
    print(f'Rank changes: {counts}')
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    preview() if args.preview else main()
