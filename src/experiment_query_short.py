"""Compare fixed, manually shortened queries with frozen retrieval baselines.

Development diagnostic only: both datasets have already been inspected.
Rewrites use the questions, not reference answers; this is not an automatic
query rewriting system or an independent generalization test."""
import argparse
import hashlib
import json
import eval_retriever as evaluator
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from eval_retriever import GOLDENS_PATH, TOP_K, load_and_validate, score_results

BASELINE_PATH = Path('data/eval/retrieval_baseline.json')
OUTPUT_PATH = Path('data/eval/retrieval_query_wording.json')
METRICS = {'Hit@1': 'hit_at_1', 'Hit@3': 'hit_at_3',
           'Hit@5': 'hit_at_5', 'MRR@5': 'reciprocal_rank_at_5'}


SHORT_QUERIES = {'cough_01': 'When is a chest X-ray recommended for a patient with cough?', 'cough_02': 'What cough red flags suggest serious pathology?', 'cough_03': 'What are the most common causes of acute cough?', 'cough_04': 'What are the most common causes of chronic cough?', 'cough_05': 'Are antibiotics usually indicated for acute bronchitis, and what is the exception?', 'headache_01': 'What are the location, severity, and sensation of tension headaches?', 'headache_02': 'How long do tension headaches and migraines typically last?', 'headache_03': 'What visual disturbances occur during migraine aura?', 'headache_04': 'What does thunderclap mean about headache onset?', 'headache_05': 'When do obstructive sleep apnea headaches tend to occur?', 'dyspnea_01': 'What minimum objective assessment is initially recommended for acute dyspnea?', 'dyspnea_02': 'Does normal oxygen saturation rule out serious causes of shortness of breath?', 'acute_abdominal_01': 'Where does appendicitis pain classically begin and move?', 'acute_abdominal_02': 'What posture relieves pancreatitis pain?', 'chronic_abdominal_01': 'What duration defines chronic abdominal pain, including recurrent episodes?', 'hemoptysis_01': 'What are the most common causes of hemoptysis in developed countries?', 'fever_01': 'What temperature counts as a fever in children?', 'cold_flu_01': 'How do usual influenza symptoms differ from common cold symptoms?', 'rash_01': 'What symptoms accompanying an adult rash require emergency medical care?', 'dandruff_01': 'Is dandruff caused by poor hygiene, and how does infrequent shampooing affect it?', 'validation_cough_01': 'What medication-class switch is described for suspected ACE inhibitor cough?', 'validation_headache_01': 'What does secondary headache mean?', 'validation_dyspnea_01': 'How does respiratory compensation respond to metabolic acidosis in acute dyspnea?', 'validation_acute_abdominal_01': "What physical examination finding is Murphy's sign?", 'validation_hemoptysis_01': 'Does a normal chest CT rule out cancer in hemoptysis?', 'validation_chronic_abdominal_01': 'What test diagnoses gastroparesis in chronic abdominal pain?', 'validation_fever_01': 'What are the signs of dehydration in children with fever?', 'validation_cold_flu_01': 'What self-care measures are recommended for a common cold?', 'validation_rash_01': 'What signs suggest an infected rash in adults?', 'validation_dandruff_01': 'What active ingredients are suggested for dandruff shampoo?'}
ACTIVE_CASES = []


def search_query(query):
    case = next(c for c in ACTIVE_CASES if c["query"] == query)
    return SHORT_QUERIES[case["id"]]


def configure(dataset):
    global GOLDENS_PATH, BASELINE_PATH, OUTPUT_PATH, ACTIVE_CASES
    stem = "retrieval" if dataset == "pilot" else "retrieval_validation"
    GOLDENS_PATH = Path(f"data/eval/{stem}_goldens.json")
    BASELINE_PATH = Path(f"data/eval/{stem}_baseline.json")
    OUTPUT_PATH = Path(f"data/eval/{stem}_query_short.json")
    evaluator.GOLDENS_PATH = GOLDENS_PATH
    goldens, _, _ = load_and_validate()
    ACTIVE_CASES = goldens["cases"]
    missing = {c["id"] for c in ACTIVE_CASES} - SHORT_QUERIES.keys()
    if missing:
        raise ValueError(f"Missing query rewrites: {sorted(missing)}")


def preview():
    for case in ACTIVE_CASES:
        print(f"{case['id']}:\n  Original: {case['query']}\n  Short:    {search_query(case['query'])}")
    print(f"Preview: {len(ACTIVE_CASES)} questions; no retrieval run")


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
            print("Query:", case["query"])
            print("Saved ranking:", old["retrieved_ids"])
            print("Current ranking:", original_ids)
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
              'experiment': 'Fixed manually shortened queries; development diagnostic',
              'evaluation_role': 'development; previously inspected datasets',
              'rewrite_method': 'fixed manual question-only rewrites',
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
    parser.add_argument('dataset', choices=['pilot', 'validation'])
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    configure(args.dataset)
    preview() if args.preview else main()
