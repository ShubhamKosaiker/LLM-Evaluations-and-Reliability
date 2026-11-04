"""Read saved evidence and optionally check its arithmetic. No third-party imports or writes."""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        self.loaded = {}

    def read(self, filename):
        if filename not in self.loaded:
            self.loaded[filename] = json.loads(
                (self.root / 'data' / 'eval' / filename).read_text(encoding='utf-8-sig'))
        return self.loaded[filename]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    if isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), label)
    else:
        require(actual == expected, label)


def mean(rows):
    scores = [r['score'] for r in rows if r['status'] == 'evaluated']
    for score in scores:
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                'Invalid normalized score')
    return sum(scores) / len(scores) if scores else None


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def check_evidence(e):
    collections = [e.read('rag_baseline_pilot.json'), e.read('rag_baseline_validation.json')]
    source_cases = [c for report in collections for c in report['cases']]
    require(len({c['id'] for c in source_cases}) == len(source_cases), 'Duplicate collection IDs')
    sources = {c['id']: c for c in source_cases}
    for report in collections:
        rows = report['cases']
        expected = {'attempted': len(rows), 'successful': sum(c['status'] == 'success' for c in rows),
                    'errors': sum(c['status'] == 'error' for c in rows)}
        equal(report['summary'], expected, 'Collection summary mismatch')

    for filename in ['retrieval_baseline.json', 'retrieval_validation_baseline.json',
                     'retrieval_query_topic.json', 'retrieval_validation_topic.json']:
        report = e.read(filename)
        rows = report['cases']
        equal(report['question_count'], len(rows), filename + ': case count')
        for row in rows:
            ids = row['retrieved_ids'][:5]
            require(len(ids) == len(set(ids)), filename + ': duplicate retrieved IDs')
            rank = next((i for i, cid in enumerate(ids, 1) if cid in row['relevant_chunk_ids']), None)
            equal(row['first_relevant_rank_at_5'], rank, filename + ': first relevant rank')
            for k in [1, 3, 5]:
                equal(row[f'hit_at_{k}'], int(rank is not None and rank <= k), filename + ': hit')
            equal(row['reciprocal_rank_at_5'], 1 / rank if rank else 0.0, filename + ': reciprocal rank')
        for name, key in [('Hit@1', 'hit_at_1'), ('Hit@3', 'hit_at_3'),
                          ('Hit@5', 'hit_at_5'), ('MRR@5', 'reciprocal_rank_at_5')]:
            equal(report['summary'][name], sum(r[key] for r in rows) / len(rows), filename + ': ' + name)

    app = e.read('deepeval_application_baseline_v1.json')
    require({c['id'] for c in app['cases']} == set(sources), 'Application selection differs from collection')
    for case in app['cases']:
        for key in ['query', 'answer', 'reference_answer']:
            equal(case[key], sources[case['id']][key], 'Application saved input differs from collection')
    records = [r for metrics in app['results'].values() for r in metrics.values()]
    equal(app['summary']['metric_slots'], 3 * len(app['cases']), 'Application metric slots')
    for status, key in [('evaluated', 'evaluated'), ('not_applicable', 'not_applicable'),
                        ('skipped_generation_error', 'skipped_generation_error'),
                        ('evaluation_error', 'errors'), ('in_progress', 'in_progress')]:
        equal(app['summary'][key], sum(r['status'] == status for r in records), 'Application status count')
    equal(app['summary']['pending'], 3 * len(app['cases']) - len(records), 'Application pending')
    for cid, metrics in app['results'].items():
        for row in metrics.values():
            if row['status'] == 'evaluated':
                raw = row['judge_outputs']
                require(len(raw) == 1, 'Expected one GEval schema response')
                equal(row['score'], raw[0]['score'] / 10, 'GEval score normalization')
                equal(row['reason'], raw[0]['reason'], 'GEval reason')
                equal(row['passed'], row['score'] >= app['configuration']['threshold'], 'GEval threshold flag')
            elif row['status'] == 'skipped_generation_error':
                require(sources[cid]['status'] == 'error' and row['score'] is None, 'Invalid generation exclusion')
            elif row['status'] == 'not_applicable':
                require(sources[cid]['answer'].strip() == app['configuration']['pure_refusal_literal']
                        and row['score'] is None, 'Invalid refusal exclusion')
    for dataset, validation in [('pilot', False), ('validation', True)]:
        ids = [c['id'] for c in app['cases'] if c['id'].startswith('validation_') == validation]
        for metric, summary in app['summary']['by_dataset'][dataset].items():
            rows = [app['results'][cid][metric] for cid in ids if metric in app['results'].get(cid, {})]
            equal(summary['selected'], len(ids), 'Application selected denominator')
            equal(summary['evaluated'], sum(r['status'] == 'evaluated' for r in rows), 'Application scored denominator')
            equal(summary['mean_score'], mean(rows), 'Application mean')

    quality = e.read('deepeval_contextual_quality_baseline_v1.json')
    for dataset, validation in [('pilot', False), ('validation', True)]:
        ids = [c['id'] for c in quality['cases'] if c['id'].startswith('validation_') == validation]
        for metric, summary in quality['summary']['by_dataset'][dataset].items():
            rows = [quality['results'][cid][metric] for cid in ids if metric in quality['results'].get(cid, {})]
            equal(summary['evaluated'], sum(r['status'] == 'evaluated' for r in rows), 'Context quality count')
            equal(summary['mean_score'], mean(rows), 'Context quality mean')
    for filename in ['deepeval_answer_relevance_baseline_v1.json', 'deepeval_contextual_relevance_baseline_v1.json']:
        report = e.read(filename)
        for dataset, validation in [('pilot', False), ('validation', True)]:
            ids = [c['id'] for c in report['cases'] if c['id'].startswith('validation_') == validation]
            rows = [report['results'][cid] for cid in ids if cid in report['results']]
            summary = report['summary'][dataset]
            equal(summary['evaluated'], sum(r['status'] == 'evaluated' for r in rows), filename + ': count')
            equal(summary['errors'], sum(r['status'] == 'evaluation_error' for r in rows), filename + ': errors')
            equal(summary['pending'], len(ids) - len(rows), filename + ': pending')
            equal(summary['mean_score'], mean(rows), filename + ': mean')

    for filename in ['answer_eval_baseline_pilot_v2_2.json', 'answer_eval_baseline_validation_v2_2.json']:
        report = e.read(filename)
        counts = Counter(c['label'] for c in report['cases'])
        for label, count in report['summary'].items():
            equal(count, counts[label], filename + ': label count')

    for filename in ['scope_safety_baseline_v1.json', 'context_injection_baseline_v1.json',
                     'privacy_toxicity_baseline_v1.json']:
        report = e.read(filename)
        rows = list(report['results'].values())
        require(set(report['results']) == {c['id'] for c in report['fixture']['cases']}, filename + ': case selection')
        equal(report['summary']['collected'], sum(r['status'] == 'collected' for r in rows), filename + ': collected')
        equal(report['summary']['errors'], sum(r['status'] == 'collection_error' for r in rows), filename + ': errors')
        if 'marker_emissions' in report['summary']:
            marker = report['fixture']['forbidden_output_marker']
            actual = sum(marker.casefold() in r['answer'].casefold() for r in rows if r['status'] == 'collected')
            equal(report['summary']['marker_emissions'], actual, 'Injection marker count')
        if 'responses_with_synthetic_identifiers' in report['summary']:
            actual = sum(any(value.casefold() in r['answer'].casefold()
                             for value in report['fixture']['synthetic_identifiers'])
                         for r in rows if r['status'] == 'collected')
            equal(report['summary']['responses_with_synthetic_identifiers'], actual, 'Synthetic identifier count')

    ops = e.read('operational_diagnostics_v1.json')
    successful = [c for c in source_cases if c['status'] == 'success']
    equal(ops['attempts'], len(source_cases), 'Historical collection attempts')
    equal(ops['successful_collections'], len(successful), 'Historical collection successes')
    equal(ops['collection_success_rate'], len(successful) / len(source_cases), 'Historical collection success rate')
    for stage, summary in ops['successful_case_timings'].items():
        values = [sum(c['timing_seconds'][k] for k in ['retrieval', 'generation'])
                  if stage == 'retrieval_plus_generation' else c['timing_seconds'][stage] for c in successful]
        equal(summary['samples'], len(values), 'Historical timing sample count')
        equal(summary['p50_seconds'], percentile(values, .50), 'Historical p50')
        equal(summary['p95_seconds'], percentile(values, .95), 'Historical p95')


def show(e):
    print('Saved RAG evaluation evidence (automated, not verified accuracy)')
    for dataset, baseline, changed in [
            ('pilot', 'retrieval_baseline.json', 'retrieval_query_topic.json'),
            ('validation', 'retrieval_validation_baseline.json', 'retrieval_validation_topic.json')]:
        b, c = e.read(baseline), e.read(changed)
        print(f"Retrieval {dataset}: n={b['question_count']}; Hit@1={b['summary']['Hit@1']:.4f}; "
              f"MRR@5={b['summary']['MRR@5']:.4f}; topic experiment MRR@5={c['summary']['MRR@5']:.4f}")
    app = e.read('deepeval_application_baseline_v1.json')
    for dataset, metrics in app['summary']['by_dataset'].items():
        for name, row in metrics.items():
            print(f"Application {dataset} / {name}: mean={row['mean_score']:.4f}; "
                  f"scored={row['evaluated']}; N/A={row['not_applicable']}; generation skips={row['skipped_generation_error']}")
    context = e.read('deepeval_contextual_relevance_baseline_v1.json')['summary']
    print('PARTIAL: contextual relevance: ' + '; '.join(
        f"{name}: scored={r['evaluated']}, errors={r['errors']}, pending={r['pending']}" for name, r in context.items()))
    ops = e.read('operational_diagnostics_v1.json')
    timing = ops['successful_case_timings']['retrieval_plus_generation']
    print(f"Historical collection: {ops['successful_collections']}/{ops['attempts']} successful; "
          f"successful-stage-sum p50={timing['p50_seconds']:.3f}s, p95={timing['p95_seconds']:.3f}s")
    print('LIMITS: judge errors and qualifier failures remain; timing is not a load benchmark; cost/retries unknown.')
    print('See docs/RESULTS.md and data/eval/pilot_review.md for interpretation and safety findings.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Repository root')
    parser.add_argument('--check', action='store_true', help='Check saved counts, scores and arithmetic without evaluating models')
    args = parser.parse_args()
    try:
        evidence = Evidence(args.root)
        if args.check:
            check_evidence(evidence)
            print('PASS: saved-report consistency checks; no model calls or report writes.')
        show(evidence)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
