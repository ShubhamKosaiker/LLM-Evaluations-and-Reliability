"""Evaluate original questions with BM25 keyword search plus dense search.

Development experiment on previously inspected datasets. Fixed settings,
no manual query rewrites, no new dependencies, no index writes.
BM25 uses positive log-IDF and exact token lengths, not a Lucene backend.
References:
https://lucene.apache.org/core/9_12_3/core/org/apache/lucene/search/similarities/BM25Similarity.html
https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion"""
import argparse
import hashlib
import json
import re
import math
from collections import Counter, defaultdict
import eval_retriever as evaluator
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from eval_retriever import GOLDENS_PATH, TOP_K, load_and_validate, score_results

BASELINE_PATH = Path('data/eval/retrieval_baseline.json')
OUTPUT_PATH = Path('data/eval/retrieval_query_wording.json')
METRICS = {'Hit@1': 'hit_at_1', 'Hit@3': 'hit_at_3',
           'Hit@5': 'hit_at_5', 'MRR@5': 'reciprocal_rank_at_5'}


CANDIDATE_K = 20
RRF_CONSTANT = 60
BM25_K1 = 1.2
BM25_B = 0.75


def tokenize(text):
    # Identical preprocessing for corpus and questions. No stopword removal.
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordIndex:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("Empty keyword corpus")
        self.ids = sorted(chunks)
        self.counts = {cid: Counter(tokenize(chunks[cid])) for cid in self.ids}
        self.lengths = {cid: sum(self.counts[cid].values()) for cid in self.ids}
        self.avg_length = sum(self.lengths.values()) / len(self.ids)
        if self.avg_length == 0:
            raise ValueError("Keyword corpus has no tokens")
        df = Counter(term for cid in self.ids for term in self.counts[cid])
        self.idf = {term: math.log1p((len(self.ids) - count + 0.5) / (count + 0.5))
                    for term, count in df.items()}

    def rank(self, query, limit=CANDIDATE_K):
        terms = set(tokenize(query))  # Repeated query terms count once.
        scores = {}
        for cid in self.ids:
            length_norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[cid] / self.avg_length)
            score = 0.0
            for term in sorted(terms):
                tf = self.counts[cid].get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (BM25_K1 + 1) / (tf + length_norm)
            if score > 0:
                scores[cid] = score
        ids = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
        return ids, {cid: scores[cid] for cid in ids}


def fuse(dense_ids, keyword_ids, limit=TOP_K):
    scores = defaultdict(float)
    for ranking in (dense_ids, keyword_ids):
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate IDs in candidate list")
        for rank, cid in enumerate(ranking, 1):
            scores[cid] += 1.0 / (RRF_CONSTANT + rank)
    # Equal weight; exact ties resolved by chunk ID for reproducibility.
    ids = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    return ids, {cid: scores[cid] for cid in ids}


def configure(dataset):
    global GOLDENS_PATH, BASELINE_PATH, OUTPUT_PATH
    stem = "retrieval" if dataset == "pilot" else "retrieval_validation"
    GOLDENS_PATH = Path(f"data/eval/{stem}_goldens.json")
    BASELINE_PATH = Path(f"data/eval/{stem}_baseline.json")
    OUTPUT_PATH = Path(f"data/eval/{stem}_hybrid.json")
    evaluator.GOLDENS_PATH = GOLDENS_PATH


def self_test():
    index = KeywordIndex({"a": "rare common", "b": "common common", "c": "other other"})
    ids, scores = index.rank("RARE")
    expected = math.log(1 + 2.5 / 1.5)
    assert ids == ["a"] and math.isclose(scores["a"], expected)
    assert index.rank("absent")[0] == []
    assert index.rank("rare rare") == index.rank("rare")
    ids, _ = KeywordIndex({"a": "rare rare", "b": "rare other"}).rank("rare")
    assert ids == ["a", "b"]
    ids, _ = KeywordIndex({"z": "same", "a": "same"}).rank("same")
    assert ids == ["a", "z"]
    ids, scores = fuse(["a", "b", "c"], ["b", "d", "a"])
    assert ids[0] == "b" and len(ids) == len(set(ids))
    assert math.isclose(scores["b"], 1 / 62 + 1 / 61)
    assert fuse(["a", "b"], [])[0] == ["a", "b"]
    try:
        fuse(["a", "a"], [])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate candidate IDs were accepted")
    print("PASS: BM25 scoring, repeated/missing terms, stable ties, RRF and duplicate rejection")


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

    keyword_index = KeywordIndex(chunks)
    rows = []
    for case in goldens['cases']:
        old = old_cases[case['id']]
        if (old['query'] != case['query'] or
                old['relevant_chunk_ids'] != case['relevant_chunk_ids']):
            raise ValueError(f"Baseline query or labels changed: {case['id']}")
        # Reproduce the original ranking before comparing the modified query.
        original_ids = retrieve(case['query'], top_k=TOP_K)['ids'][0]
        if original_ids != old['retrieved_ids']:
            raise ValueError(f"Original ranking no longer reproduces: {case['id']}\n"
                             f"Saved: {old['retrieved_ids']}\nCurrent: {original_ids}")
        query = case['query']
        dense_ids = retrieve(query, top_k=min(CANDIDATE_K, len(chunks)))['ids'][0]
        if (len(dense_ids) != min(CANDIDATE_K, len(chunks)) or
                len(dense_ids) != len(set(dense_ids)) or
                any(cid not in chunks for cid in dense_ids)):
            raise ValueError(f"Invalid dense candidates: {case['id']}")
        keyword_ids, keyword_scores = keyword_index.rank(query)
        ids, fusion_scores = fuse(dense_ids, keyword_ids)
        keyword_metrics = score_results(keyword_ids, case['relevant_chunk_ids'])
        if len(ids) != TOP_K or any(cid not in chunks for cid in ids):
            raise ValueError(f"Invalid retrieval result: {case['id']}")
        scores = score_results(ids, case['relevant_chunk_ids'])
        old_scores = score_results(original_ids, case['relevant_chunk_ids'])
        delta = scores['reciprocal_rank_at_5'] - old_scores['reciprocal_rank_at_5']
        outcome = 'improved' if delta > 0 else 'worsened' if delta < 0 else 'unchanged'
        rows.append({'id': case['id'], 'topic': case['topic'],
                     'original_query': case['query'], 'search_query': query,
                     'query_changed': False,
                     'dense_candidate_ids': dense_ids,
                     'keyword_candidate_ids': keyword_ids, 'keyword_scores': keyword_scores,
                     'keyword_metrics': keyword_metrics, 'fusion_scores': fusion_scores,
                     'relevant_chunk_ids': case['relevant_chunk_ids'],
                     'baseline_retrieved_ids': original_ids, 'retrieved_ids': ids,
                     'baseline_scores': old_scores, **scores,
                     'reciprocal_rank_delta': delta, 'outcome': outcome})
        print(f"{case['id']}: {old_scores['first_relevant_rank_at_5']} -> "
              f"{scores['first_relevant_rank_at_5']} ({outcome})")
    summary = {metric: sum(row[key] for row in rows) / len(rows)
               for metric, key in METRICS.items()}
    keyword_summary = {metric: sum(row['keyword_metrics'][key] for row in rows) / len(rows)
                       for metric, key in METRICS.items()}
    counts = {outcome: sum(row['outcome'] == outcome for row in rows)
              for outcome in ('improved', 'worsened', 'unchanged')}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'experiment': 'Dense + BM25 positive-IDF, equal-weight reciprocal rank fusion',
              'evaluation_role': 'development; previously inspected datasets',
              'settings': {'candidate_k_per_method': CANDIDATE_K, 'rrf_constant': RRF_CONSTANT,
                           'bm25_k1': BM25_K1, 'bm25_b': BM25_B,
                           'tokenizer': 'lowercase ASCII alphanumeric; unique query terms',
                           'stopwords': False, 'stemming': False,
                           'zero_keyword_scores': 'excluded', 'tie_break': 'chunk ID ascending'},
              'dataset_status': goldens['status'], 'goldens_sha256': golden_hash,
              'corpus_text_sha256': goldens['corpus_text_sha256'],
              'baseline_report_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
              'retriever_source_sha256': retriever_hash,
              'experiment_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'package_versions': packages, 'original_rankings_reproduced': True,
              'top_k': TOP_K, 'question_count': len(rows),
              'changed_query_count': sum(row['query_changed'] for row in rows),
              'baseline_summary': baseline['summary'], 'summary': summary,
              'keyword_only_summary': keyword_summary,
              'rank_changes': counts, 'cases': rows}
    with OUTPUT_PATH.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print('\nBASELINE -> HYBRID SEARCH EXPERIMENT')
    for metric, value in summary.items():
        print(f"{metric}: {baseline['summary'][metric]:.4f} -> {value:.4f}")
    print(f'Rank changes: {counts}')
    print('Keyword-only diagnostic:', {k: round(v, 4) for k, v in keyword_summary.items()})
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', choices=['pilot', 'validation'])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.dataset is None:
        parser.error('dataset required unless --self-test is used')
    else:
        configure(args.dataset)
        main()
