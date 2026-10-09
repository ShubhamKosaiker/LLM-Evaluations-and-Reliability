"""Collect baseline RAG outputs for later answer evaluation; no quality scores.

Run from the repository root. Uses the same functions as rag_pipeline.py,
but calls them separately to preserve retrieved evidence if generation fails.
Reports never overwrite existing files. Requires the existing indexed corpus.
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import eval_retriever as evaluator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator_settings(path):
    """Read literal settings without constructing an API client."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'create'
             and any(kw.arg == 'model' for kw in node.keywords)]
    if len(calls) != 1:
        raise ValueError('Cannot identify one generator model call')
    settings = {}
    for kw in calls[0].keywords:
        if kw.arg in ('model', 'temperature'):
            settings[kw.arg] = ast.literal_eval(kw.value)
    if 'model' not in settings or 'temperature' not in settings:
        raise ValueError('Generator model and temperature must be explicit')
    return settings


def valid_retrieval(result, chunks, top_k):
    ids = result['ids'][0]
    documents = result['documents'][0]
    sources = result['metadatas'][0]
    if (len(ids) != top_k or len(set(ids)) != top_k or
            len(documents) != top_k or len(sources) != top_k):
        raise ValueError('Unexpected retrieval result size or duplicate IDs')
    for cid, text in zip(ids, documents):
        if cid not in chunks or text != chunks[cid]:
            raise ValueError('Retrieved passage differs from reviewed corpus')
    return ids, documents, sources


def collect_case(case, chunks, top_k, retrieve, generate):
    row = {
        'id': case['id'], 'topic': case['topic'], 'query': case['query'],
        'reference_answer': case['reference_answer'],
        'relevant_chunk_ids': case['relevant_chunk_ids'],
        'reference_evidence': case['evidence'],
        'status': 'pending', 'answer': None, 'retrieved_ids': [],
        'contexts': [], 'sources': [],
        'timing_seconds': {}, 'error': None,
    }
    started = time.perf_counter()
    try:
        result = retrieve(case['query'], top_k=top_k)
        ids, documents, sources = valid_retrieval(result, chunks, top_k)
        row.update(retrieved_ids=ids, contexts=documents, sources=sources)
    except Exception as error:
        row['status'] = 'error'
        # Do not persist API responses, credentials or raw exception messages.
        row['error'] = {'stage': 'retrieval', 'type': type(error).__name__}
        row['timing_seconds']['retrieval'] = time.perf_counter() - started
        return row
    row['timing_seconds']['retrieval'] = time.perf_counter() - started
    started = time.perf_counter()
    try:
        # The model receives only the question and retrieved text, never gold labels.
        answer = generate(case['query'], row['contexts'])
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Empty generator output')
        row['answer'] = answer
        row['status'] = 'success'
    except Exception as error:
        row['status'] = 'error'
        row['error'] = {'stage': 'generation', 'type': type(error).__name__}
    row['timing_seconds']['generation'] = time.perf_counter() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['pilot', 'validation'])
    parser.add_argument('--limit', type=int, help='First N cases, in dataset order')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Passages sent to generator; default matches run_rag')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    # Reserve output before making chargeable model requests.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'Report already exists: {args.output}')
    stem = 'retrieval' if args.dataset == 'pilot' else 'retrieval_validation'
    evaluator.GOLDENS_PATH = Path(f'data/eval/{stem}_goldens.json')
    goldens, chunks, golden_hash = evaluator.load_and_validate()
    cases = goldens['cases']
    if args.limit is not None:
        if args.limit > len(cases):
            parser.error('--limit exceeds dataset size')
        cases = cases[:args.limit]
    if args.top_k > len(chunks):
        parser.error('--top-k exceeds corpus size')
    if not Path('data/chroma_db/chroma.sqlite3').is_file():
        raise FileNotFoundError('Existing Chroma database not found')
    settings = generator_settings(Path('src/generator.py'))
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv('GROQ_API_KEY'):
        raise ValueError('GROQ_API_KEY is not configured in the environment or .env')
    from retriever import collection, retrieve
    stored = collection.get(include=['documents'])
    if dict(zip(stored['ids'], stored['documents'])) != chunks:
        raise ValueError('Index differs from the reviewed corpus; run check_index.py')
    from generator import generate_answer
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=False)
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                           text=True, check=False)
    report = {
        'schema_version': 1, 'run_type': 'baseline_rag_answer_collection',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_name': goldens['dataset_name'], 'dataset_status': goldens['status'],
        'evaluation_role': 'development; previously inspected draft questions',
        'goldens_sha256': golden_hash, 'corpus_text_sha256': goldens['corpus_text_sha256'],
        'git_head': revision.stdout.strip() if revision.returncode == 0 else None,
        'working_tree_dirty': bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        'source_sha256': {name: digest(Path('src', name)) for name in
                          ['retriever.py', 'generator.py', 'eval_retriever.py']},
        'collector_source_sha256': digest(Path(__file__)),
        'package_versions': {name: version(name) for name in
                             ['chromadb', 'sentence-transformers', 'groq', 'python-dotenv']},
        'configuration': {'retrieval_method': 'original_dense', 'top_k': args.top_k,
                          'generator': settings},
        'dataset_question_count': len(goldens['cases']),
        'selected_case_ids': [case['id'] for case in cases],
        'run_status': 'in_progress', 'quality_evaluation_status': 'not_scored',
        'limitations': ['Draft references require human review.',
                        'Successful generation does not establish answer correctness.',
                        'Timings are diagnostic, not a latency benchmark.',
                        'Only completed cases are checkpointed; report is not resumable.'],
        'cases': [],
    }
    # Exclusive create prevents accidental replacement even by concurrent runs.
    with args.output.open('x', encoding='utf-8') as handle:
        def checkpoint():
            handle.seek(0)
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
            handle.truncate()
            handle.flush()
        checkpoint()
        for case in cases:
            row = collect_case(case, chunks, args.top_k, retrieve, generate_answer)
            report['cases'].append(row)
            checkpoint()
            detail = row['error']['stage'] + ':' + row['error']['type'] if row['error'] else 'answer collected'
            print(f"{row['id']}: {row['status']} ({detail})")
        successes = sum(row['status'] == 'success' for row in report['cases'])
        report['summary'] = {'attempted': len(report['cases']), 'successful': successes,
                             'errors': len(report['cases']) - successes}
        report['run_status'] = 'completed'
        report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint()
    print('Collection summary:', report['summary'])
    print('Quality evaluation: not scored yet')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
