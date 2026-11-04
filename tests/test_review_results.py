"""Offline consistency checks for the portfolio evidence reader; never call the RAG pipeline."""
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('review_results', ROOT / 'tools' / 'review_results.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReviewResultsTests(unittest.TestCase):
    def test_committed_evidence_is_internally_consistent(self):
        review.check_evidence(review.Evidence(ROOT))

    def test_changed_retrieval_rank_is_rejected(self):
        evidence = review.Evidence(ROOT)
        report = evidence.read('retrieval_baseline.json')
        report['cases'][0]['first_relevant_rank_at_5'] = 99
        with self.assertRaises(ValueError):
            review.check_evidence(evidence)

    def test_changed_application_mean_is_rejected(self):
        evidence = review.Evidence(ROOT)
        report = evidence.read('deepeval_application_baseline_v1.json')
        report['summary']['by_dataset']['pilot']['completeness']['mean_score'] = .123
        with self.assertRaises(ValueError):
            review.check_evidence(evidence)

    def test_changed_geval_raw_score_is_rejected(self):
        evidence = review.Evidence(ROOT)
        report = evidence.read('deepeval_application_baseline_v1.json')
        report['results']['cough_01']['correctness']['judge_outputs'][0]['score'] = 2
        with self.assertRaises(ValueError):
            review.check_evidence(evidence)

    def test_changed_historical_latency_is_rejected(self):
        evidence = review.Evidence(ROOT)
        report = evidence.read('operational_diagnostics_v1.json')
        report['successful_case_timings']['retrieval_plus_generation']['p95_seconds'] = .01
        with self.assertRaises(ValueError):
            review.check_evidence(evidence)

    def test_partial_coverage_is_visible_and_not_zero_filled(self):
        evidence = review.Evidence(ROOT)
        review.check_evidence(evidence)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            review.show(evidence)
        self.assertIn('PARTIAL: contextual relevance:', out.getvalue())
        self.assertIn('validation: scored=8, errors=1, pending=1', out.getvalue())


if __name__ == '__main__':
    unittest.main()
