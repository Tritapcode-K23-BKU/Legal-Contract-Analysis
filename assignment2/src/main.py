"""
Assignment 2 - Main Pipeline
Chạy toàn bộ Assignment 2: NER + SRL + Intent Classification
Usage: python main.py [--task {ner,srl,intent,all}]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Thêm src/ vào sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from task21_ner    import NERPipeline
from task22_srl    import SRLPipeline
from task23_intent import IntentPipeline, SAMPLE_TRAINING_DATA


BASE   = Path(__file__).parent
INPUT  = BASE / "input"  / "clauses.txt"
OUTPUT = BASE / "output"


def run_ner(verbose: bool = True):
    print("\n" + "="*60)
    print("  TASK 2.1 — Named Entity Recognition (NER)")
    print("="*60)
    t0 = time.time()
    pipeline = NERPipeline()
    results = pipeline.process_file(
        str(INPUT),
        str(OUTPUT / "ner_results.json"),
    )
    elapsed = time.time() - t0
    total_entities = sum(len(r["entities"]) for r in results)
    print(f"\n  Completed in {elapsed:.2f}s | {len(results)} clauses | {total_entities} entities")
    if verbose:
        _print_ner_summary(results)
    return results


def run_srl(verbose: bool = True):
    print("\n" + "="*60)
    print("  TASK 2.2 — Semantic Role Labeling (SRL)")
    print("="*60)
    t0 = time.time()
    pipeline = SRLPipeline()
    results = pipeline.process_file(
        str(INPUT),
        str(OUTPUT / "srl_results.json"),
    )
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.2f}s | {len(results)} clauses")
    if verbose:
        _print_srl_summary(results)
    return results


def run_intent(verbose: bool = True):
    print("\n" + "="*60)
    print("  TASK 2.3 — Intent Classification")
    print("="*60)
    t0 = time.time()
    pipeline = IntentPipeline()
    # Train TF-IDF trên sample data
    texts, labels = zip(*SAMPLE_TRAINING_DATA)
    pipeline.tfidf.train(list(texts), list(labels))
    results = pipeline.process_file(
        str(INPUT),
        str(OUTPUT / "intent_classification.txt"),
        json_path=str(OUTPUT / "intent_classification.json"),
    )
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.2f}s | {len(results)} clauses")
    return results


def run_all():
    print("\n" + "╔" + "═"*58 + "╗")
    print("  ║  Assignment 2 — Legal NLP Full Pipeline             ║")
    print("╚" + "═"*58 + "╝")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    ner_results    = run_ner(verbose=False)
    srl_results    = run_srl(verbose=False)
    intent_results = run_intent(verbose=False)
    total_time = time.time() - t_start

    # Tạo combined output
    combined = []
    clauses_text = [r["clause"] for r in ner_results]
    for i, clause in enumerate(clauses_text):
        ner_r    = ner_results[i] if i < len(ner_results) else {}
        srl_r    = srl_results[i] if i < len(srl_results) else {}
        intent_r = intent_results[i] if i < len(intent_results) else None

        combined.append({
            "clause_id":   i + 1,
            "clause":      clause,
            "ner":         ner_r.get("entities", []),
            "srl":         srl_r.get("frames", []),
            "intent":      intent_r.intent if intent_r else "Unknown",
            "confidence":  intent_r.confidence if intent_r else 0.0,
            "explanation": intent_r.explanation if intent_r else "",
        })

    combined_path = OUTPUT / "combined_results.json"
    combined_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Final report
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    total_entities = sum(len(r.get("ner", [])) for r in combined)
    intent_dist = {}
    for r in combined:
        k = r["intent"]
        intent_dist[k] = intent_dist.get(k, 0) + 1

    print(f"  Clauses processed : {len(combined)}")
    print(f"  Total entities    : {total_entities}")
    print(f"  Total time        : {total_time:.2f}s")
    print(f"\n  Intent distribution:")
    for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
        bar = "█" * count + "░" * (len(combined) - count)
        print(f"    {intent:25s} {bar} ({count})")

    print(f"\n  Output files:")
    for f in sorted(OUTPUT.glob("*.json")) + sorted(OUTPUT.glob("*.txt")):
        size = f.stat().st_size
        print(f"    {f.name:35s} {size:>6} bytes")

    print(f"\n  ✓ All outputs saved to: {OUTPUT}/")


def _print_ner_summary(results):
    print("\n  ── NER Preview ──")
    for r in results[:3]:
        print(f"\n  [{r['clause_id']}] {r['clause'][:55]}...")
        for e in r["entities"]:
            print(f"       [{e['label']:8s}] {e['text']}")


def _print_srl_summary(results):
    print("\n  ── SRL Preview ──")
    for r in results[:3]:
        print(f"\n  [{r['clause_id']}] {r['clause'][:55]}...")
        for frame in r["frames"]:
            print(f"       Predicate: {frame['predicate']}")
            for role, val in frame["roles"].items():
                print(f"       {role:10s}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2 — Legal NLP Pipeline")
    parser.add_argument(
        "--task",
        choices=["ner", "srl", "intent", "all"],
        default="all",
        help="Task to run (default: all)"
    )
    args = parser.parse_args()

    if args.task == "ner":
        run_ner()
    elif args.task == "srl":
        run_srl()
    elif args.task == "intent":
        run_intent()
    else:
        run_all()
