import json
from pipeline import RAGPipeline


def load_gold(path: str = "gold.json"):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def hit_rate(retrieved_chunk_ids: list, gold_sources: list) -> float:
    retrieved_sources = {cid.split("__")[0] for cid in retrieved_chunk_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def evaluate(pipeline: RAGPipeline, gold_questions: list, k: int = 5, verbose: bool = False) -> dict:
    total = 0.0
    results_by_type = {}
    results = []

    for q in gold_questions:
        q_id = q['id']
        q_type = q['type']
        q_text = q['question']
        gold_sources = q['gold_sources']

        retrieved = pipeline.retrieve(q_text, k=k)
        retrieved_chunk_ids = [chunk_id for _, _, chunk_id in retrieved]

        score = hit_rate(retrieved_chunk_ids, gold_sources)
        total += score

        results.append({
            'id': q_id,
            'type': q_type,
            'score': score,
            'gold': gold_sources,
            'retrieved_sources': [cid.split("__")[0] for cid in retrieved_chunk_ids]
        })

        if q_type not in results_by_type:
            results_by_type[q_type] = {'total_score': 0.0, 'count': 0}
        results_by_type[q_type]['total_score'] += score
        results_by_type[q_type]['count'] += 1

        if verbose:
            mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
            print(f"  [{q_id:2d}] {q_type:25s} hit@{k} = {score:.2f}  {mark}  {q_text[:60]}...")

    mean = total / len(gold_questions)
    if verbose:
        print(f"\n  TOTAL: hit-rate@{k} = {mean:.2f} ({total:.1f} / {len(gold_questions)})")

    return {'mean': mean, 'by_type': results_by_type, 'results': results}


def main():
    print("=" * 60)
    print("COMPARING CHUNKING STRATEGIES")
    print("=" * 60)

    gold = load_gold()
    print(f"\nLoaded {len(gold)} questions")

    print("\n[1] Strategy A: Fixed-size (2000 chars, no overlap)")
    rag_fixed = RAGPipeline('data', strategy='fixed')
    rag_fixed.build_index()
    result_fixed = evaluate(rag_fixed, gold, k=5, verbose=True)

    print("\n[2] Strategy B: Recursive (chunk_size=400, overlap=80)")
    rag_recursive = RAGPipeline('data', strategy='recursive')
    rag_recursive.build_index()
    result_recursive = evaluate(rag_recursive, gold, k=5, verbose=True)

    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    for name, rag in [('Fixed', rag_fixed), ('Recursive', rag_recursive)]:
        stats = rag.corpus_stats()
        print(f"\n{name}:")
        print(f"  Documents: {stats['num_documents']}")
        print(f"  Total chars: {stats['total_chars']:,}")
        print(f"  Chunks: {stats['num_chunks']}")
        print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} chars")

    print("\n" + "=" * 60)
    print("RESULTS BY QUESTION TYPE (hit-rate@5)")
    print("=" * 60)
    print(f"{'Type':<15} {'Fixed':<12} {'Recursive':<12} {'Delta':<8}")
    print("-" * 50)

    all_types = set(result_fixed['by_type'].keys()) | set(result_recursive['by_type'].keys())
    for t in sorted(all_types):
        fixed_score = result_fixed['by_type'].get(t, {}).get('total_score', 0) / max(
            result_fixed['by_type'].get(t, {}).get('count', 1), 1)
        rec_score = result_recursive['by_type'].get(t, {}).get('total_score', 0) / max(
            result_recursive['by_type'].get(t, {}).get('count', 1), 1)
        print(f"{t:<15} {fixed_score:<12.2%} {rec_score:<12.2%} {rec_score - fixed_score:+.2%}")

    print(f"\n{'TOTAL':<15} {result_fixed['mean']:<12.2%} {result_recursive['mean']:<12.2%} "
          f"{result_recursive['mean'] - result_fixed['mean']:+.2%}")

    print("\n" + "=" * 60)
    print("RESULTS SAVED TO eval_results.json")
    print("=" * 60)

    with open('eval_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'fixed': {
                'hit_rate': result_fixed['mean'],
                'by_type': {t: {'hit_rate': d['total_score'] / d['count']} for t, d in result_fixed['by_type'].items()},
                'results': result_fixed['results']
            },
            'recursive': {
                'hit_rate': result_recursive['mean'],
                'by_type': {t: {'hit_rate': d['total_score'] / d['count']} for t, d in
                            result_recursive['by_type'].items()},
                'results': result_recursive['results']
            }
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()