"""CPU-only readout for PCQR-TTA v2, no hidden aggregation assumptions."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    root = args.run
    if (root / 'FAILED.json').exists():
        doc = json.loads((root / 'FAILED.json').read_text())
        raise SystemExit('FAILED: ' + str(doc.get('traceback', doc.get('error'))))
    if not (root / 'COMPLETE.json').exists():
        raise SystemExit('No COMPLETE.json (job may not have completed)')
    complete = json.loads((root / 'COMPLETE.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted(root.glob('*_case*_result.json'))]
    if len(rows) != complete['case_results']:
        raise SystemExit('Mismatch: result count vs COMPLETE.json')
    print('run:', root)
    print('complete:', json.dumps(complete, ensure_ascii=False))
    print('MODE       CASES  AVG_SR_RAW  REPLANS  UPDATES  POLICY_S  MATCHED_PREUPDATE')
    for mode in complete['modes']:
        items = [r for r in rows if r['mode'] == mode]
        if not items:
            continue
        rates = [float(r['metrics']['success_rate']) for r in items]
        print(f'{mode:<10} {len(items):>5} {mean(rates):>11.2f} '
              f'{sum(r["replans"] for r in items):>8} '
              f'{sum(r["applied_optimizer_steps"] for r in items):>8} '
              f'{sum(r["policy_wall_ms"] for r in items)/1000:>9.2f} '
              f'{sum(not r.get("preupdate_mismatch") for r in items)}/{len(items)}')
        status = Counter()
        for p in sorted(root.glob(f'{mode}_case*.jsonl')):
            for line in p.read_text().splitlines():
                if line.strip():
                    event = json.loads(line)
                    if event.get('type') == 'adapt':
                        status[event.get('status', 'unknown')] += 1
        print(' '*12 + f'Adapt events: {dict(status)}')
    print('NOTE: CEM budget is per replan; changing receding horizon changes total CEM work.')
    print('NOTE: this pilot tests adaptation but includes no dynamics perturbation.')


if __name__ == '__main__':
    main()
