"""Print a table of every eval run in results/."""
import glob
import json
import os

rows = []
for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "*.json"))):
    with open(path) as f:
        r = json.load(f)
    rows.append(r)

rows.sort(key=lambda r: -r["success_rate"])

hdr = f"{'run':<34} {'success':>10} {'rate':>7} {'mean steps':>11} {'n_act':>6} {'temp_ens':>9}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    ms = r.get("mean_steps_to_success")
    print(f"{r['label']:<34} "
          f"{r['successes']:>4}/{r['episodes']:<5} "
          f"{r['success_rate']:>6.1%} "
          f"{(f'{ms:.1f}' if ms else '-'):>11} "
          f"{str(r.get('n_action_steps')):>6} "
          f"{str(r.get('temporal_ensemble_coeff')):>9}")
