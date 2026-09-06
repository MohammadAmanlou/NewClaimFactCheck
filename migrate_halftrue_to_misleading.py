#!/usr/bin/env python3
"""One-off migration: universal labeling in results/ — map every Half-True to Misleading.

Scope (read-only for source code; only files under results/ are touched):
  1. predictions JSON + CSV : record fields `label` / `prediction`,
     "Half-True" (any case/spacing variant) -> "Misleading".
  2. leaf metrics (*_metrics.json with a sibling *predictions.json) : recomputed
     from the remapped predictions with the exact formulas of src/evaluation.py.
     Canonical set = original per-label keys (Half-True folded to Misleading)
     UNION the gold labels present, in universal order
     [Supported, Refuted, Misleading, Not Enough Information].
  3. aggregates (summary.json, posthoc_balanced_summary.json, combined_summary.json):
     rebuilt from the remapped predictions; non-metric fields (sampling_info,
     timestamps, prompt_method, ...) preserved. `canonical_labels` / sampling
     `labels` / `per_label_info` keys folded the same way.
  4. Run logs (*.txt) are historical records and are intentionally left untouched.

Only files actually containing Half-True (or a Half-True canonical list) are
rewritten; everything else is left byte-identical.

Usage:
    python migrate_halftrue_to_misleading.py           # apply migration
    python migrate_halftrue_to_misleading.py --dry-run # preview only
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

HALF_TRUE_VARIANTS = {"half-true", "half true", "halftrue", "half_true"}
UNIVERSAL_ORDER = ["Supported", "Refuted", "Misleading", "Not Enough Information"]
NEI_LABEL = "Not Enough Information"


def fold(value):
    """Half-True (any variant) -> Misleading; everything else unchanged."""
    if isinstance(value, str) and value.strip().lower() in HALF_TRUE_VARIANTS:
        return "Misleading"
    return value


def is_half_true(value):
    return isinstance(value, str) and value.strip().lower() in HALF_TRUE_VARIANTS


def fold_list(items):
    """Fold a label list, preserving order and dropping duplicates."""
    out = []
    for item in items or []:
        m = fold(item)
        if m not in out:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Metric formulas — exact mirror of src/evaluation.py (so numbers stay
# comparable with pipeline output).
# ---------------------------------------------------------------------------

def _valid(results):
    return [r for r in results
            if r.get("prediction") and r.get("label")]


def calculate_metrics(results, canonical_labels):
    valid = _valid(results)
    if not valid:
        return {"error": "No valid results"}
    label_counts = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0
    for r in valid:
        true, pred = r["label"], r["prediction"]
        label_counts[true] += 1
        confusion[true][pred] += 1
        if true == pred:
            correct += 1
    total = len(valid)
    per_label = {}
    for label in canonical_labels:
        if label_counts[label] == 0:
            continue
        tp = confusion[label][label]
        fn = label_counts[label] - tp
        fp = sum(confusion[o][label] for o in canonical_labels if o != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0
        per_label[label] = {"precision": round(p, 4), "recall": round(rec, 4),
                            "f1": round(f1, 4), "support": label_counts[label]}
    def avg(k):
        return round(sum(m[k] for m in per_label.values()) / len(per_label), 4) if per_label else 0.0
    pairs = [(r["label"], r["prediction"]) for r in valid]
    n_nei = sum(1 for _, pr in pairs if pr == NEI_LABEL)
    non_nei = sum(1 for t, _ in pairs if t != NEI_LABEL)
    false_nei = sum(1 for t, pr in pairs if pr == NEI_LABEL and t != NEI_LABEL)
    return {"accuracy": round(correct / total, 4) if total else 0.0,
            "macro_precision": avg("precision"), "macro_recall": avg("recall"),
            "macro_f1": avg("f1"), "total_samples": total,
            "correct_predictions": correct, "per_label_metrics": per_label,
            "nei_prediction_rate": round(n_nei / total, 4) if total else 0.0,
            "false_nei_rate": round(false_nei / non_nei, 4) if non_nei else 0.0}


def statistical_tests(seen_results, unseen_results):
    """Mirror of evaluation._run_statistical_tests; placeholder on failure."""
    try:
        from scipy import stats
        seen = [1 if (r.get("prediction") and r.get("label")
                      and r["prediction"] == r["label"]) else 0
                for r in seen_results if r.get("prediction") and r.get("label")]
        unseen = [1 if (r.get("prediction") and r.get("label")
                        and r["prediction"] == r["label"]) else 0
                  for r in unseen_results if r.get("prediction") and r.get("label")]
        if len(seen) <= 1 or len(unseen) <= 1:
            raise ValueError("Need more than one sample per group.")
        t_stat, p_value = stats.ttest_ind(seen, unseen)
        cont = [[sum(seen), len(seen) - sum(seen)],
                [sum(unseen), len(unseen) - sum(unseen)]]
        chi2, chi_p, dof, _ = stats.chi2_contingency(cont)
        return {"t_test": {"t_statistic": round(float(t_stat), 4),
                           "p_value": round(float(p_value), 4),
                           "significant_at_0.05": bool(p_value < 0.05),
                           "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference"},
                "chi_square_test": {"chi2_statistic": round(float(chi2), 4),
                                    "p_value": round(float(chi_p), 4),
                                    "degrees_of_freedom": int(dof),
                                    "significant_at_0.05": bool(chi_p < 0.05),
                                    "interpretation": "Significant difference" if chi_p < 0.05 else "No significant difference"}}
    except Exception as e:
        return f"Insufficient data for statistical testing ({e})"


def canonical_for(old_keys, gold_labels):
    """Original per-label keys folded, UNION gold labels present, universal order."""
    want = {fold(k) for k in (old_keys or [])} | set(gold_labels or [])
    want.discard("Half-True")
    return [l for l in UNIVERSAL_ORDER if l in want]


def fold_sampling_info(info):
    """Fold Half-True in sampling metadata (labels list + per_label_info keys)."""
    if not isinstance(info, dict):
        return info, False
    changed = False
    if isinstance(info.get("labels"), list):
        new_labels = fold_list(info["labels"])
        if new_labels != info["labels"]:
            info["labels"] = new_labels
            changed = True
    pli = info.get("per_label_info")
    if isinstance(pli, dict) and any(is_half_true(k) for k in pli):
        merged = {}
        for k, v in pli.items():
            nk = fold(k)
            if nk in merged and isinstance(v, dict) and isinstance(merged[nk], dict):
                for f in ("original_seen", "original_unseen", "sampled_seen",
                          "sampled_unseen"):
                    if isinstance(v.get(f), int) and isinstance(merged[nk].get(f), int):
                        merged[nk][f] += v[f]
            else:
                merged[nk] = v
        info["per_label_info"] = merged
        changed = True
    return info, changed


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats = {"pred_files": 0, "pred_files_changed": 0, "gold_folded": 0,
             "pred_folded": 0, "csv_changed": 0, "metrics_recomputed": 0,
             "summaries_rebuilt": 0, "combined_rebuilt": 0, "skipped": 0}
    affected_preds = set()  # prediction JSON paths whose records changed

    # -- Phase 1: remap predictions JSON + CSV -------------------------------
    pred_files = sorted(p for p in RESULTS.rglob("*.json")
                        if "prediction" in p.name.lower())
    stats["pred_files"] = len(pred_files)
    for pf in pred_files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"SKIP (unreadable): {pf.relative_to(ROOT)}: {e}")
            stats["skipped"] += 1
            continue
        if not isinstance(data, list):
            continue
        changed = False
        for r in data:
            if not isinstance(r, dict):
                continue
            if "label" in r and is_half_true(r["label"]):
                r["label"] = "Misleading"
                stats["gold_folded"] += 1
                changed = True
            if "prediction" in r and is_half_true(r["prediction"]):
                r["prediction"] = "Misleading"
                stats["pred_folded"] += 1
                changed = True
        if changed:
            stats["pred_files_changed"] += 1
            affected_preds.add(pf)
            if not args.dry_run:
                dump(pf, data)
            # sibling CSV
            cf = pf.with_suffix(".csv")
            if cf.exists():
                try:
                    with open(cf, encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    fieldnames = None
                    cchanged = False
                    with open(cf, encoding="utf-8") as f:
                        fieldnames = csv.DictReader(f).fieldnames
                    for r in rows:
                        for k in ("label", "prediction"):
                            if k in r and r[k] is not None and is_half_true(r[k]):
                                r[k] = "Misleading"
                                cchanged = True
                    if cchanged:
                        stats["csv_changed"] += 1
                        if not args.dry_run:
                            with open(cf, "w", encoding="utf-8", newline="") as f:
                                w = csv.DictWriter(f, fieldnames=fieldnames)
                                w.writeheader()
                                w.writerows(rows)
                except Exception as e:
                    print(f"SKIP CSV: {cf.relative_to(ROOT)}: {e}")

    # -- Phase 2: recompute leaf metrics (only for affected predictions) -----
    for pf in sorted(affected_preds):
        mf = pf.parent / pf.name.replace("predictions", "metrics")
        if not mf.exists():
            continue
        try:
            old_metrics = json.loads(mf.read_text(encoding="utf-8"))
            data = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"SKIP metrics: {mf.relative_to(ROOT)}: {e}")
            continue
        gold = sorted({r["label"] for r in data
                       if isinstance(r, dict) and r.get("label")})
        canon = canonical_for(list(old_metrics.get("per_label_metrics", {}).keys()), gold)
        new_metrics = calculate_metrics(data, canon)
        stats["metrics_recomputed"] += 1
        if not args.dry_run:
            dump(mf, new_metrics)

    # -- Phase 3: rebuild aggregates -----------------------------------------
    def load_preds(d):
        files = sorted((d).glob("*predictions.json")) if d.is_dir() else []
        if not files:
            return None
        try:
            return json.loads(files[0].read_text(encoding="utf-8"))
        except Exception:
            return None

    def rebuild_pair(seen, unseen, old_comparison):
        s_gold = sorted({r["label"] for r in seen
                         if isinstance(r, dict) and r.get("label")}) if seen else []
        u_gold = sorted({r["label"] for r in unseen
                         if isinstance(r, dict) and r.get("label")}) if unseen else []
        oldm = (old_comparison or {}).get("seen_metrics", {})
        canon = canonical_for(list(oldm.get("per_label_metrics", {}).keys()),
                              sorted(set(s_gold) | set(u_gold)))
        sm = calculate_metrics(seen, canon) if seen is not None else {"error": "No valid results"}
        um = calculate_metrics(unseen, canon) if unseen is not None else {"error": "No valid results"}
        comp = {"seen_metrics": sm, "unseen_metrics": um,
                "statistical_tests": statistical_tests(seen or [], unseen or []),
                "accuracy_difference": round(um.get("accuracy", 0.0) - sm.get("accuracy", 0.0), 4)}
        if isinstance(old_comparison, dict):  # preserve sampling/meta fields
            for k, v in old_comparison.items():
                if k not in comp:
                    comp[k] = v
            if isinstance(comp.get("sampling_info"), dict):
                comp["sampling_info"], _ = fold_sampling_info(comp["sampling_info"])
        return sm, um, comp

    # 3a. summary.json (per method: seen/ + unseen/ siblings)
    for sf in sorted(RESULTS.rglob("summary.json")):
        if "posthoc" in sf.parts or sf.parent.name == "results":
            continue
        method_dir = sf.parent
        seen, unseen = load_preds(method_dir / "seen"), load_preds(method_dir / "unseen")
        if seen is None and unseen is None:
            continue
        try:
            old = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        # only rebuild if sources were affected or canonical mentions Half-True
        src_affected = any((method_dir / s / f"{s}_predictions.json") in affected_preds
                           for s in ("seen", "unseen"))
        canon_has_ht = any(is_half_true(x) for x in
                           old.get("experiment_info", {}).get("canonical_labels", []))
        if not (src_affected or canon_has_ht):
            continue
        sm, um, comp = rebuild_pair(seen or [], unseen or [], old.get("comparison"))
        new = {"experiment_info": dict(old.get("experiment_info", {})),
               "seen_data": sm, "unseen_data": um, "comparison": comp}
        new["experiment_info"]["canonical_labels"] = fold_list(
            new["experiment_info"].get("canonical_labels", []))
        for k in old:  # keep any extra top-level keys in original order
            if k not in new:
                new[k] = old[k]
        stats["summaries_rebuilt"] += 1
        if not args.dry_run:
            dump(sf, new)

    # 3b. posthoc_balanced_summary.json
    for sf in sorted(RESULTS.rglob("posthoc_balanced_summary.json")):
        posthoc_dir = sf.parent
        seen, unseen = load_preds(posthoc_dir / "seen"), load_preds(posthoc_dir / "unseen")
        if seen is None and unseen is None:
            continue
        try:
            old = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        src_affected = any((posthoc_dir / s / f"{s}_posthoc_balanced_predictions.json") in affected_preds
                           for s in ("seen", "unseen"))
        if not src_affected:
            continue
        sm, um, comp = rebuild_pair(seen or [], unseen or [], old.get("comparison"))
        new = dict(old)
        new["seen_metrics"], new["unseen_metrics"], new["comparison"] = sm, um, comp
        if isinstance(new.get("sampling_info"), dict):
            new["sampling_info"], _ = fold_sampling_info(new["sampling_info"])
        stats["summaries_rebuilt"] += 1
        if not args.dry_run:
            dump(sf, new)

    # 3c. sampling info sidecars (label-name renames only)
    for name in ("posthoc_balanced_sampling_info.json", "sampling_info.json"):
        for fp in RESULTS.rglob(name):
            try:
                info = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            _, changed = fold_sampling_info(info)
            if changed and not args.dry_run:
                dump(fp, info)

    # 3d. combined_summary.json — refresh affected methods from rebuilt summaries
    # (method dirs are usually subdirs of the combined file's dir, but some
    # runs wrote combined_summary.json inside the method dir itself)
    def entry_has_half_true(entry):
        found = []

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if is_half_true(k) or (isinstance(v, str) and is_half_true(v)):
                        found.append(True)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    if isinstance(v, str) and is_half_true(v):
                        found.append(True)
                    walk(v)
        walk(entry)
        return bool(found)

    for cf in sorted(RESULTS.rglob("combined_summary.json")):
        try:
            old = json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            continue
        changed = False
        for method in list(old.get("methods", {}).keys()):
            candidates = [cf.parent / method]
            if cf.parent.name == method:
                candidates.append(cf.parent)
            mdir = next((d for d in candidates if d.is_dir()), None)
            if mdir is None:
                continue
            sfile = mdir / "summary.json"
            if not sfile.exists():
                continue
            affected = any(str(p.relative_to(ROOT)).startswith(
                str(mdir.relative_to(ROOT))) for p in affected_preds)
            if not affected and not entry_has_half_true(old["methods"][method]):
                continue
            try:
                s = json.loads(sfile.read_text(encoding="utf-8"))
            except Exception:
                continue
            entry = {"seen_metrics": s.get("seen_data"),
                     "unseen_metrics": s.get("unseen_data"),
                     "comparison": s.get("comparison"),
                     "full_data": {"seen_metrics": s.get("seen_data"),
                                   "unseen_metrics": s.get("unseen_data"),
                                   "comparison": s.get("comparison")}}
            pb = mdir / "posthoc_balanced" / "posthoc_balanced_summary.json"
            if pb.exists():
                try:
                    p = json.loads(pb.read_text(encoding="utf-8"))
                    entry["posthoc_balanced"] = {
                        "seen_metrics": p.get("seen_metrics"),
                        "unseen_metrics": p.get("unseen_metrics"),
                        "comparison": p.get("comparison")}
                except Exception:
                    pass
            else:
                if "posthoc_balanced" in old["methods"][method]:
                    entry["posthoc_balanced"] = old["methods"][method]["posthoc_balanced"]
            old["methods"][method] = entry
            changed = True
        if changed:
            stats["combined_rebuilt"] += 1
            if not args.dry_run:
                dump(cf, old)

    print("\n==== Half-True -> Misleading migration "
          + ("(DRY RUN)" if args.dry_run else "(APPLIED)") + " ====")
    print(f"prediction files scanned : {stats['pred_files']}")
    print(f"prediction files changed : {stats['pred_files_changed']}")
    print(f"gold labels folded       : {stats['gold_folded']}")
    print(f"predictions folded       : {stats['pred_folded']}")
    print(f"CSV files changed        : {stats['csv_changed']}")
    print(f"leaf metrics recomputed  : {stats['metrics_recomputed']}")
    print(f"summaries rebuilt        : {stats['summaries_rebuilt']}")
    print(f"combined summaries rebuilt: {stats['combined_rebuilt']}")
    print("Run logs (*.txt) left untouched by design.")

    # -- verification rescan (label-bearing fields only, not claim text) ----
    remaining_files = set()

    def check_value(v):
        return is_half_true(v)

    def walk_labels(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if is_half_true(k):  # e.g. per_label_metrics / per_label_info keys
                    remaining_files.add(str(path.relative_to(ROOT)))
                if k in ("label", "prediction", "canonical_labels", "labels") and (
                        check_value(v) or (isinstance(v, list) and any(check_value(x) for x in v))):
                    remaining_files.add(str(path.relative_to(ROOT)))
                elif k == "per_label_info" and isinstance(v, dict):
                    if any(is_half_true(x) for x in v):
                        remaining_files.add(str(path.relative_to(ROOT)))
                else:
                    walk_labels(v, path)
        elif isinstance(obj, list):
            for item in obj:
                walk_labels(item, path)

    for p in list(RESULTS.rglob("*.json")) + list(RESULTS.rglob("*.csv")):
        if p.suffix == ".json":
            try:
                walk_labels(json.loads(p.read_text(encoding="utf-8")), p)
            except Exception:
                pass
        else:
            try:
                with open(p, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames and any(
                            is_half_true(r.get(k, "")) for r in reader
                            for k in ("label", "prediction") if k in (reader.fieldnames or [])):
                        remaining_files.add(str(p.relative_to(ROOT)))
            except Exception:
                pass
    print(f"files with Half-True left in label fields: {len(remaining_files)}")
    for f in sorted(remaining_files)[:20]:
        print(f"  REMAINING: {f}")


if __name__ == "__main__":
    sys.exit(main())
