#!/usr/bin/env python3
"""One-off fix: DeepSeek runs used the wrong temporal split (2024-05-01).

Correct split is 2024-08-01. For every method under results/deepseek that has
seen/ + unseen/ predictions, this script pools both files and re-splits rows
by claim_date (claims dated before the split -> seen, rest -> unseen, empty /
unparseable dates -> seen, mirroring split_by_date in src/processing.py).

Then it consistently refreshes, for affected methods only:
  * seen/unseen predictions JSON + CSV (membership only; predictions untouched)
  * leaf metrics (recomputed with the EXACT formulas of src/evaluation.py and
    the ORIGINAL canonical label set stored in each metrics file)
  * posthoc balanced subsets (re-sampled with the same stratified algorithm
    and seed as src/pipeline.py, from the new pools) + their metrics
  * summary.json / posthoc_balanced_summary.json (metrics, statistical tests,
    sampling counts; timestamps and other meta preserved)
  * sampling sidecars (counts / claim-id lists)
  * combined_summary.json entries + experiment_info date (only when all
    methods under it share the new split)

Run logs (*.txt), source code and data/ are left untouched.
Files whose membership is already correct are left byte-identical.

Usage:
    python fix_deepseek_split.py [--provider results/deepseek] [--date 2024-08-01] [--dry-run]
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEI_LABEL = "Not Enough Information"
UNIVERSAL_ORDER = ["Supported", "Refuted", "Misleading", "Not Enough Information"]


def parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.fromisoformat(s.replace("T00:00:00", ""))
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Metric formulas — exact mirror of src/evaluation.py
# ---------------------------------------------------------------------------

def calculate_metrics(results, canonical_labels):
    valid = [r for r in results if r.get("prediction") and r.get("label")]
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
    try:
        from scipy import stats
        seen = [1 if r["prediction"] == r["label"] else 0
                for r in seen_results if r.get("prediction") and r.get("label")]
        unseen = [1 if r["prediction"] == r["label"] else 0
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


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def file_text(path):
    return path.read_text(encoding="utf-8")


def write_predictions(split_dir, split_name, records, dry_run, stats, exact=None):
    """Write seen/unseen predictions JSON+CSV. Returns True if changed."""
    jp = split_dir / exact if exact else split_dir / f"{split_name}_predictions.json"
    if not jp.exists():
        cands = sorted(split_dir.glob("*predictions.json"))
        if not cands:
            return False
        jp = cands[0]
    old_ids = None
    try:
        old_ids = [r.get("claim_id") for r in load_json(jp)
                   if isinstance(r, dict)]
    except Exception:
        pass
    new_ids = [r.get("claim_id") for r in records]
    if old_ids == new_ids:
        return False
    stats["pred_files_rewritten"] += 1
    if dry_run:
        return True
    dump(jp, records)
    cp = jp.with_suffix(".csv")
    if cp.exists():
        try:
            with open(cp, encoding="utf-8") as f:
                fieldnames = csv.DictReader(f).fieldnames
            with open(cp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in records:
                    w.writerow({k: ("" if v is None else v) for k, v in r.items()
                                if k in fieldnames})
            stats["csv_rewritten"] += 1
        except Exception as e:
            print(f"  SKIP CSV {cp.relative_to(ROOT)}: {e}")
    return True


def metrics_path_for(pred_json):
    return pred_json.parent / pred_json.name.replace("predictions", "metrics")


def recompute_metrics_file(pred_json, records, dry_run, stats):
    mp = metrics_path_for(pred_json)
    if not mp.exists():
        return
    try:
        old = load_json(mp)
    except Exception:
        return
    if isinstance(old, dict) and "error" in old and len(old) == 1:
        return  # 100%-None runs: nothing to compute
    canon = list(old.get("per_label_metrics", {}).keys())
    new = calculate_metrics(records, canon)
    if json.dumps(new, indent=4, ensure_ascii=False) == json.dumps(old, indent=4, ensure_ascii=False):
        return
    stats["metrics_recomputed"] += 1
    if not dry_run:
        dump(mp, new)


# ---------------------------------------------------------------------------
# Posthoc stratified re-sampling — mirror of pipeline._posthoc_stratified_balance_results
# ---------------------------------------------------------------------------

def result_label(r):
    for k in ("label", "gold_label", "true_label", "ground_truth", "actual_label"):
        if r.get(k) is not None:
            return r.get(k)
    return None


def resample_posthoc(seen, unseen, label_order, seed):
    rng = random.Random(seed)
    by_s, by_u = defaultdict(list), defaultdict(list)
    for r in seen:
        lab = result_label(r)
        if lab is not None:
            by_s[lab].append(r)
    for r in unseen:
        lab = result_label(r)
        if lab is not None:
            by_u[lab].append(r)
    common = [l for l in label_order if l in by_s and l in by_u]
    extra = sorted((set(by_s) & set(by_u)) - set(common), key=str)
    common.extend(extra)
    s_out, u_out, info = [], [], {}
    for lab in common:
        n = min(len(by_s[lab]), len(by_u[lab]))
        if n == 0:
            continue
        s_out.extend(rng.sample(by_s[lab], n))
        u_out.extend(rng.sample(by_u[lab], n))
        info[str(lab)] = {"original_seen": len(by_s[lab]), "original_unseen": len(by_u[lab]),
                          "sampled_seen": n, "sampled_unseen": n}
    rng.shuffle(s_out)
    rng.shuffle(u_out)
    return s_out, u_out, common, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="results/deepseek")
    ap.add_argument("--date", default="2024-08-01")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    provider = ROOT / args.provider
    split_dt = datetime.strptime(args.date, "%Y-%m-%d")
    stats = {"methods": 0, "methods_fixed": 0, "pred_files_rewritten": 0,
             "csv_rewritten": 0, "rows_moved": 0, "metrics_recomputed": 0,
             "summaries_rebuilt": 0, "combined_rebuilt": 0}

    rebuilt_summaries = set()

    # -- per method dir -------------------------------------------------------
    method_dirs = sorted(d for d in provider.rglob("*")
                         if d.is_dir() and (d / "seen").is_dir() and (d / "unseen").is_dir()
                         and "posthoc" not in d.parts)
    # skip dataset roots that merely contain method subdirs with their own splits
    for mdir in method_dirs:
        seen_f = mdir / "seen" / "seen_predictions.json"
        unseen_f = mdir / "unseen" / "unseen_predictions.json"
        if not seen_f.exists() or not unseen_f.exists():
            continue
        try:
            seen_old, unseen_old = load_json(seen_f), load_json(unseen_f)
        except Exception as e:
            print(f"SKIP {mdir.relative_to(ROOT)}: {e}")
            continue
        stats["methods"] += 1
        pool = [r for r in list(seen_old) + list(unseen_old) if isinstance(r, dict)]
        new_seen = [r for r in pool
                    if (lambda d: d is None or d < split_dt)(parse_date(r.get("claim_date", "")))]
        new_seen_ids = {r.get("claim_id") for r in new_seen}
        new_unseen = [r for r in pool if r.get("claim_id") not in new_seen_ids]

        # stored split label
        sfile = mdir / "summary.json"
        stored_split = None
        if sfile.exists():
            try:
                ei = load_json(sfile).get("experiment_info", {})
                stored_split = ei.get("temporal_split_date", ei.get("temporal_split"))
            except Exception:
                pass
        old_seen_ids = [r.get("claim_id") for r in seen_old]
        if ([r.get("claim_id") for r in new_seen] == old_seen_ids
                and stored_split == args.date):
            continue  # already correct (e.g. tracer cognitive_bias_aware)
        stats["methods_fixed"] += 1
        stats["rows_moved"] += len({r.get("claim_id") for r in new_seen} - set(old_seen_ids))
        print(f"FIX {mdir.relative_to(ROOT)}: seen {len(seen_old)}->{len(new_seen)}, "
              f"unseen {len(unseen_old)}->{len(new_unseen)} (split {stored_split}->{args.date})")

        if not args.dry_run:
            write_predictions(mdir / "seen", "seen", new_seen, False, stats)
            write_predictions(mdir / "unseen", "unseen", new_unseen, False, stats)
        else:
            write_predictions(mdir / "seen", "seen", new_seen, True, stats)
            write_predictions(mdir / "unseen", "unseen", new_unseen, True, stats)
        for split_name, recs in (("seen", new_seen), ("unseen", new_unseen)):
            recompute_metrics_file(mdir / split_name / f"{split_name}_predictions.json",
                                   recs, args.dry_run, stats)

        # -- summary.json -----------------------------------------------------
        if sfile.exists():
            try:
                old = load_json(sfile)
                old_comp = old.get("comparison", {})
                oldm = old_comp.get("seen_metrics", {})
                canon = list(oldm.get("per_label_metrics", {}).keys())
                sm, um = calculate_metrics(new_seen, canon), calculate_metrics(new_unseen, canon)
                comp = {"seen_metrics": sm, "unseen_metrics": um,
                        "statistical_tests": statistical_tests(new_seen, new_unseen),
                        "accuracy_difference": round(um.get("accuracy", 0.0) - sm.get("accuracy", 0.0), 4)}
                for k, v in old_comp.items():
                    if k not in comp:
                        comp[k] = v
                si = comp.get("sampling_info")
                if isinstance(si, dict):
                    si["original_seen_count"] = len(new_seen)
                    si["original_unseen_count"] = len(new_unseen)
                new = {"experiment_info": dict(old.get("experiment_info", {})),
                       "seen_data": sm, "unseen_data": um, "comparison": comp}
                new["experiment_info"]["temporal_split_date"] = args.date
                for k in old:
                    if k not in new:
                        new[k] = old[k]
                stats["summaries_rebuilt"] += 1
                rebuilt_summaries.add(sfile)
                if not args.dry_run:
                    dump(sfile, new)
            except Exception as e:
                print(f"  SKIP summary {sfile.relative_to(ROOT)}: {e}")

        # -- posthoc balanced: re-sample from the NEW pools -------------------
        pb = mdir / "posthoc_balanced"
        if pb.is_dir() and (pb / "seen").is_dir() and (pb / "unseen").is_dir():
            sidecar = mdir / "posthoc_balanced_sampling_info.json"
            seed, order = 42, list(UNIVERSAL_ORDER)
            if sidecar.exists():
                try:
                    sc = load_json(sidecar)
                    seed = sc.get("seed", 42)
                    if isinstance(sc.get("labels"), list) and sc["labels"]:
                        order = sc["labels"]
                except Exception:
                    pass
            b_seen, b_unseen, common, info = resample_posthoc(new_seen, new_unseen, order, seed)
            for split_name, recs in (("seen", b_seen), ("unseen", b_unseen)):
                exact = f"{split_name}_posthoc_balanced_predictions.json"
                write_predictions(pb / split_name, split_name, recs, args.dry_run, stats, exact=exact)
                recompute_metrics_file(pb / split_name / exact, recs, args.dry_run, stats)
            # posthoc summary
            pbs = pb / "posthoc_balanced_summary.json"
            if pbs.exists():
                try:
                    old = load_json(pbs)
                    old_comp = old.get("comparison", {})
                    oldm = old_comp.get("seen_metrics", old.get("seen_metrics", {}))
                    canon = list(oldm.get("per_label_metrics", {}).keys())
                    sm, um = calculate_metrics(b_seen, canon), calculate_metrics(b_unseen, canon)
                    comp = {"seen_metrics": sm, "unseen_metrics": um,
                            "statistical_tests": statistical_tests(b_seen, b_unseen),
                            "accuracy_difference": round(um.get("accuracy", 0.0) - sm.get("accuracy", 0.0), 4)}
                    for k, v in old_comp.items():
                        if k not in comp:
                            comp[k] = v
                    new = dict(old)
                    new["seen_metrics"], new["unseen_metrics"], new["comparison"] = sm, um, comp
                    new["temporal_split"] = args.date
                    si = new.get("sampling_info")
                    if isinstance(si, dict):
                        si["original_seen_count"] = len(new_seen)
                        si["original_unseen_count"] = len(new_unseen)
                        si["sampled_seen_count"] = len(b_seen)
                        si["sampled_unseen_count"] = len(b_unseen)
                        si["labels"] = common
                        si["per_label_info"] = info
                        si["sampled_seen_claim_ids"] = [r.get("claim_id") for r in b_seen]
                        si["sampled_unseen_claim_ids"] = [r.get("claim_id") for r in b_unseen]
                    stats["summaries_rebuilt"] += 1
                    rebuilt_summaries.add(pbs)
                    if not args.dry_run:
                        dump(pbs, new)
                except Exception as e:
                    print(f"  SKIP posthoc summary {pbs.relative_to(ROOT)}: {e}")
            # sidecar counts
            if sidecar.exists() and not args.dry_run:
                try:
                    sc = load_json(sidecar)
                    sc["original_seen_count"] = len(new_seen)
                    sc["original_unseen_count"] = len(new_unseen)
                    sc["sampled_seen_count"] = len(b_seen)
                    sc["sampled_unseen_count"] = len(b_unseen)
                    sc["labels"] = common
                    sc["per_label_info"] = info
                    dump(sidecar, sc)
                except Exception as e:
                    print(f"  SKIP sidecar {sidecar.relative_to(ROOT)}: {e}")

    # -- combined summaries ----------------------------------------------------
    for cf in sorted(provider.rglob("combined_summary.json")):
        try:
            old = json.loads(file_text(cf))
        except Exception:
            continue
        changed = False
        for method in list(old.get("methods", {}).keys()):
            cands = [cf.parent / method]
            if cf.parent.name == method:
                cands.append(cf.parent)
            mdir = next((d for d in cands if (d / "summary.json") in rebuilt_summaries
                         or (d / "summary.json").exists() and d / "summary.json" in rebuilt_summaries), None)
            if mdir is None:
                # rebuild only if that method was fixed in this run: check sibling summary rebuilt
                mdir2 = next((d for d in cands if d.is_dir()), None)
                if mdir2 is None or (mdir2 / "summary.json") not in rebuilt_summaries:
                    continue
                mdir = mdir2
            try:
                s = load_json(mdir / "summary.json")
            except Exception:
                continue
            entry = {"seen_metrics": s.get("seen_data"), "unseen_metrics": s.get("unseen_data"),
                     "comparison": s.get("comparison"),
                     "full_data": {"seen_metrics": s.get("seen_data"),
                                   "unseen_metrics": s.get("unseen_data"),
                                   "comparison": s.get("comparison")}}
            pb = mdir / "posthoc_balanced" / "posthoc_balanced_summary.json"
            if pb.exists():
                try:
                    p = load_json(pb)
                    entry["posthoc_balanced"] = {"seen_metrics": p.get("seen_metrics"),
                                                 "unseen_metrics": p.get("unseen_metrics"),
                                                 "comparison": p.get("comparison")}
                except Exception:
                    pass
            elif "posthoc_balanced" in old["methods"][method]:
                entry["posthoc_balanced"] = old["methods"][method]["posthoc_balanced"]
            old["methods"][method] = entry
            changed = True
        if changed:
            splits = set()
            for mdir_cand in [cf.parent / m for m in old["methods"]] + (
                    [cf.parent] if cf.parent.name in old["methods"] else []):
                sf = mdir_cand / "summary.json"
                if sf.exists():
                    try:
                        ei = load_json(sf).get("experiment_info", {})
                        splits.add(ei.get("temporal_split_date", ei.get("temporal_split")))
                    except Exception:
                        pass
            if len(splits) == 1:
                old.get("experiment_info", {})["temporal_split"] = splits.pop()
            stats["combined_rebuilt"] += 1
            if not args.dry_run:
                dump(cf, old)

    print(f"\n==== DeepSeek split fix -> {args.date} "
          + ("(DRY RUN)" if args.dry_run else "(APPLIED)") + " ====")
    print(f"method dirs scanned : {stats['methods']}")
    print(f"methods fixed       : {stats['methods_fixed']}")
    print(f"pred files rewritten: {stats['pred_files_rewritten']}")
    print(f"CSV files rewritten : {stats['csv_rewritten']}")
    print(f"rows moved seen<->unseen: {stats['rows_moved']}")
    print(f"metrics recomputed  : {stats['metrics_recomputed']}")
    print(f"summaries rebuilt   : {stats['summaries_rebuilt']}")
    print(f"combined rebuilt    : {stats['combined_rebuilt']}")
    print("Run logs (*.txt), source code and data/ left untouched.")


if __name__ == "__main__":
    sys.exit(main())
