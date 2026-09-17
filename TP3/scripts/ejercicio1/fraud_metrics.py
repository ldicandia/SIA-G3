def confusion_counts(y_true: list, y_prob: list, threshold: float) -> dict:
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for yt, yp in zip(y_true, y_prob):
        actual = int(yt) == 1
        predicted = float(yp) >= float(threshold)
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

def precision_recall_f1(y_true: list, y_prob: list, threshold: float) -> dict:
    counts = confusion_counts(y_true, y_prob, threshold)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def sweep(y_true: list, y_prob: list, thresholds: list[float]) -> list[dict]:
    results = []
    for t in thresholds:
        metrics = precision_recall_f1(y_true, y_prob, t)
        results.append({"threshold": t, **metrics})
    return results

def best_threshold_by_f1(sweep_results: list[dict]) -> dict:
    if not sweep_results:
        raise ValueError("Cannot find best threshold in empty sweep results")
    # Maximize F1; break ties with the smaller threshold value
    return max(sweep_results, key=lambda r: (r["f1"], -r["threshold"]))
