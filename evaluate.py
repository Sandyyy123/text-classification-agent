"""Per-category precision / recall / F1 on the held-out test set.

Run:  python evaluate.py
"""
from __future__ import annotations

from collections import defaultdict

from classifier import HybridClassifier
from data import TEST, TRAIN


def evaluate(clf: HybridClassifier, test: list[tuple[str, str]]) -> dict:
    labels = sorted({lbl for _, lbl in test} | set(clf.labels))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0

    for text, gold in test:
        pred = clf.predict(text)
        if pred.label == gold:
            tp[gold] += 1
            correct += 1
        else:
            fp[pred.label] += 1
            fn[gold] += 1

    per_label = {}
    for lbl in labels:
        prec = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) else 0.0
        rec = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label[lbl] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": tp[lbl] + fn[lbl]}

    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label)
    return {
        "accuracy": correct / len(test),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "n": len(test),
    }


def main() -> None:
    clf = HybridClassifier(TRAIN)
    report = evaluate(clf, TEST)
    print(f"\nText-classification agent - evaluation on {report['n']} held-out items")
    print("=" * 58)
    print(f"{'category':<14}{'precision':>11}{'recall':>9}{'f1':>8}{'n':>5}")
    print("-" * 58)
    for lbl, m in report["per_label"].items():
        print(f"{lbl:<14}{m['precision']:>11.2f}{m['recall']:>9.2f}"
              f"{m['f1']:>8.2f}{m['support']:>5}")
    print("-" * 58)
    print(f"{'accuracy':<14}{report['accuracy']:>11.2f}")
    print(f"{'macro F1':<14}{report['macro_f1']:>11.2f}")
    print()


if __name__ == "__main__":
    main()
