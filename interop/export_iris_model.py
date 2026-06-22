"""
Train a tiny scikit-learn LogisticRegression on the Iris dataset and export it to ONNX.

Outputs:
  - interop/iris_logreg.onnx   - the exported model (consumed by the .NET demo)
  - interop/iris_reference.json - reference predictions for a fixed set of inputs,
                                  so the .NET side can prove it gets bit-identical answers.

Run from repo root:
  ./.venv/Scripts/python.exe interop/export_iris_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skl2onnx import to_onnx


OUTPUT_DIR = Path(__file__).parent
ONNX_PATH = OUTPUT_DIR / "iris_logreg.onnx"
REFERENCE_PATH = OUTPUT_DIR / "iris_reference.json"


def main() -> None:
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy:  {test_acc:.4f}")

    # Export to ONNX. options pin label output to int64 and probabilities to a flat tensor
    # (default skl2onnx wraps probabilities in a ZipMap producing a sequence-of-maps, which
    # is awkward to consume from a generic wrapper).
    onnx_model = to_onnx(
        clf,
        X_train[:1],
        target_opset=18,
        options={id(clf): {"zipmap": False}},
    )
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())
    print(f"wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size} bytes)")

    # Pick a small fixed reference batch so the .NET side has something deterministic to match.
    samples = X_test[:5]
    expected_labels = clf.predict(samples).astype(int).tolist()
    expected_probs = clf.predict_proba(samples).astype(float).tolist()

    reference = {
        "feature_names": iris.feature_names,
        "class_names": iris.target_names.tolist(),
        "samples": samples.astype(float).tolist(),
        "expected_labels": expected_labels,
        "expected_probabilities": expected_probs,
    }
    REFERENCE_PATH.write_text(json.dumps(reference, indent=2))
    print(f"wrote {REFERENCE_PATH}")

    print("\nfirst sample:")
    print(f"  features:    {samples[0].tolist()}")
    print(f"  label:       {expected_labels[0]} ({iris.target_names[expected_labels[0]]})")
    print(f"  probs:       {[round(p, 4) for p in expected_probs[0]]}")


if __name__ == "__main__":
    main()
