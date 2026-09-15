from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_DATA = BASE_DIR / "data" / "processed" / "train.csv"
CURRENT_DATA = BASE_DIR / "data" / "processed" / "test.csv"

REPORT_DIR = BASE_DIR / "monitoring" / "reports"
REPORT_FILE = REPORT_DIR / "data_drift_report.html"


DRIFT_COLUMNS = [
    "interaction_score",
    "interaction_count",
    "viewed",
    "added_to_cart",
    "purchased",
]


def load_data():
    reference = pd.read_csv(REFERENCE_DATA)
    current = pd.read_csv(CURRENT_DATA)

    return reference, current


def generate_drift_report(reference, current):
    reference = reference[DRIFT_COLUMNS].copy()
    current = current[DRIFT_COLUMNS].copy()

    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )

    print("Snapshot type:", type(snapshot))
    print("Snapshot methods:")
    print([
        method
        for method in dir(snapshot)
        if "html" in method.lower() or "save" in method.lower()
    ])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot.save_html(str(REPORT_FILE))

    print("File exists:", REPORT_FILE.exists())
    print("File size:", REPORT_FILE.stat().st_size if REPORT_FILE.exists() else 0)

if __name__ == "__main__":
    print("Loading reference and current datasets...")

    reference, current = load_data()

    print(f"Reference dataset: {reference.shape}")
    print(f"Current dataset:   {current.shape}")

    print("Generating data drift report...")

    generate_drift_report(reference, current)

    print(f"Report saved to: {REPORT_FILE}")