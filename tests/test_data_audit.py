import pandas as pd

from meniscinet.constants import TARGETS
from meniscinet.data.audit import audit_competition_data


def test_audit_reports_expected_counts(tmp_path):
    train = pd.DataFrame([
        ["train-gold", "report", *([1] * len(TARGETS))],
        ["train-unlabelled", "report", *([None] * len(TARGETS))],
    ], columns=["StudyInstanceUID", "Report", *TARGETS])
    train.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["test-1"]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["train-gold"], "SeriesInstanceUID": ["series-1"]}).to_csv(tmp_path / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["test-1"], "SeriesInstanceUID": ["series-2"]}).to_csv(tmp_path / "test_series.csv", index=False)

    report = audit_competition_data(tmp_path)

    assert report["gold_labelled_train_studies"] == 1
    assert report["train_studies_without_series"] == 1
    assert report["test_studies_without_series"] == 0
