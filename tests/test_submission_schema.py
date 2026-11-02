import pandas as pd
import pytest

from meniscinet.cli import validate_submission
from meniscinet.constants import TARGETS


def test_validate_submission_accepts_valid_file(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame([["study-1", *([0.5] * len(TARGETS))]],
                 columns=["StudyInstanceUID", *TARGETS]).to_csv(path, index=False)
    validate_submission(path, expected_rows=1)


def test_validate_submission_rejects_invalid_probability(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame([["study-1", *([2.0] * len(TARGETS))]],
                 columns=["StudyInstanceUID", *TARGETS]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="probabilities"):
        validate_submission(path)
