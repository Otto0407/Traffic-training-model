import pandas as pd
import pytest
import torch

from config import CLASS_ORDER
from model import CongestionMLP
from predict import load_trained, predict_table, pseudonymise, read_table


@pytest.fixture(scope="module")
def tiny_model(fitted_preprocessor, small_df):
    from features import select_features
    torch.manual_seed(0)
    input_dim = fitted_preprocessor.transform(select_features(small_df.head(1))[0]).shape[1]
    model = CongestionMLP(input_dim, [16], len(CLASS_ORDER), 0.0)
    model.eval()
    return model


def test_end_to_end_on_messy_semicolon_cp1252_file(tmp_path, small_df, fitted_preprocessor, tiny_model, cpu):
    rows = small_df.sample(20, random_state=2).reset_index(drop=True).copy()
    rows = rows.astype(str)
    rows["driver_name"] = "José Müller"
    rows["plate"] = [f"XYZ-{i:03d}" for i in range(20)]
    rows.loc[3, ["hour", "timestamp"]] = ""
    path = tmp_path / "in.csv"
    rows.to_csv(path, sep=";", index=False, encoding="cp1252")

    result, report = predict_table(read_table(path), tiny_model, fitted_preprocessor, cpu)
    assert len(result) == 20 and report["rows_predicted"] == 19
    assert pd.isna(result.loc[3, "predicted_congestion_level"])
    assert result["predicted_congestion_level"].dropna().isin(CLASS_ORDER).all()
    assert result["confidence"].dropna().between(0.25, 1).all()
    assert set(result.columns) == {"n_features_imputed", "data_quality", "predicted_congestion_level", "confidence"}
    out = result.to_csv(index=False)
    for secret in ("José", "Müller", "XYZ-000"):
        assert secret not in out


def test_reads_tab_separated_utf8(tmp_path):
    (tmp_path / "t.tsv").write_text("hour\tweather\n8\tClear\n", encoding="utf-8")
    df = read_table(tmp_path / "t.tsv")
    assert list(df.columns) == ["hour", "weather"] and df.iloc[0, 0] == "8"


def test_pseudonymised_ids_are_stable_and_unreadable():
    ids = pd.Series(["REC_1", "REC_2"])
    assert (pseudonymise(ids, "k1") == pseudonymise(ids, "k1")).all()
    assert (pseudonymise(ids, "k1") != pseudonymise(ids, "k2")).all()
    assert not pseudonymise(ids, "k1").str.contains("REC").any()


def test_saved_model_round_trip(tmp_path, fitted_preprocessor, tiny_model, cpu):
    import joblib
    from dataclasses import asdict, replace
    from config import TrainConfig

    cfg = asdict(replace(TrainConfig(), hidden_sizes=[16], dropout=0.0))
    input_dim = tiny_model.net[0].in_features
    torch.save({"state_dict": tiny_model.state_dict(), "input_dim": input_dim, "config": cfg}, tmp_path / "model.pt")
    joblib.dump(fitted_preprocessor, tmp_path / "preprocessor.joblib")
    model, pre = load_trained(tmp_path, cpu)
    x = torch.randn(3, input_dim)
    assert torch.allclose(model(x), tiny_model(x))
    assert not model.training
