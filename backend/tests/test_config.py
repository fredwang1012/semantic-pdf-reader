from app.config import ROOT, Settings


def test_relative_data_dir_resolves_against_the_repo_root():
    assert Settings(data_dir="data").data_dir == ROOT / "data"


def test_absolute_data_dir_is_kept(tmp_path):
    assert Settings(data_dir=tmp_path).data_dir == tmp_path


def test_reranking_is_opt_in():
    assert Settings.model_fields["rerank"].default == "off"
