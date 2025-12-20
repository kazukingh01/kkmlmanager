from kktestdata import DatasetRegistry
from kkmlmanager.features import get_features_by_correlation


if __name__ == "__main__":
    reg = DatasetRegistry()
    dataset = reg.create("boatrace_original_20210101_20210630")
    df = dataset.load_data("polars")

    df_corr1 = get_features_by_correlation(df[:, :1000], dtype="float32", is_gpu=False, corr_type="pearson",    batch_size=100,  min_n=100, n_jobs=8)
    df_corr2 = get_features_by_correlation(df[:, :1000], dtype="float32", is_gpu=False, corr_type="spearman",   batch_size=100,  min_n=100, n_jobs=8)
    df_corr3 = get_features_by_correlation(df[:, :1000], dtype="float32", is_gpu=False, corr_type="chatterjee", batch_size=100,  min_n=100, n_jobs=8)
    df_corr4 = get_features_by_correlation(df[:, :1000], dtype="float32", is_gpu=True,  corr_type="pearson",    batch_size=1000, min_n=100, n_jobs=8)
    df_corr5 = get_features_by_correlation(df[:, :1000], dtype="float32", is_gpu=True,  corr_type="spearman",   batch_size=1000, min_n=100, n_jobs=8)
    df_corr6 = get_features_by_correlation(df[:, :1000], dtype="float32", is_gpu=True,  corr_type="chatterjee", batch_size=1000, min_n=100, n_jobs=8)
