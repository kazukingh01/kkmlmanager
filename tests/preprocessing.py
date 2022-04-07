import pandas as pd
import numpy as np
from kkmlmanager.manager import MLManager


if __name__ == "__main__":
    # create dataframe
    ncols, nrows, nclass = 10, 2000, 5
    df = pd.DataFrame(np.random.rand(nrows, ncols), columns=[f"col_{i}" for i in range(ncols)])
    df["col_nan1"] = float("nan")
    df["col_nan2"] = float("nan")
    df.loc[np.random.permutation(np.arange(nrows))[:nrows//20], "col_nan2"] = np.random.rand(nrows//20)
    df["col_all1"] = 1
    df["col_sqrt"] = df["col_0"].pow(1/2)
    df["col_pw2"]  = df["col_0"].pow(2)
    df["col_pw3"]  = df["col_0"].pow(3)
    df["col_log"]  = np.log(df["col_0"].values)
    df["answer"]   = np.random.randint(0, nclass, nrows)
    """
    >>> df.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 18 columns):
    #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
    0   col_0     2000 non-null   float64
    1   col_1     2000 non-null   float64
    2   col_2     2000 non-null   float64
    3   col_3     2000 non-null   float64
    4   col_4     2000 non-null   float64
    5   col_5     2000 non-null   float64
    6   col_6     2000 non-null   float64
    7   col_7     2000 non-null   float64
    8   col_8     2000 non-null   float64
    9   col_9     2000 non-null   float64
    10  col_nan1  0 non-null      float64
    11  col_nan2  100 non-null    float64
    12  col_all1  2000 non-null   int64  
    13  col_sqrt  2000 non-null   float64
    14  col_pw2   2000 non-null   float64
    15  col_pw3   2000 non-null   float64
    16  col_log   2000 non-null   float64
    17  answer    2000 non-null   int64  
    dtypes: float64(16), int64(2)
    """

    # set manager
    manager = MLManager(df.columns[df.columns.str.contains("^col_")].tolist(), "answer")
    """
    >>> manager.columns
    array([ 'col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6',
            'col_7', 'col_8', 'col_9', 'col_nan1', 'col_nan2', 'col_all1',
            'col_sqrt', 'col_pw2', 'col_pw3', 'col_log'], dtype='<U8')
    """

    # cut features by valiance ( except nan )
    manager.cut_features_by_variance(df, cutoff=0.8, ignore_nan=True, batch_size=5)
    """
    >>> manager.columns
    array([ 'col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6',
            'col_7', 'col_8', 'col_9', 'col_nan2', 'col_sqrt', 'col_pw2',
            'col_pw3', 'col_log'], dtype=object)
    >>> manager.features_var_True_08
    col_0       False
    col_1       False
    col_2       False
    col_3       False
    col_4       False
    col_5       False
    col_6       False
    col_7       False
    col_8       False
    col_9       False
    col_nan1     True
    col_nan2    False
    col_all1     True
    col_sqrt    False
    col_pw2     False
    col_pw3     False
    col_log     False
    dtype: bool
    """    

    # cut features by valiance
    manager.cut_features_by_variance(df, cutoff=0.8, ignore_nan=False, batch_size=5)
    """
    >>> manager.columns
    array([ 'col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6',
            'col_7', 'col_8', 'col_9', 'col_sqrt', 'col_pw2', 'col_pw3',
            'col_log'], dtype=object)
    >>> manager.features_var_False_08
    col_0       False
    col_1       False
    col_2       False
    col_3       False
    col_4       False
    col_5       False
    col_6       False
    col_7       False
    col_8       False
    col_9       False
    col_nan2     True
    col_sqrt    False
    col_pw2     False
    col_pw3     False
    col_log     False
    dtype: bool
    """

    # cut features by correltion ( pearson )
    manager.cut_features_by_correlation(df, cutoff=0.95, sample_size=None, dtype="float32", is_gpu=False, corr_type="pearson",  batch_size=1)
    """
    >>> manager.columns
    array([ 'col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6',
            'col_7', 'col_8', 'col_9', 'col_log'], dtype=object)
    >>> manager.features_corr_pearson
              col_0     col_1     col_2     col_3     col_4     col_5     col_6     col_7     col_8     col_9  col_sqrt   col_pw2   col_pw3   col_log
    col_0       NaN -0.000308  0.007505  0.010611 -0.000173 -0.027433 -0.038541 -0.005915 -0.008437  0.016328  0.979890  0.969353  0.919349  0.854601
    col_1       NaN       NaN -0.021343 -0.019627  0.001560 -0.042358  0.010089 -0.017834  0.014259 -0.004822  0.000324  0.001133  0.002325  0.004356
    col_2       NaN       NaN       NaN  0.021969  0.021066  0.005496 -0.017845 -0.004642  0.013540  0.032147  0.004499  0.011287  0.011933  0.005272
    col_3       NaN       NaN       NaN       NaN  0.004975  0.001423  0.017767 -0.017592  0.017696 -0.011438  0.012703  0.007073  0.004867  0.015482
    col_4       NaN       NaN       NaN       NaN       NaN  0.006551 -0.009123  0.022668  0.030991 -0.004802  0.001638 -0.006857 -0.013398 -0.002357
    col_5       NaN       NaN       NaN       NaN       NaN       NaN -0.064477  0.002730  0.051351  0.007732 -0.035086 -0.015685 -0.009141 -0.039109
    col_6       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.014296  0.008590 -0.010900 -0.036591 -0.038864 -0.038426 -0.028100
    col_7       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN -0.000115  0.028498 -0.011965  0.002035  0.006265 -0.019044
    col_8       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.019168 -0.014118 -0.003434 -0.002789 -0.023429
    col_9       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.016352  0.019918  0.022548  0.023011
    col_sqrt    NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.905854  0.836163  0.933576
    col_pw2     NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.986466  0.737173
    col_pw3     NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.656739
    col_log     NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
    """

    # cut features by correltion ( spearman ). GPU only
    # manager.cut_features_by_correlation(df, cutoff=0.95, sample_size=None, dtype="float32", is_gpu=True, corr_type="spearman", batch_size=1)
    # cut features by adversarial validation
    df_test = df.copy()
    df_test.loc[:, manager.columns[0]] += 1
    manager.cut_features_by_adversarial_validation(df, df_test=df_test, cutoff=100, n_split=5, n_cv=3)
    """
    >>> manager.columns
    array([ 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7',
            'col_8', 'col_9', 'col_log'], dtype=object)
    >>> manager.features_adversarial
            count  importance    ratio
    col_0      300    479151.0  1597.17
    col_1       99         0.0     0.00
    col_2       60         0.0     0.00
    col_3      111         0.0     0.00
    col_4       54         0.0     0.00
    col_5      108         0.0     0.00
    col_6       99         0.0     0.00
    col_7       84         0.0     0.00
    col_8       66         0.0     0.00
    col_9       93         0.0     0.00
    col_log     75         0.0     0.00
    """

    # sort and cut features by random tree importance
    manager.cut_features_by_randomtree_importance(df, cutoff=0.9, max_iter=1, min_count=100)
    """
    >>> manager.columns
    array([ 'col_7', 'col_6', 'col_4', 'col_3', 'col_8', 'col_2', 'col_9',
            'col_1', 'col_5'], dtype=object)
    >>> manager.features_treeimp
            count   importance     ratio
    col_7    4177.0  4951.642142  1.185454
    col_6    4082.0  4758.425222  1.165709
    col_4    4081.0  4749.604940  1.163834
    col_3    4089.0  4742.047617  1.159708
    col_8    4061.0  4697.331351  1.156693
    col_2    3965.0  4569.806983  1.152536
    col_9    4141.0  4768.886678  1.151627
    col_1    4029.0  4636.219213  1.150712
    col_5    4041.0  4618.554522  1.142924
    col_log  4018.0  4514.304347  1.123520
    """

    # log
    """
    >>> print(manager.logger.internal_stream.getvalue())
    2022-04-06 11:33:42,263 - MLManager.139756623862944 - __init__ - INFO : START
    2022-04-06 11:33:42,263 - MLManager.139756623862944 - initialize - INFO : START
    2022-04-06 11:33:42,263 - MLManager.139756623862944 - proc_check_init - INFO : START
    2022-04-06 11:33:42,264 - MLManager.139756623862944 - proc_check_init - INFO : END
    2022-04-06 11:33:42,264 - MLManager.139756623862944 - initialize - INFO : END
    2022-04-06 11:33:42,264 - MLManager.139756623862944 - __init__ - INFO : END
    2022-04-06 11:33:42,264 - MLManager.139756623862944 - cut_features_by_variance - INFO : START
    2022-04-06 11:33:42,264 - MLManager.139756623862944 - cut_features_by_variance - INFO : df: (2000, 18), cutoff: 0.8, ignore_nan: True
    2022-04-06 11:33:42,406 - MLManager.139756623862944 - cut_features_by_variance - INFO : feature: col_nan1, n nan: 2000, max count: nan, value unique: [nan]
    2022-04-06 11:33:42,407 - MLManager.139756623862944 - cut_features_by_variance - INFO : feature: col_all1, n nan: 0, max count: (1, 2000), value unique: [1]
    2022-04-06 11:33:42,407 - MLManager.139756623862944 - update_features - INFO : START
    2022-04-06 11:33:42,408 - MLManager.139756623862944 - update_features - INFO : columns new: (15,), columns before:(17,)
    2022-04-06 11:33:42,408 - MLManager.139756623862944 - update_features - INFO : END
    2022-04-06 11:33:42,408 - MLManager.139756623862944 - cut_features_by_variance - INFO : END
    2022-04-06 11:33:42,408 - MLManager.139756623862944 - cut_features_by_variance - INFO : START
    2022-04-06 11:33:42,408 - MLManager.139756623862944 - cut_features_by_variance - INFO : df: (2000, 18), cutoff: 0.8, ignore_nan: False
    2022-04-06 11:33:42,784 - MLManager.139756623862944 - cut_features_by_variance - INFO : feature: col_nan2, n nan: 1900, max count: (0.7571598392085422, 1), value unique: [       nan 0.75715984 0.10512504 0.70688389 0.16159426]
    2022-04-06 11:33:42,784 - MLManager.139756623862944 - update_features - INFO : START
    2022-04-06 11:33:42,784 - MLManager.139756623862944 - update_features - INFO : columns new: (14,), columns before:(15,)
    2022-04-06 11:33:42,784 - MLManager.139756623862944 - update_features - INFO : END
    2022-04-06 11:33:42,784 - MLManager.139756623862944 - cut_features_by_variance - INFO : END
    2022-04-06 11:33:42,784 - MLManager.139756623862944 - cut_features_by_correlation - INFO : START
    2022-04-06 11:33:42,786 - MLManager.139756623862944 - cut_features_by_correlation - INFO : df: (2000, 14), cutoff: 0.95, dtype: float32, is_gpu: False, corr_type: pearson
    2022-04-06 11:33:42,804 - MLManager.139756623862944 - cut_features_by_correlation - INFO : feature: col_sqrt, compare: col_0, corr: 0.9804840592235514
    2022-04-06 11:33:42,805 - MLManager.139756623862944 - cut_features_by_correlation - INFO : feature: col_pw2, compare: col_0, corr: 0.96928806694477
    2022-04-06 11:33:42,807 - MLManager.139756623862944 - cut_features_by_correlation - INFO : feature: col_pw3, compare: col_pw2, corr: 0.9862313087292189
    2022-04-06 11:33:42,807 - MLManager.139756623862944 - update_features - INFO : START
    2022-04-06 11:33:42,808 - MLManager.139756623862944 - update_features - INFO : columns new: (11,), columns before:(14,)
    2022-04-06 11:33:42,808 - MLManager.139756623862944 - update_features - INFO : END
    2022-04-06 11:33:42,808 - MLManager.139756623862944 - cut_features_by_correlation - INFO : END
    2022-04-06 11:33:42,810 - MLManager.139756623862944 - cut_features_by_adversarial_validation - INFO : START
    2022-04-06 11:33:44,343 - MLManager.139756623862944 - cut_features_by_adversarial_validation - INFO : feature: col_0, ratio: 1597.17
    2022-04-06 11:33:44,343 - MLManager.139756623862944 - update_features - INFO : START
    2022-04-06 11:33:44,343 - MLManager.139756623862944 - update_features - INFO : columns new: (10,), columns before:(11,)
    2022-04-06 11:33:44,344 - MLManager.139756623862944 - update_features - INFO : END
    2022-04-06 11:33:44,344 - MLManager.139756623862944 - cut_features_by_adversarial_validation - INFO : END
    2022-04-06 11:33:44,344 - MLManager.139756623862944 - cut_features_by_randomtree_importance - INFO : START
    2022-04-06 11:33:44,344 - MLManager.139756623862944 - cut_features_by_randomtree_importance - INFO : df: (2000, 18), cutoff: 0.9, max_iter: 1, min_count: 100
    2022-04-06 11:33:44,630 - MLManager.139756623862944 - cut_features_by_randomtree_importance - INFO : feature: col_log, ratio: 1.1166531231787227
    2022-04-06 11:33:44,630 - MLManager.139756623862944 - update_features - INFO : START
    2022-04-06 11:33:44,630 - MLManager.139756623862944 - update_features - INFO : columns new: (9,), columns before:(10,)
    2022-04-06 11:33:44,630 - MLManager.139756623862944 - update_features - INFO : END
    2022-04-06 11:33:44,630 - MLManager.139756623862944 - cut_features_by_randomtree_importance - INFO : END
    """
