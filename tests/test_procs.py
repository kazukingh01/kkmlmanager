import datetime 
from zoneinfo import ZoneInfo
import polars as pl
import numpy as np
from kkmlmanager.procs import ProcMinMaxScaler, ProcStandardScaler, ProcRankGauss, \
    ProcPCA, ProcOneHotEncoder, ProcFillNa, ProcFillNaMinMaxRandomly, ProcReplaceValue, \
    ProcReplaceInf, ProcToValues, ProcMap, ProcAsType, ProcReshape, ProcDropNa, \
    ProcCondition, ProcEval


def create_test_df_polars():
    df = pl.DataFrame({
        "id": [
            1, 2, 3, 4, 5, 6
        ],
        "datetime_no_nan": [
            datetime.datetime(2023, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC")),
            datetime.datetime(2023, 5, 5, 12, 30, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime.datetime(2024, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("Asia/Tokyo")),
            datetime.datetime(2025, 7, 4, 18, 0, 0),
            datetime.datetime(2025, 7, 4, 18, 0, 0, tzinfo=ZoneInfo("Europe/Paris")),
            datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("UTC")),
        ],
        "datetime_with_nan": [
            datetime.datetime(2023, 1, 2, 0, 0, 0),
            None,
            datetime.datetime(2024, 1, 1, 13, 30, 30, tzinfo=ZoneInfo("Asia/Tokyo")),
            None,
            datetime.datetime(2023, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("America/Los_Angeles")),
            datetime.datetime(2023, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("Europe/Paris")),
        ],
        "int_no_nan": [
            0,
            100,
            -50,
            999999,
            2**31 - 1,  # 32bit int max
            -2**31,     # 32bit int min
        ],
        "int_with_nan": [42, None, 0, 123456, None, -999],
        "float_no_nan": [
            1.23,
            1e20,
            1e-20,
            np.inf,
            -np.inf,
            9999.9999,
        ],
        "float_with_nan": [
            None,
            0.1,
            np.pi,
            -42.42,
            1e-10,
            1e-5,
        ],
        "float_normal1": np.linspace(0, 2, 6),
        "float_normal2": np.linspace(2, 4, 6),
        "str_no_nan": [
            "Hello",
            "123'ABC",
            "Special chars: !@#$%^&*()",
            "日本語\n改行",
            "Emoji🔥 \"quoted\" text",
            "Mixedあいう123\\slash"
        ],
        "str_with_nan": [
            None,
            "foo's bar",
            None,
            "line1\nline2",
            "He said \"Hi\"",
            "escape\\slash"
        ],
        "bool_no_nan": [
            True,
            False,
            True,
            True,
            False,
            True
        ],
        "bool_with_nan": [
            False,
            None,
            True,
            None,
            False,
            True
        ],
        "category_column": ["A", "B", "A", "C", "B", "A"]
    })
    df = df.with_columns(pl.col("category_column").cast(pl.Categorical))
    return df


if __name__ == "__main__":
    df_org  = create_test_df_polars()
    DICTCOL = {x: i for i, x in enumerate(df_org.columns)}
    def columns_coverter(columns, itype: str, flg: int=0):
        if itype == "np":
            if flg == 1:
                return [DICTCOL[x] for x in columns]
            elif flg == 2:
                return DICTCOL[columns]
            else:
                return slice(None), [DICTCOL[x] for x in columns]
        else:
            return columns
    for df, itype in zip([df_org.clone(), df_org.clone().to_pandas(), df_org.clone().to_numpy()], ["pl", "pd", "np"]):
        proc = ProcMinMaxScaler()
        proc.fit(df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)])
        proc    (df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)])
        proc = ProcStandardScaler()
        proc.fit(df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)])
        proc    (df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)])
        proc = ProcRankGauss()
        proc.fit(df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)])
        proc    (df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)])
        proc = ProcPCA()
        proc.fit(df[columns_coverter(['float_normal1', 'float_normal2'], itype)])
        proc    (df[columns_coverter(['float_normal1', 'float_normal2'], itype)])
        proc = ProcOneHotEncoder()
        proc.fit(df[columns_coverter(['id'], itype)])
        proc    (df[columns_coverter(['id'], itype)])
        if itype == "np":
            for x in ["mean", "max", "min", "median", 1, 2.2, [1,1,1,1]]:
                proc = ProcFillNa(x)
                proc.fit(df[columns_coverter(['float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2'], itype)].astype(float))
                proc    (df[columns_coverter(['float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2'], itype)].astype(float))
        else:
            for x in [
                "mean", "max", "min", "median", 1, 2.2,
                [1, datetime.datetime.now(tz=datetime.UTC), datetime.datetime.now(), 1, 1, 1, 1, 1, 1, 1, 1, True, False, "A"]
            ]:
                proc = ProcFillNa(x)
                proc.fit(df)
                proc    (df)
        if itype == "np":
            proc = ProcFillNaMinMaxRandomly()
            proc.fit(df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)].astype(float))
            proc    (df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)].astype(float))
        else:
            try:
                proc = ProcFillNaMinMaxRandomly()
                proc.fit(df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_with_nan'], itype)])
                raise
            except TypeError:
                pass
        proc = ProcReplaceValue({float("inf"): 100, float("-inf"): -100}, columns=columns_coverter(['float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2'], itype, flg=1))
        proc.fit(df)
        proc    (df)
        proc = ProcReplaceValue({
            columns_coverter('float_no_nan',   itype, flg=2): {float("inf"): 100, float("-inf"): -100},
            columns_coverter('float_with_nan', itype, flg=2): {float("inf"): 100, float("-inf"): -100},
        })
        proc.fit(df)
        proc    (df)
        proc = ProcReplaceInf()
        proc.fit(df)
        proc    (df)
        if itype == "np":
            try:
                proc = ProcToValues()
                proc.fit(df)
                raise
            except TypeError:
                pass
        else:
            proc = ProcToValues()
            proc.fit(df)
            proc    (df)
        proc = ProcMap({"Hello": "D"}, columns_coverter("str_no_nan", itype, flg=2), fill_null="E")
        proc.fit(df)
        proc    (df)
        if itype == "pl":
            proc = ProcAsType(pl.Float32, columns=columns_coverter(['int_no_nan', 'int_with_nan', 'float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2'], itype, flg=1))
            proc.fit(df)
            proc    (df)
        elif itype == "pd":
            proc = ProcAsType(np.float32, columns=columns_coverter(['int_no_nan', 'int_with_nan', 'float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2'], itype, flg=1))
            proc.fit(df)
            proc    (df)
        else:
            proc = ProcAsType(np.float32)
            proc.fit(df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2'], itype)])
            proc    (df[columns_coverter(['int_no_nan', 'int_with_nan', 'float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2'], itype)])
        if itype == "np":
            proc = ProcDropNa(columns=columns_coverter(['int_with_nan', 'float_with_nan'], itype, flg=1))
            proc.fit(df)
            proc    (df)
        else:
            proc = ProcDropNa(columns=columns_coverter(["str_with_nan", "bool_with_nan"], itype, flg=1))
            proc.fit(df)
            proc    (df)
        if itype == "np":
            try:
                proc = ProcCondition("int_no_nan > 0")
                proc.fit(df)
                raise
            except TypeError:
                pass
        else:
            proc = ProcCondition("int_no_nan > 0")
            proc.fit(df)
            proc    (df)
        if itype == "np":
            proc = ProcReshape(-1)
            proc.fit(df)
            proc    (df)
        else:
            try:
                proc = ProcReshape(-1)
                proc.fit(df)
                raise
            except TypeError:
                pass
        proc = ProcEval("np.array(__input.shape)")
        proc.fit(df)
        proc    (df)
