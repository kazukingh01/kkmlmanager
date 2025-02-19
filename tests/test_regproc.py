import numpy as np
import polars as pl
from kkmlmanager.regproc import RegistryProc
from kkmlmanager.procs import ProcFillNa, ProcReplaceInf, ProcToValues, ProcAsType
from test_procs import create_test_df_polars

if __name__ == "__main__":
    df = create_test_df_polars()
    rp = RegistryProc(n_jobs=2, is_auto_colslct=True)
    rp.register(ProcAsType(pl.Float32, columns=['int_no_nan', 'int_with_nan', 'float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2']))
    rp.register(ProcReplaceInf())
    rp.register(ProcToValues())
    rp.register(ProcFillNa(-100))
    rp.register(ProcAsType(np.float64))
    rp.fit(df[:, 3:-5])
    dfwk1 = rp(df)
    rp = RegistryProc(n_jobs=2, is_auto_colslct=True)
    rp.register(ProcAsType(np.float32, columns=['int_no_nan', 'int_with_nan', 'float_no_nan', 'float_with_nan', 'float_normal1', 'float_normal2']))
    rp.register(ProcReplaceInf())
    rp.register(ProcToValues())
    rp.register(ProcFillNa(-100))
    rp.register(ProcAsType(np.float64))
    rp.fit(df[:, 3:-5].to_pandas())
    dfwk2 = rp(df.to_pandas())
    print(np.all(dfwk1 == dfwk2))
