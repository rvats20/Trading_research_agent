import numpy as np
import pandas as pd

from src.outcomes import add_forward_outcomes

def test_forward_return_alignment():
    df = pd.DataFrame({"close": [100., 110., 121., 133.1, 146.41]})
    result = add_forward_outcomes(df, horizons=(1, 2))

    assert np.isclose(result.loc[0, "fwd_return_1d"], 0.10)
    assert np.isclose(result.loc[0, "fwd_return_2d"], 0.21)
    assert np.isclose(result.loc[2, "fwd_return_1d"], 0.10)
    assert np.isnan(result.loc[4, "fwd_return_1d"])

def test_forward_outcome_does_not_fill_beyond_available_data():
    df = pd.DataFrame({"close": np.arange(100., 110.)})
    result = add_forward_outcomes(df, horizons=(3,))
    assert result["fwd_return_3d"].iloc[-3:].isna().all()

def test_mae_mfe_are_forward_only():
    df = pd.DataFrame({"close": [100., 110., 90., 120.]})
    result = add_forward_outcomes(df, horizons=(2,))
    assert np.isclose(result.loc[0, "mae_2d"], -0.10)
    assert np.isclose(result.loc[0, "mfe_2d"], 0.20)
