"""
Tests for Model B's learned impulse response (spec D9).

Run from studies/:  python 05_greybox_monitoring/tests/test_model_b.py
"""
import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction  # noqa: E402

logging.getLogger('NP').setLevel(logging.ERROR)


class TestLaggedWeights(unittest.TestCase):

    def test_the_peak_weight_sits_at_the_injected_lag(self):
        rng = np.random.default_rng(0)
        n = 24 * 60
        index = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
        x = rng.normal(0, 1, n)
        y = 5.0 + 3.0 * np.roll(x, 3) + rng.normal(0, 0.1, n)
        frame = pd.DataFrame({'y': y, 'x': x}, index=index)
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=(), task='nowcast', epochs=20, freq='1h',
            yearly=False, daily_order=1, quantiles=(), lagged_regressors=('x',),
            lagged_n_lags=8, learning_rate=0.05)
        weights = prediction.lagged_regressor_weights(model)
        peak = int(weights.loc[weights['weight'].abs().idxmax(), 'lag'])
        self.assertEqual(peak, 3)

    def test_summary_reads_delay_and_time_constant_from_a_known_response(self):
        lags = np.arange(1, 25)
        w = 2.0 * np.exp(-(lags - 1) / 4.0) * (lags >= 1)
        weights = pd.DataFrame({'regressor': 'x', 'lag': lags, 'weight': w})
        summary = prediction.impulse_response_summary(weights, dt_hours=1.0)
        self.assertAlmostEqual(summary.loc[0, 'tau_h'], 4.0, delta=1.0)
        self.assertGreater(summary.loc[0, 'gain'], 0)

    def test_physical_weights_rescale_by_the_models_own_normalisation(self):
        # NeuralProphet's get_covar_weights() acts on whatever normalised
        # representation the model actually trained on -- standardized,
        # 'soft' (min/quantile-95 width), or otherwise, per column -- never
        # on the regressor's raw units. A weight of -1.94 next to Study 03's
        # -2.79 mdeg/degC is therefore not a discrepancy to explain, it is
        # two different units being compared. physical=True must undo
        # exactly the affine map the fitted model recorded for itself: read
        # back from model.config_normalization.global_data_params (a dict of
        # column name to a ShiftScale carrying .shift and .scale), not
        # recomputed from the raw series, since a 'soft' column's scale is
        # not its standard deviation.
        rng = np.random.default_rng(1)
        n = 24 * 60
        index = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
        x = rng.normal(10.0, 4.0, n)
        y = 5.0 + 0.5 * np.roll(x, 2) + rng.normal(0, 0.1, n)
        frame = pd.DataFrame({'y': y, 'x': x}, index=index)
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=(), task='nowcast', epochs=10, freq='1h',
            yearly=False, daily_order=1, quantiles=(), lagged_regressors=('x',),
            lagged_n_lags=6, learning_rate=0.05)

        raw = prediction.lagged_regressor_weights(model)
        physical = prediction.lagged_regressor_weights(model, physical=True)

        scale_y = model.config_normalization.global_data_params['y'].scale
        scale_x = model.config_normalization.global_data_params['x'].scale
        expected = raw['weight'].to_numpy() * (scale_y / scale_x)
        np.testing.assert_allclose(physical['weight'].to_numpy(), expected,
                                   rtol=1e-6)
        # The default is unchanged: physical=False still returns exactly
        # what NeuralProphet's own tensor carries, no rescaling applied.
        self.assertFalse(np.allclose(raw['weight'].to_numpy(), expected))


if __name__ == '__main__':
    unittest.main(verbosity=2)
