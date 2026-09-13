"""Small deterministic tests of invariance, adverse scenarios and label access."""
import unittest
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sc_arc_core import (PreparedTask, HistoricalModels, knn_weights,
                         local_instability, fit_measured, score_queries, decide)

class CoreTests(unittest.TestCase):
    def test_common_offset_invariance(self):
        rng = np.random.default_rng(6103)
        for _ in range(1000):
            z = rng.normal(size=(8,4))
            dl = cdist(z[:5],z[:5]); np.fill_diagonal(dl,np.inf)
            wl, wq = knn_weights(dl),knn_weights(cdist(z[5:],z[:5]))
            r = rng.normal(size=5)
            shift = rng.uniform(-5,5)
            np.testing.assert_allclose(local_instability(r,wl,wq),
                local_instability(r+shift,wl,wq),atol=5e-15,rtol=0)

    def task(self):
        z = np.arange(18,dtype=float).reshape(6,3)/20
        return PreparedTask('test',.25,np.array([f'c{i}' for i in range(6)]),
                            z,z,z,np.full(6,.84),np.full((6,9),.84),np.arange(5))

    def test_stable_but_adverse_neighborhood(self):
        task = self.task()
        fit = dict(selected=np.arange(5),prediction=np.full(6,.84),residuals=np.full(5,-.05))
        scores = score_queries(task,fit,thresholds=(.8,))
        self.assertAlmostEqual(float(scores.local_scale.iloc[0]),.02)
        self.assertEqual(float(scores.action_score.iloc[0]),0.)
        self.assertFalse(bool(decide(scores,1.5123).issued.iloc[0]))

    def test_prediction_at_threshold_and_strict_cutoff(self):
        fit = dict(selected=np.arange(5),prediction=np.full(6,.8),residuals=np.zeros(5))
        scores = score_queries(self.task(),fit,thresholds=(.8,))
        self.assertFalse(bool(decide(scores,0).issued.iloc[0]))
        scores.loc[0,['eligible','action_score']] = [True,2.]
        self.assertFalse(bool(decide(scores,2.).issued.iloc[0]))
        self.assertTrue(bool(decide(scores,1.9).issued.iloc[0]))
        self.assertFalse(bool(decide(scores,np.inf).issued.iloc[0]))

    def test_measurement_api_rejects_extra_or_missing_labels(self):
        task = self.task()
        for labels in ({'c0':.8},{f'c{i}':.8 for i in range(6)}):
            with self.assertRaises(ValueError):
                fit_measured(task,labels)

    def test_preparation_rejects_query_outcomes(self):
        uninitialized = object.__new__(HistoricalModels)
        for name in ('soh','truth'):
            with self.assertRaises(ValueError):
                uninitialized.prepare(pd.DataFrame({name:[.8]}),.25)

if __name__ == '__main__':
    unittest.main(verbosity=2)
