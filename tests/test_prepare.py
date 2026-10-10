"""Ön işlemenin yalnızca eğitim verisinden öğrenildiğini ve model girdisinin temiz olduğunu sınar."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sdnids.data.preprocess import make_preprocessor, split_matrices
from sdnids.data.split import make_split
from sdnids.data.synthetic import make_synthetic_flows
from sdnids.schema import FORBIDDEN_INPUT_COLUMNS, SchemaMismatchError, build_schema_from_frame


class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_flows(seed=3)
        cls.schema, _ = build_schema_from_frame(cls.df)
        res = make_split(cls.df, cls.schema.names, seed=5)
        cls.mats = split_matrices(cls.df, res.assignment, cls.schema)

    def test_model_input_has_schema_columns_in_order_and_nothing_forbidden(self):
        for part in self.mats.values():
            self.assertEqual(list(part["X"].columns), self.schema.names)
            self.assertFalse(set(part["X"].columns) & FORBIDDEN_INPUT_COLUMNS)
            self.assertNotIn("label", part["X"].columns)

    def test_labels_and_record_ids_align_with_rows(self):
        for part in self.mats.values():
            n = len(part["X"])
            self.assertEqual(len(part["y_binary"]), n)
            self.assertEqual(len(part["y_multi"]), n)
            self.assertEqual(len(part["record_ids"]), n)
            self.assertTrue(set(np.unique(part["y_binary"])) <= {0, 1})

    def test_imputer_statistics_come_from_train_only(self):
        train = self.mats["train"]["X"].to_numpy(dtype=float)
        test = self.mats["test"]["X"].to_numpy(dtype=float) * 1000.0  # test'i bilerek bozuyoruz

        pre = make_preprocessor(scale=False).fit(self.mats["train"]["X"])
        train_clean = np.where(np.isinf(train), np.nan, train)
        np.testing.assert_allclose(pre.named_steps["impute"].statistics_, np.nanmedian(train_clean, axis=0))
        before = pre.named_steps["impute"].statistics_.copy()
        pre.transform(self.mats["test"]["X"])
        np.testing.assert_array_equal(before, pre.named_steps["impute"].statistics_)

        # negatif kontrol: test de öğrenmeye girseydi istatistikler değişirdi
        leaky = make_preprocessor(scale=False).fit(np.vstack([train, test]))
        self.assertFalse(np.allclose(before, leaky.named_steps["impute"].statistics_))

    def test_scaler_statistics_come_from_train_only(self):
        pre = make_preprocessor(scale=True).fit(self.mats["train"]["X"])
        scaler = pre.named_steps["scale"]
        clean = pre.named_steps["impute"].transform(pre.named_steps["inf_to_nan"].transform(self.mats["train"]["X"]))
        np.testing.assert_allclose(scaler.mean_, clean.mean(axis=0))

    def test_transform_output_is_finite(self):
        pre = make_preprocessor().fit(self.mats["train"]["X"])
        for part in self.mats.values():
            self.assertTrue(np.isfinite(pre.transform(part["X"])).all())

    def test_live_frame_missing_a_feature_fails_instead_of_zero_filling(self):
        live = self.mats["test"]["X"].drop(columns=[self.schema.names[0]])
        with self.assertRaises(SchemaMismatchError):
            self.schema.align(live)


if __name__ == "__main__":
    unittest.main()
