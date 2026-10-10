import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sdnids.data.dedupe import deduplicate_vectors
from sdnids.data.synthetic import make_synthetic_flows
from sdnids.schema import build_schema_from_frame


class DedupeTests(unittest.TestCase):
    def test_handcrafted_case(self):
        df = pd.DataFrame(
            {
                "record_id": ["r0", "r1", "r2", "r3", "r4"],
                "f1": [1.0, 1.0, 2.0, 3.0, 3.0],
                "f2": [0.0, 0.0, 5.0, 7.0, 7.0],
                # A: iki kopya (Normal), B: tek, C: aynı vektör iki farklı etiketle
                "label": ["Normal", "Normal", "DoS", "Normal", "DoS"],
            }
        )
        unique, excluded, rep = deduplicate_vectors(df, ["f1", "f2"])
        self.assertEqual(list(unique["record_id"]), ["r0", "r2"])
        self.assertEqual(list(unique["n_copies"]), [2, 1])
        self.assertEqual(sorted(excluded["record_id"]), ["r3", "r4"])
        self.assertEqual(rep["rows_removed_as_copies"], 1)
        self.assertEqual(rep["rows_excluded_conflicting_label"], 2)
        self.assertEqual(rep["conflicting_vectors"], 1)

    def test_on_synthetic_data_invariants_hold(self):
        df = make_synthetic_flows(seed=4)
        feats = build_schema_from_frame(df)[0].names
        unique, excluded, rep = deduplicate_vectors(df, feats)

        self.assertFalse(unique.duplicated(subset=feats).any(), "benzersiz tabloda tekrar kalmış")
        self.assertEqual(int(unique["n_copies"].sum()), len(df) - len(excluded))
        self.assertEqual(rep["rows_in"], len(df))
        self.assertEqual(
            rep["rows_unique_kept"] + rep["rows_removed_as_copies"] + rep["rows_excluded_conflicting_label"],
            len(df),
        )
        # Dışlanan her vektör gerçekten birden fazla etikete sahipti
        orig = df.assign(h=pd.util.hash_pandas_object(df[feats], index=False).to_numpy())
        nlab = orig.groupby("h")["label"].nunique()
        excl_h = orig[orig["record_id"].isin(excluded["record_id"])]["h"].unique()
        self.assertTrue((nlab.loc[excl_h] > 1).all())
        # Dışlanmayan kayıtların vektörleri çelişkisizdir
        kept_h = orig[~orig["record_id"].isin(excluded["record_id"])]["h"].unique()
        self.assertTrue((nlab.loc[kept_h] == 1).all())
        self.assertGreater(len(excluded), 0)

    def test_n_copies_is_never_a_model_feature(self):
        df = make_synthetic_flows(seed=4)
        feats = build_schema_from_frame(df)[0].names
        unique, _, _ = deduplicate_vectors(df, feats)
        schema, _ = build_schema_from_frame(unique)
        self.assertNotIn("n_copies", schema.names)
        self.assertEqual(schema.names, feats)

    def test_idempotent(self):
        df = make_synthetic_flows(seed=4)
        feats = build_schema_from_frame(df)[0].names
        unique, _, _ = deduplicate_vectors(df, feats)
        again, excluded, rep = deduplicate_vectors(unique.drop(columns=["n_copies"]), feats)
        self.assertEqual(len(again), len(unique))
        self.assertEqual(len(excluded), 0)
        self.assertEqual(rep["rows_removed_as_copies"], 0)


if __name__ == "__main__":
    unittest.main()
