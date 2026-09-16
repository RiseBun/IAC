import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_as_common_source_pool.py"
SPEC = importlib.util.spec_from_file_location("build_as_common_source_pool", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CommonSourcePoolTest(unittest.TestCase):
    def test_selects_only_available_sources_without_using_scores(self):
        benchmark = [
            {"source_key": "a", "stratum": "turn", "benchmark_id": "1"},
            {"source_key": "b", "stratum": "turn", "benchmark_id": "2"},
            {"source_key": "c", "stratum": "stop", "benchmark_id": "3"},
        ]
        available = [
            {"source_key": key, "lineage": {"source_sample": f"/{key}.pkl"}, "score": 1.0}
            for key in ("a", "b", "c")
        ]
        selected = MODULE.select_sources(benchmark, available, {"turn": 1, "stop": 1}, seed=7)
        self.assertEqual({row["stratum"] for row in selected}, {"turn", "stop"})
        self.assertTrue(all(row["selection_uses_model_output"] is False for row in selected))
        self.assertTrue(all("score" not in row for row in selected))

    def test_fails_when_a_stratum_is_unavailable(self):
        with self.assertRaises(ValueError):
            MODULE.select_sources(
                [{"source_key": "a", "stratum": "stop"}],
                [],
                {"stop": 1},
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
