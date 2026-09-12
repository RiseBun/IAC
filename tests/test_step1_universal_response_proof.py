import unittest

from tools.prove_step1_universal_response import prove


class Step1UniversalResponseProofTest(unittest.TestCase):
    def test_synthetic_controls_have_expected_separation(self) -> None:
        report = prove()
        self.assertTrue(report["synthetic_only"])
        decision = report["decision"]
        self.assertTrue(decision["normal_direction_passes"])
        self.assertTrue(decision["reversed_direction_fails"])
        self.assertTrue(decision["identity_is_unavailable"])
        self.assertTrue(decision["zero_contrast_is_unavailable"])
        self.assertTrue(decision["common_mode_signed_difference_is_stable"])
        self.assertFalse(decision["benchmark_promotion"])


if __name__ == "__main__":
    unittest.main()
