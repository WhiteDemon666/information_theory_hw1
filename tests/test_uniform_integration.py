# GPT generated

import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from coding.asymptotic import eventual_error_answer
from coding.checks import check_memoryless_stationary
from coding.uniform import optimal_uniform_code
from sources import MixtureSource, load_source
from sources.loader import parse_source_data


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests"


class SourceValidationTests(unittest.TestCase):
    def test_iid_legacy(self):
        result = check_memoryless_stationary(load_source(str(DATA / "iid_example.json")))
        self.assertTrue(result.stationary)
        self.assertTrue(result.memoryless)
        self.assertEqual(result.distribution["0"], Fraction(1, 2))

    def test_nonstationary_legacy(self):
        result = check_memoryless_stationary(load_source(str(DATA / "legacy_example.json")))
        self.assertFalse(result.stationary)

    def test_stationary_markov_with_memory(self):
        result = check_memoryless_stationary(load_source(str(DATA / "markov_example.json")))
        self.assertTrue(result.stationary)
        self.assertFalse(result.memoryless)

    def test_stationary_nonergodic_mixture_with_memory(self):
        result = check_memoryless_stationary(load_source(str(DATA / "nonergodic_example.json")))
        self.assertTrue(result.stationary)
        self.assertFalse(result.memoryless)

    def test_iid_markov(self):
        source = parse_source_data({
            "type": "markov", "alphabet": ["0", "1"], "depth": 1,
            "initial": [
                {"context": ["0"], "probability": "1/2"},
                {"context": ["1"], "probability": "1/2"},
            ],
            "models": {"m": [
                {"context": ["0"], "distribution": {"0": "1/2", "1": "1/2"}},
                {"context": ["1"], "distribution": {"0": "1/2", "1": "1/2"}},
            ]},
            "switches": {"s": {"m": "1"}}, "source": ["s"],
        })
        result = check_memoryless_stationary(source)
        self.assertTrue(result.stationary)
        self.assertTrue(result.memoryless)

    def test_stationary_mixture_of_nonstationary_components(self):
        # Одинаково вероятные 0101... и 1010... дают стационарный процесс с памятью.
        src1 = parse_source_data({
            "models": {"a": {"0": "1"}, "b": {"1": "1"}},
            "switches": {"a": {"a": "1"}, "b": {"b": "1"}},
            "source": ["a", "b"],
        })
        src2 = parse_source_data({
            "models": {"a": {"0": "1"}, "b": {"1": "1"}},
            "switches": {"a": {"a": "1"}, "b": {"b": "1"}},
            "source": ["b", "a"],
        })
        mix = MixtureSource({"a": src1, "b": src2}, {"a": Fraction(1, 2), "b": Fraction(1, 2)}, 2)
        result = check_memoryless_stationary(mix)
        self.assertTrue(result.stationary)
        self.assertFalse(result.memoryless)

    def test_mixture_of_equal_iid_sources(self):
        source = load_source(str(DATA / "iid_example.json"))
        mix = MixtureSource({"a": source, "b": source}, {"a": Fraction(1, 2), "b": Fraction(1, 2)}, 4)
        result = check_memoryless_stationary(mix)
        self.assertTrue(result.stationary)
        self.assertTrue(result.memoryless)


class AsymptoticTests(unittest.TestCase):
    def test_strict_cases(self):
        p = {"0": Fraction(1, 2), "1": Fraction(1, 2)}
        self.assertTrue(eventual_error_answer(p, "1.1", Fraction(1, 10000), 2).exists)
        self.assertFalse(eventual_error_answer(p, "0.9", Fraction(1, 10000), 2).exists)

    def test_uniform_boundary_zero_error(self):
        p = {"0": Fraction(1, 2), "1": Fraction(1, 2)}
        self.assertTrue(eventual_error_answer(p, "1", Fraction(1, 10000), 2).exists)
        self.assertFalse(eventual_error_answer(p, "1", Fraction(1, 10000), 4).exists)
        self.assertTrue(eventual_error_answer(p, "1", Fraction(1, 2), 4).exists)

    def test_deterministic_boundary(self):
        p = {"0": Fraction(1), "1": Fraction(0)}
        self.assertTrue(eventual_error_answer(p, "0", Fraction(1, 10000), 7).exists)

    def test_full_codebook_description(self):
        p = {"0": Fraction(1, 2), "1": Fraction(1, 2)}
        result = optimal_uniform_code(p, 10, "100", 7, with_description=True)
        self.assertEqual(result.error, 0)
        self.assertTrue(result.compact_code_description()["all_blocks_encoded"])
        self.assertEqual(result.compact_code_description()["selected_types"], [])


class CLITests(unittest.TestCase):
    def call(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *map(str, args)],
            capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
        )

    def test_check(self):
        r = self.call("uniform-check", DATA / "iid_example.json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1.459147", r.stdout)

    def test_bad_source(self):
        r = self.call("uniform-code", DATA / "nonergodic_example.json", "--rate", "1", "-q", "7")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("память", r.stderr)

    def test_code_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "code.json"
            r = self.call("uniform-code", DATA / "iid_example.json", "--rate", "1",
                          "--epsilon", "1/10000", "-q", "7", "--n-max", "5", "--output", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue("Минимальное подходящее" in r.stdout or "подходящего размера блока нет" in r.stdout)
            description = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(description["code_alphabet_size"], 7)
            self.assertIn("selection_rule", description)
            self.assertIn("типы", description["selection_rule"])

    def test_variant(self):
        r = self.call("--variant")
        self.assertEqual(r.returncode, 0)
        self.assertIn("s=11", r.stdout)

    def test_old_generate(self):
        r = self.call("generate", DATA / "iid_example.json", "-n", "10", "--seed", "3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(r.stdout.split()), 10)

    def test_invalid_epsilon(self):
        r = self.call("uniform-code", DATA / "iid_example.json", "--rate", "1", "--epsilon", "0", "-q", "7")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("epsilon", r.stderr)


if __name__ == "__main__":
    unittest.main()
