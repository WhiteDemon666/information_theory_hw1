# GPT generated
import json
import random
import unittest
from fractions import Fraction
from pathlib import Path

from probabilities import InputError, parse_probability, parse_distribution
from sources import empirical_frequency, MarkovSource, MixtureSource, load_source, parse_source_data

FIXTURES = Path(__file__).resolve().parent


class ProbabilityTests(unittest.TestCase):
    def test_parse_fraction(self):
        self.assertEqual(
            parse_probability("1/3"),
            Fraction(1, 3),
        )

    def test_parse_decimal(self):
        self.assertEqual(
            parse_probability("0.3"),
            Fraction(3, 10),
        )

    def test_distribution(self):
        distribution = parse_distribution({
            "0": "1/3",
            "1": "2/3",
        })

        self.assertEqual(
            distribution["0"],
            Fraction(1, 3),
        )

        self.assertEqual(
            distribution["1"],
            Fraction(2, 3),
        )

    def test_not_normalized_distribution(self):
        with self.assertRaises(InputError):
            parse_distribution({
                "0": "1/3",
                "1": "1/3",
            })


class PeriodicSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = load_source(
            str(Path(__file__).resolve().parent / "legacy_example.json")
        )

    def test_period(self):
        self.assertEqual(
            self.source.period,
            4,
        )

    def test_first_block_probability(self):
        probability = self.source.block_probability(
            ["0", "1"]
        )

        self.assertEqual(
            probability,
            Fraction(13, 60),
        )

    def test_average_block_probability(self):
        probability = (
            self.source.average_block_probability(
                ["0", "1"]
            )
        )

        self.assertEqual(
            probability,
            Fraction(56, 225),
        )

    def test_seed_reproducibility(self):
        first = self.source.generate(
            100,
            random.Random(42),
        )

        second = self.source.generate(
            100,
            random.Random(42),
        )

        self.assertEqual(first, second)

    def test_empirical_frequency(self):
        sequence = ["0", "1", "0", "1"]

        frequency = empirical_frequency(
            sequence,
            ["0", "1"],
        )

        self.assertEqual(
            frequency,
            Fraction(2, 3),
        )


class MarkovTests(unittest.TestCase):
    def test_read_and_type(self):
        source = load_source(str(FIXTURES / "markov_example.json"))
        self.assertIsInstance(source, MarkovSource)
        self.assertEqual(source.depth, 3)
        self.assertEqual(source.period, 4)
        self.assertEqual(len(source.models), 4)

    def test_block_and_phase_average(self):
        source = load_source(str(FIXTURES / "markov_example.json"))
        block = ["0", "1", "2", "0"]
        self.assertEqual(source.block_probability(block), Fraction(1, 3))
        self.assertEqual(source.block_probability(block, 1), Fraction(1, 3))
        self.assertEqual(source.average_block_probability(block), Fraction(1, 3))
        self.assertEqual(source.block_probability(["0", "0"]), Fraction(0))

    def test_stationary_initial_phase(self):
        source = load_source(str(FIXTURES / "markov_example.json"))
        for start in range(8):
            self.assertEqual(
                source.block_probability(["0", "1", "2"], start),
                Fraction(1, 3),
            )

    def test_dynamic_programming_unknown_previous_context(self):
        source = load_source(str(FIXTURES / "markov_dp_example.json"))
        # P(X_3=0, X_4=1) = 1/2*3/4 + 1/2*1/2 = 5/8.
        self.assertEqual(
            source.block_probability(["0", "1"], start_position=2),
            Fraction(5, 8),
        )
        self.assertEqual(source.block_probability(["0"], 2), Fraction(1))
        self.assertEqual(source.block_probability(["0", "0", "0"]), Fraction(1, 2))

    def test_generation_reproducible(self):
        source = load_source(str(FIXTURES / "markov_example.json"))
        a = source.generate(30, random.Random(7))
        b = source.generate(30, random.Random(7))
        self.assertEqual(a, b)
        self.assertTrue(all(a[j] == a[j + 3] for j in range(27)))

    def test_invalid_reachable_context(self):
        raw = json.loads((FIXTURES / "markov_example.json").read_text(encoding="utf-8"))
        del raw["models"]["model_4"][0]
        # Model_4 needs the removed context with positive probability.
        with self.assertRaises(InputError):
            parse_source_data(raw)

    def test_zero_weight_model_needs_no_context(self):
        raw = json.loads((FIXTURES / "markov_example.json").read_text(encoding="utf-8"))
        raw["models"]["model_4"] = raw["models"]["model_4"][:1]
        raw["switches"]["switch_4"] = {"model_1": "1", "model_4": "0"}
        source = parse_source_data(raw)
        self.assertEqual(source.block_probability(["0", "1", "2"]), Fraction(1, 3))

    def test_invalid_initial_probability(self):
        raw = json.loads((FIXTURES / "markov_example.json").read_text(encoding="utf-8"))
        raw["initial"][0]["probability"] = "1/2"
        with self.assertRaises(InputError):
            parse_source_data(raw)

    def test_invalid_missing_switch(self):
        raw = json.loads((FIXTURES / "markov_example.json").read_text(encoding="utf-8"))
        raw["source"][0] = "absent"
        with self.assertRaises(InputError):
            parse_source_data(raw)


class MixtureTests(unittest.TestCase):
    def test_read_and_probability(self):
        source = load_source(str(FIXTURES / "nonergodic_example.json"))
        self.assertIsInstance(source, MixtureSource)
        self.assertEqual(source.period, 4)
        for pos in (0, 1, 2, 5):
            self.assertEqual(source.block_probability(["0", "0"], pos), Fraction(1, 2))
            self.assertEqual(source.block_probability(["0", "1"], pos), Fraction(0))
        self.assertEqual(source.average_block_probability(["0", "0"]), Fraction(1, 2))

    def test_chosen_once_not_each_tick(self):
        source = load_source(str(FIXTURES / "nonergodic_example.json"))
        for seed in range(20):
            seq = source.generate(100, random.Random(seed))
            self.assertTrue(all(symbol == seq[0] for symbol in seq))
            self.assertIn(seq[0], ("0", "1"))

    def test_mixture_reproducibility(self):
        source = load_source(str(FIXTURES / "nonergodic_example.json"))
        self.assertEqual(source.generate(30, random.Random(12)),
                         source.generate(30, random.Random(12)))

    def test_invalid_weights(self):
        raw = json.loads((FIXTURES / "nonergodic_example.json").read_text(encoding="utf-8"))
        raw["components"]["zero"]["weight"] = "1/4"
        with self.assertRaises(InputError):
            parse_source_data(raw)

    def test_different_periods(self):
        raw = json.loads((FIXTURES / "nonergodic_example.json").read_text(encoding="utf-8"))
        raw["components"]["zero"]["source"]["source"].pop()
        with self.assertRaises(InputError):
            parse_source_data(raw)

    def test_foreign_source_type(self):
        with self.assertRaises(InputError):
            parse_source_data({"type": "whatever"})


if __name__ == "__main__":
    unittest.main()
