# GPT generated
import random
import unittest
from fractions import Fraction
from pathlib import Path

from probabilities import InputError, parse_probability, parse_distribution
from sources import load_source, empirical_frequency


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


if __name__ == "__main__":
    unittest.main()
