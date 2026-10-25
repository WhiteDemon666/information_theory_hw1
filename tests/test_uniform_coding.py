# GPT generated
import math
import unittest
from fractions import Fraction
from itertools import product

from coding.uniform import (
    codeword_length,
    optimal_uniform_code,
    source_entropy,
    type_multiplicity,
    types_of_length,
)


class UniformCodingTests(unittest.TestCase):
    def test_small_case(self):
        p = {"0": Fraction(1, 2), "1": Fraction(1, 3), "2": Fraction(1, 6)}
        r = optimal_uniform_code(p, n=2, rate="1", q=2, with_description=True)
        self.assertEqual(r.ell, 2)
        self.assertEqual(r.capacity, 4)
        self.assertEqual(r.error, Fraction(11, 36))
        self.assertEqual(sum(g[1] for g in r.selected_types), 4)

    def test_uniform(self):
        p = {"0": Fraction(1, 2), "1": Fraction(1, 2)}
        r = optimal_uniform_code(p, n=4, rate="0.5", q=2)
        self.assertEqual(r.ell, 2)
        self.assertEqual(r.error, Fraction(3, 4))
        self.assertEqual(source_entropy(p), 1.0)

    def test_full_codebook(self):
        p = {"0": Fraction(1, 2), "1": Fraction(1, 3), "2": Fraction(1, 6)}
        self.assertEqual(optimal_uniform_code(p, 10, rate="100", q=7).error, Fraction(0))

    def test_types(self):
        for n in range(1, 8):
            for k in range(1, 5):
                counts = list(types_of_length(n, k))
                self.assertEqual(len(counts), math.comb(n + k - 1, k - 1))
                self.assertEqual(sum(type_multiplicity(t) for t in counts), k**n)

    def test_against_brute_force(self):
        p = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)]
        model = {str(i): v for i, v in enumerate(p)}
        for n in range(1, 6):
            ell = codeword_length(n, "1", 2)
            capacity = 2**ell
            probs = sorted((math.prod(p[i] for i in block) for block in product(range(3), repeat=n)), reverse=True)
            expected = 1 - sum(probs[:capacity], Fraction(0))
            self.assertEqual(optimal_uniform_code(model, n, rate="1", q=2).error, expected)

    def test_deterministic(self):
        p = {"0": Fraction(1), "1": Fraction(0)}
        self.assertEqual(optimal_uniform_code(p, 50, rate="0", q=7).error, Fraction(0))


if __name__ == "__main__":
    unittest.main()