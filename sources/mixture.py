import random
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from typing import Any

from probabilities import InputError
from .common import sample_distribution


@dataclass
class MixtureSource:
    components: dict[str, Any]
    weights: dict[str, Fraction]
    period: int

    def stream(self, rng: random.Random):
        name = sample_distribution(self.weights, rng)
        # yield from сохраняет выбор на все время генерации
        yield from self.components[name].stream(rng)

    def generate(self, n: int, rng: random.Random) -> list[str]:
        if n < 0:
            raise InputError("N must be non-negative")
        return list(islice(self.stream(rng), n))

    def block_probability(
        self, block: list[str], start_position: int = 0
    ) -> Fraction:
        if not block:
            raise InputError("Block must not be empty")
        if start_position < 0:
            raise InputError("Start position must be non-negative")
        return sum(
            (
                weight * self.components[name].block_probability(
                    block, start_position
                )
                for name, weight in self.weights.items()
            ),
            Fraction(0),
        )

    def average_block_probability(self, block: list[str]) -> Fraction:
        return sum(
            (
                weight * self.components[name].average_block_probability(block)
                for name, weight in self.weights.items()
            ),
            Fraction(0),
        )
