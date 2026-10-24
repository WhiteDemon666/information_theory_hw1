import random
from dataclasses import dataclass
from fractions import Fraction

from .common import sample_distribution


def get_symbol_distribution(
    switch: dict[str, Fraction],
    models: dict[str, dict[str, Fraction]],) -> dict[str, Fraction]:

    result = {}
    for model_name, model_probability in switch.items():
        for symbol, symbol_probability in models[model_name].items():
            result[symbol] = (result.get(symbol, Fraction(0, 1))
                              + model_probability * symbol_probability)
    return result


@dataclass
class PeriodicSource:
    models: dict[str, dict[str, Fraction]]
    switches: dict[str, dict[str, Fraction]]
    source: list[str]
    phase_distributions: list[dict[str, Fraction]]

    @property
    def period(self) -> int:
        return len(self.source)

    def distribution_at(self, position: int) -> dict[str, Fraction]:
        return self.phase_distributions[position % self.period]

    def generate(
        self,
        n: int,
        rng: random.Random,
    ) -> list[str]:

        result = []

        for i in range(n):
            distribution = self.distribution_at(i)
            symbol = sample_distribution(distribution, rng)
            result.append(symbol)

        return result

    def stream(self, rng: random.Random):
        i = 0

        while True:
            distribution = self.distribution_at(i)
            yield sample_distribution(distribution, rng)
            i += 1

    def block_probability(
        self,
        block: list[str],
        start_position: int = 0,
    ) -> Fraction:

        probability = Fraction(1, 1)

        for i, symbol in enumerate(block):
            distribution = self.distribution_at(start_position + i)

            if symbol not in distribution:
                return Fraction(0, 1)

            probability *= distribution[symbol]

        return probability

    def average_block_probability(
        self,
        block: list[str],
    ) -> Fraction:

        total = Fraction(0, 1)

        for start_position in range(self.period):
            total += self.block_probability(
                block,
                start_position,
            )

        return total / self.period
