# GPT generated

import math
import random
from fractions import Fraction
from dataclasses import dataclass
import json

from probabilities import InputError, parse_distribution


def sample_distribution(
    distribution: dict[str, Fraction],
    rng: random.Random) -> str:

    denominator = 1
    for probability in distribution.values():
        denominator = math.lcm(
            denominator,
            probability.denominator,
        )

    weights = {}
    for symbol, probability in distribution.items():
        weights[symbol] = (
                probability.numerator
                * (denominator // probability.denominator)
        )

    random_number = rng.randrange(denominator)

    cur = 0
    for symbol, weight in weights.items():
        cur += weight

        if random_number < cur:
            return symbol

    raise RuntimeError("Epsilont happened")


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


def load_source(filename: str) -> PeriodicSource:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise InputError(f"File not found: {filename}")
    except json.JSONDecodeError:
        raise InputError("Invalid JSON")

    if "models" not in data:
        raise InputError("Missing models")

    if "switches" not in data:
        raise InputError("Missing switches")

    if "source" not in data:
        raise InputError("Missing source")

    models = {}

    for model_name, distribution in data["models"].items():
        models[model_name] = parse_distribution(distribution)

    switches = {}
    for switch_name, distribution in data["switches"].items():
        parsed_switch = parse_distribution(distribution)

        for model_name in parsed_switch:
            if model_name not in models:
                raise InputError(
                    f"Unknown model {model_name}"
                )

        switches[switch_name] = parsed_switch

    source = data["source"]
    if not source:
        raise InputError("Source must not be empty")

    for switch_name in source:
        if switch_name not in switches:
            raise InputError(
                f"Unknown switch {switch_name}"
            )

    phase_distributions = []
    for switch_name in source:
        distribution = get_symbol_distribution(
            switches[switch_name],
            models,
        )

        phase_distributions.append(distribution)

    return PeriodicSource(
        models=models,
        switches=switches,
        source=source,
        phase_distributions=phase_distributions,
    )


def empirical_frequency(
    sequence: list[str],
    block: list[str],) -> Fraction:

    if not block:
        raise InputError("Block must not be empty")

    if len(sequence) < len(block):
        raise InputError("Sequence is shorter than block")

    matches = 0
    block_length = len(block)
    for i in range(len(sequence) - block_length + 1):
        if sequence[i:i + block_length] == block:
            matches += 1

    return Fraction(
        matches,
        len(sequence) - block_length + 1,
    )

