import math
import random
from fractions import Fraction
from dataclasses import dataclass
import json

from probabilities import InputError, parse_distribution
from collections import deque

from probabilities import (
    InputError,
    parse_distribution,
    parse_probability,
)


def sample_distribution(
    distribution: dict,
    rng: random.Random):

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


@dataclass
class MarkovSource:
    alphabet: list[str]
    depth: int

    initial: dict[
        tuple[str, ...],
        Fraction
    ]

    models: dict[
        str,
        dict[
            tuple[str, ...],
            dict[str, Fraction]
        ]
    ]

    switches: dict[
        str,
        dict[str, Fraction]
    ]

    source: list[str]

    @property
    def period(self) -> int:
        return len(self.source)

    def distribution_at(
            self,
            context: tuple[str, ...],
            position: int) -> dict[str, Fraction]:

        switch_name = self.source[
            position % self.period
            ]

        switch = self.switches[switch_name]

        result = {}
        for model_name, model_probability in switch.items():

            model = self.models[model_name]
            if context not in model:
                raise InputError(
                    f"Missing context {context} "
                    f"in model {model_name}"
                )

            distribution = model[context]
            for symbol, symbol_probability in distribution.items():
                result[symbol] = (
                        result.get(
                            symbol,
                            Fraction(0, 1),
                        )
                        + model_probability
                        * symbol_probability
                )

        return result

    def stream(self, rng: random.Random):
        context = sample_distribution(
            self.initial,
            rng,
        )

        result = list(context)
        for symbol in result:
            yield symbol

        position = self.depth
        while True:
            context = tuple(
                result[-self.depth:]
            )

            distribution = self.distribution_at(
                context,
                position,
            )

            symbol = sample_distribution(
                distribution,
                rng,
            )

            yield symbol
            result.append(symbol)

            if len(result) > self.depth:
                result = result[-self.depth:]

            position += 1

    def generate(
            self,
            n: int,
            rng: random.Random,
    ) -> list[str]:

        generator = self.stream(rng)

        return [
            next(generator)
            for _ in range(n)
        ]

    def block_probability(
            self,
            block: list[str],
            start_position: int = 0) -> Fraction:

        if not block:
            raise InputError(
                "Block must not be empty"
            )

        states = dict(self.initial)
        end_position = (
                start_position + len(block)
        )

        # учитываем символы блока, которые могут лежать внутри стартвого контекста
        filtered_states = {}
        for context, probability in states.items():
            valid = True
            for position in range(min(self.depth, end_position)):
                if position < start_position:
                    continue

                expected = block[position - start_position]
                if context[position] != expected:
                    valid = False
                    break
            if valid:
                filtered_states[context] = probability
        states = filtered_states

        # после d первых символов
        for position in range(
                self.depth,
                end_position):

            new_states = {}
            for context, state_probability in states.items():
                distribution = self.distribution_at(
                    context,
                    position)
                for symbol, probability in distribution.items():
                    if position >= start_position:
                        expected = block[position - start_position]
                        if symbol != expected:
                            continue

                    new_context = (
                            context[1:] + (symbol,)
                    )

                    new_states[new_context] = (
                            new_states.get(
                                new_context,
                                Fraction(0, 1),
                            )
                            + state_probability
                            * probability
                    )

            states = new_states

        return sum(
            states.values(),
            Fraction(0, 1),
        )

    def average_block_probability(
            self,
            block: list[str]) -> Fraction:

        total = Fraction(0, 1)
        for start_position in range(
                self.period
        ):
            total += self.block_probability(block, start_position)
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

