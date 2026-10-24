import random
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice

from probabilities import InputError
from .common import sample_distribution


@dataclass
class MarkovSource:
    alphabet: list[str]
    depth: int
    initial: dict[tuple[str, ...], Fraction]
    models: dict[str, dict[tuple[str, ...], dict[str, Fraction]]]
    switches: dict[str, dict[str, Fraction]]
    source: list[str]

    @property
    def period(self) -> int:
        return len(self.source)

    def distribution_at(
        self, context: tuple[str, ...], position: int
    ) -> dict[str, Fraction]:
        """P(X_position=a | последние depth символов = context)"""
        switch_name = self.source[position % self.period]
        switch = self.switches[switch_name]
        result: dict[str, Fraction] = {}

        for model_name, model_probability in switch.items():
            # если модель невозможна, то пропускаем
            if model_probability == 0:
                continue

            model = self.models[model_name]
            if context not in model:
                raise InputError(
                    f"Missing context {context} in model {model_name}"
                )
            for symbol, symbol_probability in model[context].items():
                result[symbol] = (
                    result.get(symbol, Fraction(0))
                    + model_probability * symbol_probability
                )

        return result

    def stream(self, rng: random.Random):
        context = sample_distribution(self.initial, rng)
        for symbol in context:
            yield symbol

        position = self.depth
        while True:
            distribution = self.distribution_at(context, position)
            symbol = sample_distribution(distribution, rng)
            yield symbol
            context = context[1:] + (symbol,)
            position += 1

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
        if any(symbol not in self.alphabet for symbol in block):
            return Fraction(0)

        end_position = start_position + len(block)
        states: dict[tuple[str, ...], Fraction] = {}

        # первые d символов
        for context, p in self.initial.items():
            if p == 0:
                continue
            valid = all(
                context[i] == block[i - start_position]
                for i in range(start_position, min(self.depth, end_position))
            )
            if valid:
                states[context] = p

        if end_position <= self.depth:
            return sum(states.values(), Fraction(0))

        # после начального d-блока
        for position in range(self.depth, end_position):
            next_states: dict[tuple[str, ...], Fraction] = {}
            required = (
                block[position - start_position]
                if position >= start_position else None
            )
            for context, state_probability in states.items():
                distribution = self.distribution_at(context, position)
                for symbol, transition_probability in distribution.items():
                    if transition_probability == 0:
                        continue
                    if required is not None and symbol != required:
                        continue
                    new_context = context[1:] + (symbol,)
                    next_states[new_context] = (
                        next_states.get(new_context, Fraction(0))
                        + state_probability * transition_probability
                    )
            states = next_states
            if not states:
                return Fraction(0)

        return sum(states.values(), Fraction(0))

    def average_block_probability(self, block: list[str]) -> Fraction:
        return sum(
            (self.block_probability(block, j) for j in range(self.period)),
            Fraction(0),
        ) / self.period
