import random
from dataclasses import dataclass
from fractions import Fraction

from probabilities import InputError
from .common import sample_distribution


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
