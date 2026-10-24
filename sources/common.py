import math
import random


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
