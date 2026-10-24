from fractions import Fraction

from probabilities import InputError


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
