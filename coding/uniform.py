from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, localcontext
from fractions import Fraction
import math

from probabilities import InputError


def source_entropy(distribution: dict[str, Fraction]) -> float:
    return -sum(
        float(p) * math.log2(float(p))
        for p in distribution.values()
        if p > 0
    )


def codeword_length(n: int, rate: str, q: int) -> int:
    # ell = floor(n * R / log_2(q))
    if type(n) is not int or n < 1:
        raise InputError("n must be a positive integer")
    if type(q) is not int or q < 2:
        raise InputError("q must be an integer >= 2")

    try:
        R = Decimal(rate)
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise InputError("R must be a non-negative decimal string") from exc

    if not R.is_finite() or R < 0:
        raise InputError("R must be a non-negative finite number")

    with localcontext() as ctx:
        ctx.prec = 75
        if q & (q - 1) == 0:
            # q = 2^m
            raw = Decimal(n) * R / Decimal(q.bit_length() - 1)
        else:
            raw = Decimal(n) * R * Decimal(2).ln() / Decimal(q).ln()

        return int(raw.to_integral_value(rounding=ROUND_FLOOR))


def types_of_length(n: int, k: int):
    if k == 1:
        yield (n,)
    else:
        for first in range(n + 1):
            for tail in types_of_length(n - first, k - 1):
                yield (first,) + tail


def type_multiplicity(counts: tuple[int, ...]) -> int:
    remaining = sum(counts)
    multiplicity = 1
    for count in counts[:-1]:
        multiplicity *= math.comb(remaining, count)
        remaining -= count

    return multiplicity


@dataclass
class UniformResult:
    n: int
    q: int
    ell: int
    capacity: int
    total_words: int
    error: Fraction

    selected_types: list[tuple[tuple[int, ...], int, int, int]]
    denominator: int
    alphabet: list[str]

    def compact_code_description(self) -> dict:
        return {
            "alphabet": self.alphabet,
            "source_block_length": self.n,
            "code_alphabet_size": self.q,
            "codeword_length": self.ell,
            "all_blocks_encoded": self.capacity >= self.total_words,
            "selection_rule": (
                "Если all_blocks_encoded=true, все блоки перечисляются "
                "в лексикографическом порядке символов алфавита. "
                "Иначе типы рассматриваются в указанном порядке. "
                "Внутри каждого типа выбираются первые take различных блоков "
                "в лексикографическом порядке символов алфавита. "
                "Выбранным блокам последовательно присваиваются индексы 0, 1, 2, ... . "
                "Каждый индекс записывается в системе счисления с основанием q "
                "и дополняется ведущими нулями до длины codeword_length."
            ),
            "selected_types": [
                {"counts": list(counts), "take": take, "total": size}
                for counts, take, size, _ in self.selected_types
            ],
            "exact_error": str(self.error),
        }


def optimal_uniform_code(
    distribution: dict[str, Fraction],
    n: int,
    rate: str,
    q: int,
    *,
    with_description: bool = False,) -> UniformResult:
    if not distribution or sum(distribution.values(), Fraction(0)) != 1:
        raise InputError("Source distribution must be normalized")
    if any(p < 0 for p in distribution.values()):
        raise InputError("Negative source probability")

    ell = codeword_length(n, rate, q)
    alphabet = list(distribution)
    k = len(alphabet)
    total_words = k ** n

    # K = min(q^ell, k^n)
    K = 1
    for _ in range(ell):
        K *= q
        if K >= total_words:
            K = total_words
            break

    probs = list(distribution.values())
    D = math.lcm(*(p.denominator for p in probs))
    integer_weights = [p.numerator * (D // p.denominator) for p in probs]
    Dn = D ** n

    if K >= total_words:
        return UniformResult(n, q, ell, K, total_words, Fraction(0), [], D, alphabet)

    powers = [
        [weight ** exponent for exponent in range(n + 1)]
        for weight in integer_weights
    ]
    rows = [
        (
            math.prod(powers[i][count] for i, count in enumerate(counts)),
            counts,
        )
        for counts in types_of_length(n, k)
    ]
    rows.sort(reverse=True)

    remaining_capacity = K
    accepted_weight = 0
    selected_types = []
    for weight, counts in rows:
        size = type_multiplicity(counts)
        take = min(size, remaining_capacity)

        accepted_weight += take * weight

        if with_description:
            selected_types.append((counts, take, size, weight))

        remaining_capacity -= take
        if remaining_capacity == 0:
            break

    # P_e = (D^n - S) / D^n
    error = Fraction(Dn - accepted_weight, Dn)
    return UniformResult(n, q, ell, K, total_words, error, selected_types, D, alphabet)
