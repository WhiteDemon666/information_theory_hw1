from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation, localcontext
from fractions import Fraction

from probabilities import InputError


@dataclass
class AsymptoticAnswer:
    exists: bool | None
    explanation: str
    entropy: Decimal
    relation: str


def exact_entropy_decimal(distribution: dict[str, Fraction]) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 110
        log_two = Decimal(2).ln()
        total = Decimal(0)

        for p in distribution.values():
            if p == 0:
                continue

            probability = Decimal(p.numerator) / Decimal(p.denominator)
            total -= probability * probability.ln() / log_two

        return +total


def eventual_error_answer(
    distribution: dict[str, Fraction],
    rate: str,
    epsilon: Fraction,
    q: int,
) -> AsymptoticAnswer:
    # допустимость параметров кодирования
    if type(q) is not int or q < 2:
        raise InputError("q должно быть целым числом не менее 2")

    if not isinstance(epsilon, Fraction) or not 0 < epsilon < 1:
        raise InputError("Для асимптотического критерия нужно 0 < epsilon < 1")

    if (
        not isinstance(distribution, dict)
        or not distribution
        or any(not isinstance(p, Fraction) or p < 0 for p in distribution.values())
        or sum(distribution.values(), Fraction(0)) != 1
    ):
        raise InputError("Распределение источника должно быть нормированным")

    try:
        R = Decimal(rate)
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise InputError("Некорректная скорость R") from exc

    if not R.is_finite() or R < 0:
        raise InputError("Скорость R должна быть конечной и неотрицательной")

    try:
        H = exact_entropy_decimal(distribution)

        # допуск нужен из-за численной погрешности логарифмов
        with localcontext() as ctx:
            ctx.prec = 110
            delta = R - H
            tolerance = Decimal("1e-85")

            if delta > tolerance:
                return AsymptoticAnswer(
                    True,
                    "R > H(X): по прямой теореме кодирования оптимальная "
                    "вероятность ошибки стремится к нулю.",
                    H,
                    ">",
                )

            if delta < -tolerance:
                return AsymptoticAnswer(
                    False,
                    "R < H(X): по обратной теореме оптимальная "
                    "вероятность ошибки стремится к единице. "
                    "Для малых n ошибка всё же может быть небольшой.",
                    H,
                    "<",
                )

    except DecimalException as exc:
        raise InputError("Не удалось вычислить или сравнить R и H(X)") from exc

    support = [p for p in distribution.values() if p > 0]

    if len(support) == 1:
        # У детерминированного источника есть ровно один возможный блок
        # каждой длины, поэтому вероятность ошибки всегда равна нулю
        return AsymptoticAnswer(
            True,
            "Детерминированный источник: вероятность ошибки равна нулю "
            "при любом n.",
            H,
            "≈",
        )

    return AsymptoticAnswer(
        None,
        "R приблизительно равно H(X). Основные теоремы кодирования "
        "не позволяют сделать общий вывод на границе. "
        "Для окончательного ответа нужен дополнительный анализ.",
        H,
        "≈",
    )
