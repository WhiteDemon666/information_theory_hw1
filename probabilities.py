from decimal import Decimal, InvalidOperation
from fractions import Fraction


class InputError(ValueError):
    pass


def parse_probability(value: str) -> Fraction:
    value = value.strip()
    value = value.replace("∕", "/").replace("⁄", "/")

    if "/" in value:
        numerator, denominator = value.split("/", 1)

        try:
            result = Fraction(int(numerator), int(denominator))
        except (ValueError, ZeroDivisionError):
            raise InputError(f"Invalid probability: {value}")
    else:
        try:
            result = Fraction(Decimal(value))
        except InvalidOperation:
            raise InputError(f"Invalid probability: {value}")

    if result < 0:
        raise InputError(f"Negative probability: {value}")

    return result


def parse_distribution(data: dict[str, str]) -> dict[str, Fraction]:
    if not data:
        raise InputError("Distribution must not be empty")

    result = {}

    for symbol, probability in data.items():
        result[symbol] = parse_probability(probability)

    if sum(result.values(), Fraction(0, 1)) != 1:
        raise InputError("Distribution is not normalized")

    return result
