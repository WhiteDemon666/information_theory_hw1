from decimal import Decimal, InvalidOperation
from fractions import Fraction


class InputError(ValueError):
    pass


def parse_probability(value: str) -> Fraction:
    if not isinstance(value, str):
        raise InputError("Probability must be a string")
    value = value.strip().replace("∕", "/").replace("⁄", "/")
    if "/" in value:
        try:
            numerator, denominator = value.split("/")
            result = Fraction(int(numerator), int(denominator))
        except (ValueError, ZeroDivisionError) as exc:
            raise InputError(f"Invalid probability: {value}") from exc
    else:
        try:
            decimal = Decimal(value)
            if not decimal.is_finite():
                raise InputError(f"Invalid probability: {value}")
            result = Fraction(decimal)
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise InputError(f"Invalid probability: {value}") from exc
    if result < 0:
        raise InputError(f"Negative probability: {value}")
    if result > 1:
        raise InputError(f"Probability exceeds one: {value}")
    return result


def parse_distribution(data: dict[str, str]) -> dict[str, Fraction]:
    if not isinstance(data, dict) or not data:
        raise InputError("Distribution must be a non-empty object")
    result = {key: parse_probability(value) for key, value in data.items()}
    if sum(result.values(), Fraction(0)) != 1:
        raise InputError("Distribution is not normalized")
    return result
