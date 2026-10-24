import json

from probabilities import InputError, parse_distribution
from .periodic import PeriodicSource, get_symbol_distribution


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
