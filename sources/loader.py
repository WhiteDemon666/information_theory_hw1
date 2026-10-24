# GPT generated

import json
from collections import deque
from fractions import Fraction

from probabilities import InputError, parse_distribution, parse_probability
from .markov import MarkovSource
from .mixture import MixtureSource
from .periodic import PeriodicSource, get_symbol_distribution


def _object(data, name: str) -> dict:
    if not isinstance(data, dict) or not data:
        raise InputError(f"{name} must be a non-empty JSON object")
    if any(not isinstance(key, str) for key in data):
        raise InputError(f"{name} must have string keys")
    return data


def _distribution(data, name: str) -> dict[str, Fraction]:
    _object(data, name)
    if any(not isinstance(value, str) for value in data.values()):
        raise InputError(f"{name}: probabilities must be strings")
    return parse_distribution(data)


def _source_list(data, switches: dict) -> list[str]:
    if not isinstance(data, list) or not data:
        raise InputError("Source must be a non-empty list")
    if any(not isinstance(name, str) or name not in switches for name in data):
        raise InputError("Source references an unknown switch")
    return data


def _switches(data, models: dict) -> dict[str, dict[str, Fraction]]:
    _object(data, "switches")
    result = {}
    for switch_name, raw in data.items():
        distribution = _distribution(raw, f"switch {switch_name}")
        for model_name in distribution:
            if model_name not in models:
                raise InputError(f"Unknown model: {model_name}")
        result[switch_name] = distribution
    return result


def parse_legacy_source(data: dict) -> PeriodicSource:
    raw_models = _object(data.get("models"), "models")
    models = {
        name: _distribution(raw, f"model {name}")
        for name, raw in raw_models.items()
    }
    switches = _switches(data.get("switches"), models)
    source = _source_list(data.get("source"), switches)
    phases = [
        get_symbol_distribution(switches[name], models)
        for name in source
    ]
    return PeriodicSource(models, switches, source, phases)


def _context(raw, depth: int, alphabet: list[str]) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) != depth:
        raise InputError(f"Context must contain {depth} symbols")
    if any(not isinstance(x, str) or x not in alphabet for x in raw):
        raise InputError(f"Unknown symbol in context: {raw}")
    return tuple(raw)


def parse_markov_source(data: dict) -> MarkovSource:
    alphabet = data.get("alphabet")
    depth = data.get("depth")
    if (
        not isinstance(alphabet, list)
        or not alphabet
        or not all(isinstance(x, str) for x in alphabet)
        or len(set(alphabet)) != len(alphabet)
    ):
        raise InputError("Invalid alphabet")
    if type(depth) is not int or depth <= 0:
        raise InputError("Depth must be a positive integer")

    initial_data = data.get("initial")
    if not isinstance(initial_data, list) or not initial_data:
        raise InputError("Initial must be a non-empty list")
    initial = {}
    for entry in initial_data:
        if not isinstance(entry, dict):
            raise InputError("Invalid initial entry")
        ctx = _context(entry.get("context"), depth, alphabet)
        if ctx in initial:
            raise InputError(f"Duplicate initial context: {ctx}")
        raw_p = entry.get("probability")
        if not isinstance(raw_p, str):
            raise InputError("Initial probability must be a string")
        initial[ctx] = parse_probability(raw_p)
    if sum(initial.values(), Fraction(0)) != 1:
        raise InputError("Initial distribution is not normalized")

    models = {}
    for name, entries in _object(data.get("models"), "models").items():
        if not isinstance(entries, list) or not entries:
            raise InputError(f"Model {name} must be a non-empty list")
        transitions = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise InputError(f"Invalid context entry in model {name}")
            ctx = _context(entry.get("context"), depth, alphabet)
            if ctx in transitions:
                raise InputError(f"Duplicate context {ctx} in model {name}")
            dist = _distribution(entry.get("distribution"), f"model {name} context {ctx}")
            if any(symbol not in alphabet for symbol in dist):
                raise InputError(f"Unknown symbol in model {name}")
            transitions[ctx] = dist
        models[name] = transitions

    switches = _switches(data.get("switches"), models)
    source = _source_list(data.get("source"), switches)
    result = MarkovSource(alphabet, depth, initial, models, switches, source)

    # Проверяем возможные пары (контекст, фаза)
    period = len(source)
    start_phase = depth % period
    queue = deque(
        (ctx, start_phase) for ctx, p in initial.items() if p > 0
    )
    seen = set(queue)
    while queue:
        ctx, phase = queue.popleft()
        sw = switches[source[phase]]
        for name, weight in sw.items():
            if weight == 0:
                continue
            if ctx not in models[name]:
                raise InputError(f"Missing reachable context {ctx} in model {name}")
            for symbol, p in models[name][ctx].items():
                if p == 0:
                    continue
                nxt = (ctx[1:] + (symbol,), (phase + 1) % period)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return result


def parse_mixture_source(data: dict, level: int) -> MixtureSource:
    entries = _object(data.get("components"), "components")
    components = {}
    raw_weights = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise InputError(f"Invalid mixture component: {name}")
        raw_weights[name] = entry.get("weight")
        if not isinstance(raw_weights[name], str):
            raise InputError(f"Invalid weight of component {name}")
        components[name] = parse_source_data(entry.get("source"), level + 1)

    weights = _distribution(raw_weights, "mixture weights")
    periods = {component.period for component in components.values()}
    if len(periods) != 1:
        raise InputError("Mixture components must use the same source period")
    return MixtureSource(components, weights, periods.pop())


def parse_source_data(data, level: int = 0):
    if level > 20:
        raise InputError("Mixture nesting is too deep")
    if not isinstance(data, dict):
        raise InputError("Source description must be a JSON object")
    kind = data.get("type")
    if kind is None:
        return parse_legacy_source(data)
    if kind == "markov":
        return parse_markov_source(data)
    if kind == "mixture":
        return parse_mixture_source(data, level)
    raise InputError(f"Unknown source type: {kind}")


def load_source(filename: str):
    try:
        with open(filename, encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError) as exc:
        raise InputError(f"Cannot read source file: {filename}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON: {exc}") from exc
    return parse_source_data(data)
