# GPT generated

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from probabilities import InputError
from sources.markov import MarkovSource
from sources.mixture import MixtureSource
from sources.periodic import PeriodicSource


@dataclass
class _Automaton:
    # Каждая строка хранит переходы: символ -> {следующее состояние: вероятность}
    transitions: list[dict[str, dict[int, Fraction]]]
    initial: dict[int, Fraction]


def _markov_automaton(source: MarkovSource) -> _Automaton:
    # Первые d символов выдаются из выбранного начального контекста
    # Затем состояние описывается парой (последний контекст, фаза)
    transitions = []
    indices = {}
    queue = deque()

    def add_state(key):
        if key not in indices:
            indices[key] = len(transitions)
            transitions.append({})
            queue.append(key)
        return indices[key]

    initial = {}
    for context, weight in source.initial.items():
        if weight > 0:
            initial[add_state(("начало", context, 0))] = weight

    while queue:
        key = queue.popleft()
        current_id = indices[key]
        if key[0] == "начало":
            _, context, index = key
            symbol = context[index]
            if index + 1 < source.depth:
                nxt = ("начало", context, index + 1)
            else:
                nxt = ("переход", context, source.depth % source.period)
            transitions[current_id] = {symbol: {add_state(nxt): Fraction(1)}}
        else:
            _, context, phase = key
            distribution = source.distribution_at(context, phase)
            row = {}
            for symbol, weight in distribution.items():
                if weight > 0:
                    nxt = ("переход", context[1:] + (symbol,), (phase + 1) % source.period)
                    row[symbol] = {add_state(nxt): weight}
            transitions[current_id] = row

    return _Automaton(transitions, initial)


def _periodic_automaton(source: PeriodicSource) -> _Automaton:
    transitions = []
    for phase in range(source.period):
        nxt = (phase + 1) % source.period
        row = {
            symbol: {nxt: probability}
            for symbol, probability in source.distribution_at(phase).items()
            if probability > 0
        }
        transitions.append(row)
    return _Automaton(transitions, {0: Fraction(1)})


def _mixture_automaton(source: MixtureSource) -> _Automaton:
    # Компоненты объединяются как непересекающиеся множества состояний
    transitions = []
    initial = {}
    for name, weight in source.weights.items():
        if weight == 0:
            continue
        component = _build_automaton(source.components[name])
        offset = len(transitions)
        for row in component.transitions:
            transitions.append({
                symbol: {target + offset: p for target, p in targets.items()}
                for symbol, targets in row.items()
            })
        for state, p in component.initial.items():
            initial[state + offset] = weight * p
    return _Automaton(transitions, initial)


def _build_automaton(source) -> _Automaton:
    if isinstance(source, PeriodicSource):
        return _periodic_automaton(source)
    if isinstance(source, MarkovSource):
        return _markov_automaton(source)
    if isinstance(source, MixtureSource):
        return _mixture_automaton(source)
    raise InputError("Неизвестный тип источника")


def _alphabet(machine: _Automaton) -> list[str]:
    return sorted({symbol for row in machine.transitions for symbol in row})


def _next_vector(machine: _Automaton, symbol: str, vector: list[Fraction]) -> list[Fraction]:
    # Умножение столбца на матрицу переходов по конкретному символу
    result = []
    for row in machine.transitions:
        result.append(sum(
            (weight * vector[target] for target, weight in row.get(symbol, {}).items()),
            Fraction(0),
        ))
    return result


def _annihilates_all_words(machine: _Automaton, difference: list[Fraction]) -> bool:
    # Порождаем линейную оболочку векторов A_w * 1 для всех слов w
    # Размерность оболочки не превышает числа состояний автомата
    n = len(machine.transitions)
    basis = {}
    pending = deque([[Fraction(1)] * n])
    alphabet = _alphabet(machine)

    while pending:
        vector = pending.popleft()
        for pivot, normal in basis.items():
            coefficient = vector[pivot]
            if coefficient:
                vector = [v - coefficient * b for v, b in zip(vector, normal)]

        pivot = next((i for i, x in enumerate(vector) if x), None)
        if pivot is None:
            continue

        factor = vector[pivot]
        vector = [x / factor for x in vector]
        if sum((a * b for a, b in zip(difference, vector)), Fraction(0)):
            return False

        basis[pivot] = vector
        for symbol in alphabet:
            pending.append(_next_vector(machine, symbol, vector))
    return True


def _stationary(machine: _Automaton) -> bool:
    n = len(machine.transitions)
    shifted = [Fraction(0)] * n
    initial = [machine.initial.get(i, Fraction(0)) for i in range(n)]
    for state, p in machine.initial.items():
        for destinations in machine.transitions[state].values():
            for target, weight in destinations.items():
                shifted[target] += p * weight
    return _annihilates_all_words(
        machine, [x - y for x, y in zip(initial, shifted)]
    )


def _marginals(machine: _Automaton) -> dict[str, Fraction]:
    result = {}
    for state, p in machine.initial.items():
        for symbol, destinations in machine.transitions[state].items():
            result[symbol] = result.get(symbol, Fraction(0)) + p * sum(destinations.values())
    return result


def _iid(machine: _Automaton, p: dict[str, Fraction]) -> bool:
    # Сравниваем закон исходного процесса с одноcостоянием iid-автоматом
    # Равенство для всех слов доказывается проверкой конечномерной оболочки
    n = len(machine.transitions)
    transitions = [dict(row) for row in machine.transitions]
    transitions.append({symbol: {n: weight} for symbol, weight in p.items() if weight > 0})
    difference = [machine.initial.get(i, Fraction(0)) for i in range(n)] + [Fraction(-1)]
    return _annihilates_all_words(_Automaton(transitions, {}), difference)


@dataclass
class SourceCheck:
    stationary: bool
    memoryless: bool | None
    distribution: dict[str, Fraction] | None
    reason: str


def check_memoryless_stationary(source) -> SourceCheck:
    machine = _build_automaton(source)
    if not _stationary(machine):
        return SourceCheck(False, None, None, "Нарушена стационарность")
    distribution = _marginals(machine)
    if not _iid(machine, distribution):
        return SourceCheck(True, False, None, "Источник стационарен, но имеет память")
    return SourceCheck(True, True, distribution, "Источник стационарен и не имеет памяти")
