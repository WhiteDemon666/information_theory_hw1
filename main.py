# GPT generated

import argparse
import json
import random
import sys
from time import perf_counter

from probabilities import InputError, parse_probability
from sources import load_source, empirical_frequency
from coding.asymptotic import eventual_error_answer
from coding.checks import check_memoryless_stationary
from coding.uniform import optimal_uniform_code, source_entropy


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="version",
        version="s=11; k=3, M=4, L=4, d=3, q=7, epsilon=10^-4",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("file")
    generate_parser.add_argument("-n", type=int)
    generate_parser.add_argument("--seed", type=int, default=0)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("file")
    analyze_parser.add_argument("--block", required=True)
    analyze_parser.add_argument("-n", type=int, required=True)
    analyze_parser.add_argument("--seed", type=int, default=0)

    check_parser = subparsers.add_parser("uniform-check")
    check_parser.add_argument("file")

    code_parser = subparsers.add_parser("uniform-code")
    code_parser.add_argument("file")
    code_parser.add_argument("--rate", required=True)
    code_parser.add_argument("--epsilon", default="1/10000")
    code_parser.add_argument("-q", "--q", required=True, type=int)
    code_parser.add_argument("--n-max", type=int, default=50)
    code_parser.add_argument("--code-n", type=int)
    code_parser.add_argument("--output", default="uniform_code.json")
    return parser


def parse_block(value: str) -> list[str]:
    block = [symbol.strip() for symbol in value.split(",")]
    if any(not symbol for symbol in block):
        raise InputError("Block must contain non-empty comma-separated symbols")
    return block


def run_generate(args):
    source = load_source(args.file)
    rng = random.Random(args.seed)
    if args.n is not None and args.n > 0:
        print(" ".join(source.generate(args.n, rng)))
        return
    try:
        for symbol in source.stream(rng):
            print(symbol, flush=True)
    except KeyboardInterrupt:
        print("Generation stopped by Ctrl+C", file=sys.stderr)


def run_analyze(args):
    source = load_source(args.file)
    block = parse_block(args.block)
    if args.n <= 0:
        raise InputError("N must be positive")
    if args.n < len(block):
        raise InputError("N must be at least block length")
    first_probability = source.block_probability(block)
    average_probability = source.average_block_probability(block)

    rng = random.Random(args.seed)
    sequence = source.generate(max(args.n, 10**6), rng)
    empirical = empirical_frequency(sequence[:args.n], block)

    print("Exact probability at first position:", first_probability)
    print("Average probability over first L positions:", average_probability)
    print("Empirical frequency:", empirical)
    print()
    print("N\tEmpirical\tError first\tError average")
    for n in (10**2, 10**3, 10**4, 10**5, 10**6):
        if n < len(block):
            print(n, "N < block length", sep="\t")
            continue
        current = empirical_frequency(sequence[:n], block)
        print(n, float(current), float(abs(current - first_probability)),
              float(abs(current - average_probability)), sep="\t")


def _checked_distribution(filename: str):
    source = load_source(filename)
    checked = check_memoryless_stationary(source)
    if not checked.stationary or not checked.memoryless:
        raise InputError(checked.reason)
    return checked.distribution


def run_uniform_check(args):
    distribution = _checked_distribution(args.file)
    print("Источник стационарен и не имеет памяти")
    print("Распределение:", ", ".join(f"{symbol}={p}" for symbol, p in distribution.items()))
    print(f"Энтропия H(X) = {source_entropy(distribution):.12f} бит/символ")


def run_uniform_code(args):
    if args.n_max < 1:
        raise InputError("n_max должно быть положительным")
    if args.code_n is not None and not 1 <= args.code_n <= args.n_max:
        raise InputError("code_n должно находиться между 1 и n_max")

    distribution = _checked_distribution(args.file)
    epsilon = parse_probability(args.epsilon)
    if not 0 < epsilon < 1:
        raise InputError("epsilon должно находиться строго между 0 и 1")

    start = perf_counter()
    print("n\tell\tP_e(n)\tP_e(n) приближённо")
    first_suitable_n = None
    for n in range(1, args.n_max + 1):
        result = optimal_uniform_code(distribution, n, args.rate, args.q)
        print(f"{n}\t{result.ell}\t{result.error}\t{float(result.error):.9g}")
        if result.error <= epsilon and first_suitable_n is None:
            first_suitable_n = n

    elapsed = perf_counter() - start
    if first_suitable_n is None:
        print(f"Для n <= {args.n_max} подходящего размера блока нет")
    else:
        print("Минимальное подходящее n на рассмотренном отрезке:", first_suitable_n)

    answer = eventual_error_answer(distribution, args.rate, epsilon, args.q)
    exists = "да" if answer.exists else ("нет" if answer.exists is False else "не определено")
    print(f"Энтропия H(X) = {answer.entropy:.12f} бит/символ")
    print(f"Существует n0, после которого P_e(n) <= epsilon при всех n >= n0: {exists}")
    print(answer.explanation)
    print(f"Время расчёта таблицы: {elapsed:.4f} с")

    # По умолчанию сохраняем код минимальной подходящей длины, а если её нет — для n_max
    code_n = args.code_n if args.code_n is not None else (first_suitable_n or args.n_max)
    result = optimal_uniform_code(distribution, code_n, args.rate, args.q, with_description=True)
    description = result.compact_code_description()
    description["source_file"] = args.file
    description["rate"] = args.rate
    description["epsilon"] = str(epsilon)
    try:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(description, file, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise InputError(f"Не удалось сохранить описание кода: {exc}") from exc
    print(f"Описание кода для n={code_n} сохранено: {args.output}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "generate":
            run_generate(args)
        elif args.command == "analyze":
            run_analyze(args)
        elif args.command == "uniform-check":
            run_uniform_check(args)
        elif args.command == "uniform-code":
            run_uniform_code(args)
    except (InputError, BrokenPipeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
