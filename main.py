import argparse
import random
import sys

from probabilities import InputError
from sources import load_source, empirical_frequency


def build_parser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("file")
    generate_parser.add_argument("-n", type=int)
    generate_parser.add_argument("--seed", type=int, default=0)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("file")
    analyze_parser.add_argument("--block", required=True)
    analyze_parser.add_argument("-n", type=int, required=True)
    analyze_parser.add_argument("--seed", type=int, default=0)

    return parser


def parse_block(value: str) -> list[str]:
    block = value.split(",")

    if not block:
        raise InputError("Block must not be empty")

    return [symbol.strip() for symbol in block]


def run_generate(args):
    source = load_source(args.file)

    if args.n is None or args.n <= 0:
        raise InputError("N must be positive")

    rng = random.Random(args.seed)
    sequence = source.generate(args.n, rng)
    print(" ".join(sequence))


def run_analyze(args):
    source = load_source(args.file)
    block = parse_block(args.block)

    if args.n <= 0:
        raise InputError("N must be positive")

    if args.n < len(block):
        raise InputError("N must be at least block length")

    first_probability = source.block_probability(block)

    average_probability = (
        source.average_block_probability(block)
    )

    rng = random.Random(args.seed)
    sequence = source.generate(
        max(args.n, 10**6),
        rng,
    )
    empirical = empirical_frequency(
        sequence[:args.n],
        block,
    )

    print("Exact probability at first position:")
    print(first_probability)

    print("Average probability over phases:")
    print(average_probability)

    print("Empirical frequency:")
    print(empirical)

    print()
    print("N\tEmpirical\tError first\tError average")

    for n in (
        10**2,
        10**3,
        10**4,
        10**5,
        10**6,
    ):
        current_frequency = empirical_frequency(
            sequence[:n],
            block,
        )

        error_first = abs(
            current_frequency - first_probability
        )

        error_average = abs(
            current_frequency - average_probability
        )

        print(
            n,
            float(current_frequency),
            float(error_first),
            float(error_average),
            sep="\t",
        )


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "generate":
            run_generate(args)

        elif args.command == "analyze":
            run_analyze(args)

        else:
            parser.print_help()

    except InputError as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
