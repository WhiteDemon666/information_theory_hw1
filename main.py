import argparse
import random
import sys

from probabilities import InputError
from sources import load_source, empirical_frequency


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


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "generate":
            run_generate(args)
        elif args.command == "analyze":
            run_analyze(args)
    except (InputError, BrokenPipeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
