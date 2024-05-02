import argparse

import src


__all__ = []


def main() -> None: 
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='func')
    src.embdArgs(subparsers.add_parser('embd'))
    src.bam2seqArgs(subparsers.add_parser('bam2seq'))
    src.seq2embdArgs(subparsers.add_parser('seq2embd'))
    args = parser.parse_args()

    if args.func == "embd":      src.embd(**vars(args))
    if args.func == "bam2seq":   src.bam2seq(**vars(args))
    if args.func == "seq2embd":  src.seq2embd(**vars(args))


if __name__ == "__main__": main()
