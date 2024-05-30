import argparse

import src


__all__ = []


def main() -> None: 
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='func')
    src.bam2seqArgs(subparsers.add_parser('bam2seq'))
    src.seq2embdArgs(subparsers.add_parser('seq2embd'))
    src.embd2hashArgs(subparsers.add_parser('embd2hash'))
    args = parser.parse_args()

    if args.func == "bam2seq":   src.bam2seq(**vars(args))
    if args.func == "seq2embd":  src.seq2embd(**vars(args))
    if args.func == "embd2hash": src.embd2hash(**vars(args))


if __name__ == "__main__": main()
