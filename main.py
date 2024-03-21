import argparse

import src


__all__ = []


def main() -> None: 
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='func')
    src.embdArgs(subparsers.add_parser('embd'))
    src.umapArgs(subparsers.add_parser('umap'))
    src.bam2hdfArgs(subparsers.add_parser('bam2hdf'))
    src.hdf2embdArgs(subparsers.add_parser('hdf2embd'))
    args = parser.parse_args()

    if args.func == "embd":
        src.embd(**vars(args))
    if args.func == "umap":
        src.umap(**vars(args))
    if args.func == "bam2hdf": 
        src.bam2hdf(**vars(args))
    if args.func == "hdf2embd": 
        src.hdf2embd(**vars(args))


if __name__ == "__main__": main()
