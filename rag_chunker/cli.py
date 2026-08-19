"""Command-line entry point: read a markdown file, write JSON chunks."""

import argparse
import json
import sys

from .chunker import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP, chunk_markdown

__all__ = ["main", "build_parser"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rag-chunker",
        description="Structure-aware markdown chunking for retrieval pipelines.",
    )
    parser.add_argument("path", help="markdown file to chunk, or - for stdin")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help="chunk size ceiling, heading prefix included (default: %(default)s)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        metavar="N",
        help="trailing tokens repeated in the next chunk of a section (default: %(default)s)",
    )
    parser.add_argument(
        "--no-heading-prefix",
        action="store_true",
        help="do not prepend the heading path to the chunk text",
    )
    parser.add_argument(
        "--array",
        action="store_true",
        help="emit one indented JSON array instead of JSON lines",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print a size summary to stderr",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the result to a file (default: stdout)",
    )
    return parser


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_output(path, content):
    if path is None:
        sys.stdout.write(content)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)


def _stats_line(chunks):
    if not chunks:
        return "0 chunks\n"
    tokens = [chunk.token_estimate for chunk in chunks]
    oversized = sum(1 for chunk in chunks if chunk.oversized)
    return "%d chunks | tokens min %d avg %d max %d | %d oversized\n" % (
        len(chunks),
        min(tokens),
        round(sum(tokens) / float(len(tokens))),
        max(tokens),
        oversized,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        text = _read_input(args.path)
    except OSError as error:
        parser.error(str(error))

    try:
        chunks = chunk_markdown(
            text,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            heading_prefix=not args.no_heading_prefix,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.array:
        output = json.dumps([chunk.to_dict() for chunk in chunks], indent=2, ensure_ascii=False) + "\n"
    else:
        output = "".join(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n" for chunk in chunks)

    if args.stats:
        sys.stderr.write(_stats_line(chunks))

    _write_output(args.output, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
