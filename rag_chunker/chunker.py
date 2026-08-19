"""The chunk packer: turns parsed blocks into budget-sized, heading-aware chunks.

The packing unit is not the block but the "piece": a block is broken into
pieces along the finest boundary we are willing to cut at (sentences for a
paragraph, items for a list, the whole thing for code and tables, which never
split). Pieces are then packed greedily into chunks in document order, which
keeps the algorithm linear-ish and the output predictable -- no block ever
gets reordered or split across a heading, and a piece is only ever duplicated
by the overlap carry, never invented or dropped.
"""

import re
from collections import namedtuple

from .markdown import parse_blocks
from .sentences import split_sentences
from .tokens import estimate_tokens

__all__ = [
    "Chunk",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OVERLAP",
    "chunk_markdown",
    "chunks_to_jsonl",
]

DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP = 64

_LIST_ITEM_START = re.compile(r"^[ \t]*([-*+]|[0-9]+[.)])[ \t]+")

Piece = namedtuple("Piece", "text start_line end_line atomic sep_before tokens")


class Chunk(object):
    """One packed unit of text ready to embed."""

    __slots__ = (
        "index",
        "text",
        "body",
        "heading_path",
        "start_line",
        "end_line",
        "token_estimate",
        "oversized",
    )

    def __init__(self, text, body, heading_path, start_line, end_line, token_estimate, oversized, index=0):
        self.index = index
        self.text = text
        self.body = body
        self.heading_path = heading_path
        self.start_line = start_line
        self.end_line = end_line
        self.token_estimate = token_estimate
        self.oversized = oversized

    def to_dict(self):
        return {
            "index": self.index,
            "text": self.text,
            "heading_path": list(self.heading_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_estimate": self.token_estimate,
        }

    def __repr__(self):
        return "Chunk(index=%d, lines=%d-%d, tokens=%d, oversized=%s)" % (
            self.index,
            self.start_line,
            self.end_line,
            self.token_estimate,
            self.oversized,
        )


def chunk_markdown(text, max_tokens=DEFAULT_MAX_TOKENS, overlap=DEFAULT_OVERLAP, heading_prefix=True):
    """Split ``text`` into a list of :class:`Chunk` in document order."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    chunks = []
    for heading_path, section_blocks in _sections(parse_blocks(text)):
        chunks.extend(_pack_section(heading_path, section_blocks, max_tokens, overlap, heading_prefix))
    for index, chunk in enumerate(chunks):
        chunk.index = index
    return chunks


def chunks_to_jsonl(chunks):
    """Serialise ``chunks`` as JSON Lines (one object per line, no trailing newline)."""
    import json

    return "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks)


def _sections(blocks):
    """Group blocks by the heading path in effect when they appear.

    A heading does not start a new section by itself if it introduces no
    content before the next heading -- an empty section carries nothing to
    chunk, so it is dropped rather than emitted as an empty chunk.
    """
    sections = []
    stack = []  # (level, title), deepest last
    current_path = ()
    current_blocks = []

    for block in blocks:
        if block.kind == "heading":
            if current_blocks:
                sections.append((current_path, current_blocks))
            current_blocks = []
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.title))
            current_path = tuple(title for _, title in stack)
        else:
            current_blocks.append(block)

    if current_blocks:
        sections.append((current_path, current_blocks))
    return sections


def _split_list_items(text):
    """Split a list block into items, keeping wrapped continuation lines attached."""
    items = []
    current = []
    for line in text.split("\n"):
        if _LIST_ITEM_START.match(line) and current:
            items.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        items.append("\n".join(current))
    return items


def _blocks_to_pieces(blocks):
    pieces = []
    for block in blocks:
        if block.kind in ("code", "table"):
            pieces.append(
                Piece(block.text, block.start_line, block.end_line, True, "\n\n", estimate_tokens(block.text))
            )
        elif block.kind == "paragraph":
            for i, sentence in enumerate(split_sentences(block.text)):
                sep = "\n\n" if i == 0 else " "
                pieces.append(Piece(sentence, block.start_line, block.end_line, False, sep, estimate_tokens(sentence)))
        elif block.kind == "list":
            for i, item in enumerate(_split_list_items(block.text)):
                sep = "\n\n" if i == 0 else "\n"
                pieces.append(Piece(item, block.start_line, block.end_line, False, sep, estimate_tokens(item)))
    return pieces


def _assemble(pieces):
    parts = []
    for i, piece in enumerate(pieces):
        if i > 0:
            parts.append(piece.sep_before)
        parts.append(piece.text)
    return "".join(parts)


def _full_text(prefix, body):
    if prefix and body:
        return prefix + "\n\n" + body
    return body or prefix


def _overlap_carry(chunk_pieces, overlap_tokens):
    """Trailing non-atomic pieces of a chunk, up to ``overlap_tokens``, to repeat next."""
    if overlap_tokens <= 0:
        return []
    carry = []
    total = 0
    for piece in reversed(chunk_pieces):
        if piece.atomic:
            break
        carry.insert(0, piece)
        total += piece.tokens
        if total >= overlap_tokens:
            break
    return carry


def _pack_section(heading_path, blocks, max_tokens, overlap, heading_prefix):
    pieces = _blocks_to_pieces(blocks)
    if not pieces:
        return []

    prefix = " > ".join(heading_path) if heading_prefix and heading_path else ""
    total = len(pieces)
    chunks = []
    carry = []
    idx = 0

    while idx < total or carry:
        if carry and idx < total and pieces[idx].atomic:
            # Carried prose does not belong glued to the front of a code
            # block or table -- let the atomic piece start its own chunk.
            carry = []
        idx_start = idx
        current = list(carry)
        carry = []

        while idx < total:
            trial = current + [pieces[idx]]
            trial_text = _full_text(prefix, _assemble(trial))
            if estimate_tokens(trial_text) <= max_tokens:
                current = trial
                idx += 1
                continue
            break

        if idx == idx_start and idx < total:
            # Nothing new fit alongside whatever we started with (carry, or
            # nothing) -- take one piece anyway so the section keeps moving.
            current = current + [pieces[idx]]
            idx += 1

        if not current:
            break

        body = _assemble(current)
        text = _full_text(prefix, body)
        tokens = estimate_tokens(text)
        chunks.append(
            Chunk(
                text=text,
                body=body,
                heading_path=list(heading_path),
                start_line=current[0].start_line,
                end_line=current[-1].end_line,
                token_estimate=tokens,
                oversized=tokens > max_tokens,
            )
        )

        carry = _overlap_carry(current, overlap) if idx < total else []

    return chunks
