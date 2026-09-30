"""CLI-level tests: build_parser defaults and main() against the doc fixture.

test_chunker.py and test_doc_fixture.py already pin the chunking behaviour
itself, so these only cover the argparse wiring and the stdout/stderr/file
paths main() takes through it.
"""

import io
import json
from pathlib import Path

import pytest

from rag_chunker.cli import build_parser, main

DOC_PATH = str(Path(__file__).parent / "fixtures" / "doc.md")
DOC_TEXT = Path(DOC_PATH).read_text()


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([DOC_PATH])
    assert args.path == DOC_PATH
    assert args.max_tokens == 512
    assert args.overlap == 64
    assert args.no_heading_prefix is False
    assert args.array is False
    assert args.stats is False
    assert args.output is None


def test_main_writes_jsonl_to_stdout(capsys):
    exit_code = main([DOC_PATH, "--max-tokens", "512", "--overlap", "64"])
    assert exit_code == 0

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 3

    records = [json.loads(line) for line in lines]
    assert [r["heading_path"] for r in records] == [
        ["Vector index runbook"],
        ["Vector index runbook", "Reindex"],
        ["Vector index runbook", "Checks"],
    ]
    assert [r["index"] for r in records] == [0, 1, 2]


def test_main_array_flag_emits_one_json_array(capsys):
    exit_code = main([DOC_PATH, "--array"])
    assert exit_code == 0

    out = capsys.readouterr().out
    records = json.loads(out)
    assert isinstance(records, list)
    assert len(records) == 3
    assert records[0]["heading_path"] == ["Vector index runbook"]


def test_main_stats_flag_prints_summary_to_stderr(capsys):
    exit_code = main([DOC_PATH, "--stats"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert captured.err == "3 chunks | tokens min 23 avg 61 max 101 | 0 oversized\n"


def test_main_no_heading_prefix_drops_prefix_from_text_not_from_path(capsys):
    main([DOC_PATH, "--no-heading-prefix"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[1]["heading_path"] == ["Vector index runbook", "Reindex"]
    assert not records[1]["text"].startswith("Vector index runbook")


def test_main_writes_to_output_file(tmp_path):
    out_path = tmp_path / "chunks.jsonl"
    main([DOC_PATH, "-o", str(out_path)])
    lines = out_path.read_text().splitlines()
    assert len(lines) == 3


def test_main_rejects_overlap_not_smaller_than_max_tokens(capsys):
    with pytest.raises(SystemExit):
        main([DOC_PATH, "--max-tokens", "10", "--overlap", "10"])
    assert "overlap must be smaller than max_tokens" in capsys.readouterr().err


def test_main_reads_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(DOC_TEXT))
    exit_code = main(["-"])
    assert exit_code == 0

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["heading_path"] for r in records] == [
        ["Vector index runbook"],
        ["Vector index runbook", "Reindex"],
        ["Vector index runbook", "Checks"],
    ]


def test_main_rejects_binary_file(tmp_path, capsys):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x00")
    with pytest.raises(SystemExit) as excinfo:
        main([str(path)])
    assert excinfo.value.code == 2
    assert "is not valid UTF-8 text" in capsys.readouterr().err


def test_main_rejects_binary_stdin(monkeypatch, capsys):
    stream = io.TextIOWrapper(io.BytesIO(b"# ok\n\xff\xfe\x00bad"), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", stream)
    with pytest.raises(SystemExit) as excinfo:
        main(["-"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "stdin is not valid UTF-8 text" in err


def test_main_stats_with_empty_input_reports_zero_chunks(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    exit_code = main(["-", "--stats"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "0 chunks\n"


def test_main_array_with_empty_input_emits_empty_array(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("\n\n"))
    main(["-", "--array"])
    assert json.loads(capsys.readouterr().out) == []


def test_main_rejects_missing_file(capsys):
    with pytest.raises(SystemExit):
        main([str(Path(DOC_PATH).parent / "does-not-exist.md")])
    assert "No such file or directory" in capsys.readouterr().err
