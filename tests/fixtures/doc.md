# Vector index runbook

This runbook covers the nightly reindex job and the checks that follow it.

## Reindex

Run the job from the scheduler, never from a laptop:

```bash
python -m pipeline.reindex \
  --source s3://docs/current \
  --max-tokens 512 \
  --overlap 64
```

## Checks

Confirm the index is healthy before you close the ticket.

| Check | Command | Expect |
| --- | --- | --- |
| Doc count | `curl -s $INDEX/_count` | matches source within 0.1% |
| Query latency | `curl -s $INDEX/_stats/search` | p95 under 200ms |
| Stale segments | `curl -s $INDEX/_segments` | none older than 24h |
