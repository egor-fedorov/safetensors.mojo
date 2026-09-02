# Survival fuzzing

The two fuzz harnesses exercise hostile Safetensors files and shard-index JSON
against the parser, buffered readers, memory-mapped readers, raw mapped spans,
and representative native typed views. Their only verdict is process survival.
A case may be accepted or rejected without changing the result; a panic, memory
fault, nonzero harness exit, or CI timeout fails the run.

## Deterministic run

Run both deterministic corpora with:

```text
pixi run fuzz
pixi run fuzz-index
```

The `fuzz` task first invokes `fuzz-generate`. The generator derives mutations
from the committed `.safetensors` fixtures, adds structured boundary cases, and
writes the ignored corpus to `.pixi/fuzz-corpus`. The default seed is
`20260825`. The generator prints the seed and mutation count so a failing run
can be reproduced.

The `fuzz-index` task similarly invokes `fuzz-index-generate`, mutates committed
shard-index fixtures, adds schema, exact-integer, duplicate-key, nesting,
routing, and path-policy boundaries, and writes `.pixi/fuzz-index-corpus`. Every
case sits beside valid local shards so accepted documents exercise both public
index readers and consume their tensor bytes.

Fuzzing is deliberately separate from `check` and `all`. The regular suites
and committed valid and malformed fixtures provide verdict-based correctness
coverage; this larger corpus adds a survival check without lengthening every
local validation run.

## Harness invariants

Broad exception handlers around library entry points are intentional. A
structured parser, validation, mapping, or reader error is a normal rejection
for an arbitrary hostile input. Corpus setup failures, such as a missing case
file, occur outside those handlers and fail the harness.

Every byte in an accepted mapped span and owned reader result is loaded and
folded into the printed touch checksum. Values from successful representative
typed views are folded into the same checksum. The checksum is an observable
consumer that prevents an optimizing compiler from treating those reads as
dead code; it is not a correctness oracle.

Accepted, rejected, mapped, and read counts are diagnostic only. They depend on
the seed and the committed fixture set and must never be asserted as stable
results.

## Randomized and larger runs

For exploratory or scheduled runs, record a random seed and increase the
mutation count:

```bash
seed="${RANDOM}"
pixi run python tools/fuzz/generate_corpus.py --seed "${seed}" --mutations 20000
pixi run mojo run -I src tests/fuzz/fuzz_harness.mojo .pixi/fuzz-corpus
pixi run python tools/fuzz/generate_index_corpus.py --seed "${seed}" --mutations 20000
pixi run mojo run -I src tests/fuzz/index_fuzz_harness.mojo .pixi/fuzz-index-corpus
```

The generator reports the selected seed in its output. Replay a failure with
that seed and the same fixture revision. During local triage, an exception
handler can temporarily bind and print its error to reveal which rejection
path ran; committed CI behavior should retain the survival-only verdict.
