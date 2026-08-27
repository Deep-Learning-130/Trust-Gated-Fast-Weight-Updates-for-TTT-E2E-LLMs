"""Facts about the token space, recovered from the vendor's configs.

The vendor has no tokenizer
---------------------------
There is no tokenizer library in `vendor/ttt-e2e/pyproject.toml` at all -- no
`tiktoken`, no `transformers`, no `sentencepiece`, no `tokenizers`. The only
reference to one is a **dead** config field, `ttt/config.py:144`:

    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"

Grepping every `.py` and `.yaml` in the vendor tree returns that single line.
It is never read, and its Llama-**2** default is stale relative to the actual
data. So TEAM_PLAN P1-4 / task-allocation T2.1 ("use the vendor tokenizer at the
pinned SHA") has no referent, and a mismatch here would silently break length
matching and invalidate every later comparison -- which is exactly what that
task was warning about.

The real tokenizer is **Llama-3**, applied offline before the data ever reaches
the vendor. The authoritative evidence is the model configs, not the dead field:
`configs/model/{125m,350m,760m,1b,3b}.yaml:5-7` are identical and declare
`vocab_size: 128256`, `bos_token_id: 128000`, `eos_token_id: 128001` -- Llama-3
exactly (128000 base plus 256 reserved specials). The vendor's own base defaults
(`config.py:78,88,89`) are the Llama-2 values and are overridden by every model
yaml.

What follows from that
----------------------
`/train` and `/val` are single flat 1-D token-ID streams, so crafted streams are
built in **ID space** and never need a tokenizer: real text and exact
length matching come for free. A tokenizer becomes necessary only to *decode*
for fluency scoring against an independent reference model (T3.2), which is
deliberately out of this wave. See `docs/adr/ADR-007-select-span-granularity.md`.
"""

from __future__ import annotations

#: `configs/model/1b.yaml:6`. Also 125m/350m/760m/3b -- identical in all five.
BOS_TOKEN_ID = 128000

#: `configs/model/1b.yaml:7`. Threaded through every dataloader signature
#: (`lm_dataset.py:47,64,113`) and **never used** -- the zarr is a pre-packed
#: stream with no EOS insertion and no document packing in this repo.
EOS_TOKEN_ID = 128001

#: `configs/model/1b.yaml:5`. Rules out `uint16` storage for the token stream.
VOCAB_SIZE = 128256

#: Vendor commit these were read from. Same pin as `vendor_patch.PINNED_VENDOR_SHA`.
PINNED_VENDOR_SHA = "a4fc4788ace38e29b5067916d4f4be33da894085"
