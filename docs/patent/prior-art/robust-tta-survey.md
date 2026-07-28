# Prior art — robust TTA defenses beyond MedBN

**Status: STUB.** Complete before filing. MedBN gets its own dedicated diff
(`medbn-diff.md`); this file catches the rest of the robust-TTA landscape so the
novelty argument is not blindsided by a non-MedBN reference.

To survey and diff:
- Robust / online TTA methods that defend against distribution shift or malicious
  batches by means *other* than batch-norm statistics.
- Entropy-minimisation TTA (TENT and descendants) and their known failure modes.
- Any continual-learning "sanitisation" or "update filtering" work — closest in
  spirit to our gate and therefore the most important to distinguish.

For each: setting, unit defended, signal type, guarantee, whether any operate on
*fast weights* (none known to), and the line-level diff vs. our behavioural
anchor-consistency + bounded-drift + rollback system.
