# Public result artifacts

This directory contains small JSON results and provenance, never model weights, datasets, or scoring
caches. `manifest.json` records the source run, command, base commit, dirty working-tree state,
deterministic code-diff digest, configuration, and SHA-256 of every listed artifact.

These checksums are taken over LF bytes; `.gitattributes` pins `artifacts/**/*.json` to `eol=lf` so a
Windows checkout is not silently rewritten to CRLF, which would fail the checksum.

For entries classified `traceable_not_rerun`, `generation_command` is the deterministic reproduction
command reconstructed from the recorded configuration; the original shell invocation was not embedded
in the source JSON and is not claimed to have been captured verbatim.

The current files are **provisional** because they were assembled before a final review commit. Their
numerical payloads are copied unchanged from the audited local `report/` files and classified as
`traceable_not_rerun` unless a local deterministic check reproduced them. After the final code commit,
the required empirical runs must be repeated where specified and the manifest regenerated with the new
commit SHA and clean code state.

Run `python scripts/validate_artifacts.py` to verify schemas, checksums, public-document claim mappings,
and banned historical values. `report/` remains ignored and is the location for transient or large
outputs.

`structural_allocation_650m.json` is the compact, provisional summary of the GB1 and TrpB downstream
gates and the TrpB map-recovery decision. It records hashes for the large source reports instead of
copying their raw fold and partition records into the repository.
