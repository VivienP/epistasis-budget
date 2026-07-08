# docs/specs/

Two-level spec structure:

- **`docs/SPEC.md`** — the frozen overall design and public API (the source of truth for *what* to build).
- **`docs/specs/{feature}.md`** — per-feature implementation specs, produced by `@architect` (via `/spec
  {feature}`) before a non-trivial module is written. Each references the relevant `docs/SPEC.md` section
  and adds the detail the implementer needs (edge cases, test plan, the scientific contract, the
  validation/null-result plan).

`@architect` creates files here on demand; the directory starts empty. `@reviewer` matches acceptance
criteria against the feature spec if one exists, else against `docs/SPEC.md`. These per-feature specs are
planning documents — the `no-ai-narration` rule exempts them (they legitimately describe a planned change
at a point in time).
