# Agent working files

`doc/` and `docs/` are for website documentation. Do not place agent-generated
notes, plans, skill artifacts, or other working files there unless the user
explicitly requests them as documentation. Use a temporary directory for scratch
files; keep shared agent guidance under `.agents/`.

# Review guidelines

Before reviewing a pull request, read `CONTRIBUTING.md` and apply the guidelines
relevant to the changed files. Keep that document as the source of contribution
rules rather than duplicating them here. For implementation work in `espnet3/`,
`egs3/`, or CI, also read the [coding-agent contributor guide](.agents/AGENTS.md)
and the linked package-specific guidance; this file covers review guidance only.

- Check correctness, backward compatibility, recipe conventions, and tests.
- Distinguish ESPnet2 rules from ESPnet3 rules and respect their documented scope.
- Verify documentation against the implementation and configuration it describes.
- Report concrete, actionable findings with file and line references. Explain the
  affected behavior and the conditions that trigger the issue.
- Avoid formatting-only findings already handled by automated checks.
