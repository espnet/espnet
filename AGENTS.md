# Review guidelines

Before reviewing a pull request, read `CONTRIBUTING.md` and apply the guidelines
relevant to the changed files. Keep that document as the source of contribution
rules rather than duplicating them here. For implementation work in `espnet3/`,
`egs3/`, or CI, also read the [coding-agent contributor guide](.agents/AGENT.md)
and the linked package-specific guidance; this file covers review guidance only.

- Check correctness, backward compatibility, recipe conventions, and tests.
- Distinguish ESPnet2 rules from ESPnet3 rules and respect their documented scope.
- Verify documentation against the implementation and configuration it describes.
- Report concrete, actionable findings with file and line references. Explain the
  affected behavior and the conditions that trigger the issue.
- Avoid formatting-only findings already handled by automated checks.
