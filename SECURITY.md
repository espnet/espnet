# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub private vulnerability reporting](https://github.com/espnet/espnet/security/advisories/new).
Do not open a public issue or pull request for an unfixed vulnerability.

Include the affected file and function, the ESPnet version or commit, and a
minimal reproducer. We will acknowledge the report, and publish an advisory
once a fixed release is on PyPI.

## Supported versions

Fixes are made on `master` and shipped in the next PyPI release. Only the
latest release receives security fixes; please upgrade before reporting.

## Before you report: known and fixed issues

The following issues are already fixed and published. Reports that rediscover
them (including variants in the same code path on a fixed version) will be
closed as duplicates.

| Advisory | Issue | Fixed in |
|---|---|---|
| [GHSA-jcfw-5whq-98j4](https://github.com/espnet/espnet/security/advisories/GHSA-jcfw-5whq-98j4) | Tar-slip path traversal in `espnet2.main_funcs.pack_funcs.unpack` (arbitrary file write when unpacking a model archive) | 202609.post1 |
| [GHSA-64f6-3gqc-r926](https://github.com/espnet/espnet/security/advisories/GHSA-64f6-3gqc-r926) | Code execution via `torch.load(weights_only=False)` when loading pretrained models | 202609 |
| [GHSA-rvjf-ppg9-9rwg](https://github.com/espnet/espnet/security/advisories/GHSA-rvjf-ppg9-9rwg) | Code execution via unsafe `yaml.Loader` in the ParallelWaveGAN vocoder config loader | 202609.post2 |

## Loading untrusted models

Checkpoints are loaded through `espnet2.torch_utils.safe_torch_load`, which
uses `torch.load(weights_only=True)`. It falls back to unpickling arbitrary
objects only when the user opts in, by setting
`ESPNET_ALLOW_UNSAFE_TORCH_LOAD=1` or by confirming on an interactive prompt.
Opting in executes code from the checkpoint; do it only for models you trust.
Behaviour that requires this opt-in is not considered a vulnerability.

Recipe scripts under `egs2/` and `egs3/` download and process public corpora
and are meant to be run by the user on their own data; they are not hardened
against hostile input.
