# CI and PR

Good pull requests are easy to review, easy to test, and easy to merge.

This page explains what we expect before you open a PR for ESPnet3.

::: important
Please optimize for reviewer time.
Small PRs with local checks and tests are much easier to merge.
:::

## Keep PRs reviewable

Please think about the reviewer when you prepare a PR.

As a rough guide, try to keep one PR within:

- about 20 changed files
- about 2000 changed lines

That is usually enough for a bug fix, a feature addition, or a recipe change.

If your change is larger than that, split it by feature or by layer.
Small PRs move much faster than one large PR.

::: warning
Do not bundle unrelated refactors and new features into one PR unless they must
land together.
:::

## Run local checks first

Before opening a PR, run the smallest local checks that cover your change.

For formatting and style checks, run:

```bash
black espnet3/ test/espnet3/ ci/
isort espnet3/ test/espnet3/ ci/
pycodestyle espnet3/ test/espnet3/ ci/
bash ci/test_flake8.sh espnet3
pytest -q test/espnet3/
```

`ci/test_flake8.sh` takes exactly one directory argument (the strict,
docstring-checked target); `ci/test_python_espnet3.sh` calls it as
`test_flake8.sh espnet3`. Some formatting is fixed automatically in CI, but
not everything is. In practice, many CI failures come from style checks,
shell checks, or docstring formatting. Running these locally first saves
review time.

## Test what you change

If you edit `espnet3/`, **add or update unit tests.**

Coverage matters. A change without enough test coverage may fail CI even when
the implementation itself looks correct.

If you add a new model, a new system, or a large recipe-level feature, also add
integration tests when appropriate.

As a general rule:

- code changes in `espnet3/` should have unit tests
- new end-to-end behavior should have integration coverage

## Check the full user flow

For ESPnet3 features, please confirm that the full workflow still works before
you open the PR.

That includes:

- training
- inference
- metrics
- publication
- demo, if your change affects it

If you can share a working Hugging Face Space or another live demo link, that
is very helpful for review.

## Keep tests organized

Please keep the structure under `test/` as close as possible to the structure of
the package code.

If you add:

```text
espnet3/foo/bar.py
```

prefer a matching location such as:

```text
test/espnet3/foo/test_bar.py
```

Use similar directory names and file names whenever possible.

## Request the right reviewers

For ESPnet3-related PRs, please include:

- `@sw005320`
- `@Masao-Someki`

as reviewers.

## Reduce unnecessary CI runs

CI is valuable, but it is still a shared resource.

Please avoid pushing many trial-and-error commits that only test formatting or
basic lint fixes. Running lint fixes or `pytest` locally is usually faster.

If you want to simulate GitHub Actions locally, you can also use `act` for some
workflows.

## Common CI failures

Many CI failures are not deep model bugs.
The common ones are:

- formatting issues
- docstring style issues
- shell formatting or shellcheck failures
- missing tests

Most of these can be caught locally before you push.

## Before you open the PR

Use this as a final checklist:

1. The PR is small enough to review.
2. Local format and lint checks pass.
3. Relevant unit tests pass.
4. Integration coverage is added when needed.
5. The full ESPnet3 workflow was checked when relevant.
6. The right reviewers were requested.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Naming conventions"
    desc="Use stable names for files, configs, stage names, and imports."
    icon="tabler:abc"
    href="./naming-conventions.html"
  />
  <DocCard
    title="Core overview"
    desc="Read the shared architecture docs before changing common ESPnet3 behavior."
    icon="tabler:stack-2"
    href="../core/index.html"
  />
  <DocCard
    title="Writing docs"
    desc="See how docs work when your PR also changes VuePress content."
    icon="tabler:book-upload"
    href="./writing-docs.html"
  />
</DocCards>
