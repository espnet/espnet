# Contributing

This section is for ESPnet3 contributors and recipe authors.

::: important
Please keep contributions small, reviewable, and documented.
When behavior changes, tests and docs should usually change in the same PR.
:::

## Start here

If you want to work on ESPnet3 code, start with the local setup and PR flow.

<DocCards>
  <DocCard
    title="Dev setup"
    desc="Create a Pixi-based development environment for ESPnet3."
    icon="tabler:tool"
    href="./dev-setup.html"
  />
  <DocCard
    title="CI and PR"
    desc="Run checks locally and prepare changes for review."
    icon="tabler:git-pull-request"
    href="./ci-and-pr.html"
  />
  <DocCard
    title="Core overview"
    desc="Read the main ESPnet3 architecture docs before changing shared behavior."
    icon="tabler:stack-2"
    href="../core/index.html"
  />
</DocCards>

## Extend ESPnet3

Use these pages when you add new building blocks.

<DocCards>
  <DocCard
    title="Adding a stage"
    desc="Add a new stage to an existing system."
    icon="tabler:route"
    href="./adding-a-stage.html"
  />
</DocCards>

## Quality and policy

These pages explain repository expectations.

<DocCards>
  <DocCard
    title="Writing docs"
    desc="Edit Markdown, use Vue components, and understand the current VuePress setup."
    icon="tabler:book-upload"
    href="./writing-docs.html"
  />
  <DocCard
    title="Docstring guide"
    desc="Write public and private docstrings in the expected style."
    icon="tabler:file-text"
    href="./docstring-guide.html"
  />
  <DocCard
    title="Naming conventions"
    desc="Keep names consistent across code, config, docs, and imports."
    icon="tabler:abc"
    href="./naming-conventions.html"
  />
</DocCards>
