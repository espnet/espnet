# Writing Docs

This page explains how documentation works in the current ESPnet3 VuePress
site.

It is meant for contributors who want to:

- write or edit Markdown pages
- use local Vue components in docs
- add or adjust custom styles
- understand how the navbar, sidebar, and docs rail are wired

::: important
For most documentation work, you only need Markdown.
Use Vue components only when plain Markdown is not enough.
:::

## The basic structure

The documentation content lives here:

```text
doc/
```

The ESPnet3 docs pages live here:

```text
doc/espnet3/
```

The VuePress app-level config and custom UI live here:

```text
doc/vuepress/src/.vuepress/
```

Common places:

- `doc/vuepress/src/.vuepress/theme.ts`: VuePress Hope theme config
- `doc/vuepress/src/.vuepress/config.ts`: VuePress config and Vite aliasing
- `doc/vuepress/src/.vuepress/client.ts`: global component registration
- `doc/vuepress/src/.vuepress/components/`: reusable Vue components for docs
- `doc/vuepress/src/.vuepress/styles/`: custom SCSS
- `doc/vuepress/src/.vuepress/theme/components/`: theme-level overrides


## Writing a normal page

Most pages should just be Markdown under `doc/`.

Examples:

- `espnet3/get-started/what-is-a-recipe.md`
- `espnet3/config/index.md`
- `espnet3/contributing/ci-and-pr.md`

Prefer:

- short sections
- short English sentences
- cards for related pages
- callouts such as `important`, `note`, or `warning` when useful

VuePress Hope hint containers are enabled, so you can write:

```md
::: important
This page assumes you already cloned a recipe.
:::
```

## Internal links

Link to other pages with a **relative path ending in `.html`**, never `.md`
and never a bare directory:

```md
[Config overview](../config/index.html)
```

VuePress builds each Markdown source file to an `.html` route, so a link
written as `.md` or without an extension will 404 on the built site even
though it resolves while browsing the raw Markdown. Only link to pages that
actually exist under `doc/espnet3/` (check with
`find doc/espnet3 -name '*.md'` if unsure). `DocCard`/`DocCards` `href`
values follow the same rule (see the examples throughout this page).

## Using Vue components in Markdown

This site enables Vue components inside Markdown through VuePress Hope
`mdEnhance.component`.

That means you can write components directly in a page, for example:

```md
<DocCards :cols="2">
  <DocCard title="Config overview" href="../config/index.html" />
</DocCards>
```

Global components are registered in `client.ts`.

Current examples include:

- `DocCard`
- `DocCards`
- `HomeHero`
- `HomeContribute`
- `FileStageMappingWidget`

::: note
If a component is not registered in `client.ts`, it will not be available in
Markdown pages.
:::

## When to add a Vue component

Add a Vue component only when the page needs something that Markdown handles
poorly.

Good reasons:

- a reusable card layout
- a custom widget
- an interactive homepage section
- repeated UI that appears across multiple pages

Bad reasons:

- a simple paragraph block
- a normal table
- a one-off layout tweak that SCSS can handle

Put reusable components here:

```text
doc/vuepress/src/.vuepress/components/
```

## Theme-level page customization

This site uses VuePress Hope with a custom replacement for the normal docs
page.

Relevant files:

- `doc/vuepress/src/.vuepress/config.ts`
- `doc/vuepress/src/.vuepress/theme/components/NormalPage.vue`

`config.ts` aliases the Hope `NormalPage` component to the local wrapper.
That wrapper currently adds the docs rail on normal pages.

The docs rail is built from:

- `doc/vuepress/src/.vuepress/components/DocsFeedback.vue`
- `doc/vuepress/src/.vuepress/components/DocsAiSidebar.vue`

Today it contains:

- the feedback block
- the AI tools links such as `Ask ChatGPT`

This is also the likely place to extend later for:

- future docs-side actions
- extra page tools

## Where styles live

Custom styles are split by purpose.

Main files:

- `doc/vuepress/src/.vuepress/styles/index.scss`
- `doc/vuepress/src/.vuepress/styles/home-custom.scss`
- `doc/vuepress/src/.vuepress/styles/docs-tools.scss`

Use them like this:

- `index.scss`: global content styles
- `home-custom.scss`: homepage section styles
- `docs-tools.scss`: right rail / docs tool styles

## Static data files

If you need an external static file, put it under:

`doc/vuepress/src/.vuepress/public/data/`

Files there are published as static assets and can be fetched by URL from the
docs app.

Current usage note:

- `HomeQuickStart.vue` currently imports `doc/vuepress/src/.vuepress/data/systems.json`
- that is a bundled module import, not a file fetched from `public/data/`

Use `.vuepress/data/` for bundled import-time data.
Use `.vuepress/public/data/` for files that should be served directly.

## Current custom content styles

The site already applies custom styling to some Markdown elements.

Current examples:

- `details` / `summary`
- top-level ordered lists
- custom helper classes such as `.custom-h3` and `.custom-h4`

These are defined in:

- `doc/vuepress/src/.vuepress/styles/index.scss`

If you change a global selector there, you are changing the appearance of many
pages at once.

::: warning
Be careful with global Markdown selectors.
A small change in `index.scss` can affect the entire ESPnet docs set.
:::

## Navbar and sidebar

The navbar and sidebar are loaded from YAML files, then wired through VuePress
Hope helpers.

Relevant files:

- `doc/vuepress/navbars.yml`
- `doc/vuepress/sidebars.yml`
- `doc/vuepress/src/.vuepress/navbar.ts`
- `doc/vuepress/src/.vuepress/sidebar.ts`

The TypeScript files simply load the YAML and pass it to VuePress Hope.

In practice:

- add or reorder top navigation in `navbars.yml`
- add or reorder section sidebar entries in `sidebars.yml`

## Homepage components

The ESPnet3 landing page uses custom Vue components rather than plain Markdown.

Examples:

- `HomeHero.vue`
- `HomeCapabilities.vue`
- `HomeDocLinks.vue`
- `HomeRecipe.vue`
- `HomeContribute.vue`

These are registered globally in `client.ts` and then used inside the homepage
Markdown.

If you are editing homepage sections, check both:

- the component file
- `home-custom.scss`

## A practical workflow

For most doc work:

1. Edit the Markdown page first.
2. Use `DocCards` or existing components if needed.
3. Add a new component only if the UI is reusable or interactive.
4. Put page-specific layout in the component, not in random global CSS.
5. Put truly global Markdown styling in `styles/index.scss`.
6. Update `sidebars.yml` or `navbars.yml` if the page needs navigation.

## Local validation for docs work

If you changed VuePress files or docs pages, treat docs validation as part of
your local check set before you push.

### How doc generation works

Running `ci/doc.sh` generates the Markdown pages that VuePress builds from.
The output lands in `doc/vuepress/src/`:

```bash
bash ci/doc.sh
```

::: warning
Every time `ci/doc.sh` runs, **all Markdown files under `doc/vuepress/src/`
are deleted and regenerated**.
Do not save work-in-progress edits there — they will be lost on the next run.
The source of truth is `doc/`.
:::

::: tip Skipping notebook rendering
Notebook rendering in `ci/doc.sh` takes a very long time.
If you do not need notebooks for your current work,
comment them out before running the script to run ci/doc.sh faster:

```bash
# incorporate espnet/notebook repository to docs
# echo "::group::incorporate notebook to docs"
# ./doc/notebook2rst.sh
# echo "::endgroup::"
```

and further down:

```bash
# cp -r ./doc/notebook ./doc/vuepress/src/
# rm -rf ./doc/vuepress/src/notebook/ESPnetEZ
```
:::

### Dev server

Once `doc/vuepress/src/` is populated, start the local preview server:

```bash
cd doc/vuepress
./dev.sh
```

This starts the local debug server with file polling enabled.
It polls every 5 seconds, so changes are picked up even in WSL and other
environments where normal file watching is unreliable.

### Workflow for homepage or VuePress UI work

1. Run `ci/doc.sh` to populate `doc/vuepress/src/` with the generated Markdown.
2. Start `dev.sh` and open the local preview in a browser.
3. Edit the Vue components or SCSS under `doc/vuepress/src/.vuepress/` and
   watch the server pick up the changes.
4. When satisfied, copy any new or edited **Markdown content** back to `doc/`
   before running `ci/doc.sh` again — it will wipe `doc/vuepress/src/`.

### What to check before pushing

At minimum, confirm:

- the docs build still works
- links and frontmatter are valid
- custom components still render
- layout did not break on the edited page

Recipe-only documentation changes usually do not need new tests.

When docs work is part of a larger code change, include the docs update in the
same local review set before you push.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Docstring guide"
    desc="Write code documentation in the expected repository style."
    icon="tabler:file-text"
    href="./docstring-guide.html"
  />
  <DocCard
    title="CI and PR"
    desc="See what to check before opening a docs PR."
    icon="tabler:git-pull-request"
    href="./ci-and-pr.html"
  />
  <DocCard
    title="Naming conventions"
    desc="Keep names consistent across code, config, docs, and imports."
    icon="tabler:abc"
    href="./naming-conventions.html"
  />
</DocCards>
