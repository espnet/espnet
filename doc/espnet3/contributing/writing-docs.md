# Writing Docs

Edit Markdown sources under `doc/`. ESPnet3 guides live under `doc/espnet3/`;
VuePress configuration, components and styles live under
`doc/vuepress/src/.vuepress/`.

## Sources and generated pages

`ci/doc.sh` generates API/tool pages, renders notebooks and copies content to
`doc/vuepress/src/` before building the site. It installs the documentation extra,
uses the tools environment and can download Kaldi/notebooks and npm dependencies.
It is an end-to-end build, not a lightweight Markdown check.

The script deletes and regenerates non-hidden content in `doc/vuepress/src/`.
Keep source edits under `doc/`, and keep tracked VuePress implementation under
`.vuepress/`. Do not save work only in generated Markdown.

The current copy step handles top-level `doc/*.md`, API pages, notebooks, recipe
pages and images. It does not automatically copy arbitrary new subdirectories
such as `doc/espnet3/`. For a local ESPnet3 preview, explicitly stage that source
tree before building, and verify the final site's routes. A documentation PR
adding a new section must also arrange for that section to reach the deployed
build; source files alone do not guarantee publication.

## Write pages with concrete examples

Use short English explanations and examples verified against the implementation.
Check imports, config keys, stage names, field names and output paths. Keep
ESPnet2 shell conventions separate from ESPnet3's `launch`/`StageSpec` interface.

A normal page should use Markdown first. Registered Vue components can add
related-page cards:

```md
<DocCards :cols="2">
  <DocCard title="Adding a stage" href="./adding-a-stage.html" />
  <DocCard title="CI and PR" href="./ci-and-pr.html" />
</DocCards>
```

The local `DocCard`/`DocCards` components are registered in
`.vuepress/client.ts`. A component must exist and be registered before a page can
use it. Do not document homepage widgets or a docs rail unless those components
actually ship in the checkout.

For example, `DocCard`'s `href` is used directly by its rendered anchor. Use a
relative `.html` route and verify it in the built site:

```md
[Adding a stage](./adding-a-stage.html)
```

A `.md` source link may be transformed by Markdown tooling, but an arbitrary
component's string prop is not necessarily transformed the same way. A working
link in the GitHub source view is not proof that the deployed route works.

## Navigation, components and styles

| Responsibility | Source |
| --- | --- |
| VuePress and bundler settings | `doc/vuepress/src/.vuepress/config.ts` |
| Theme and Markdown features | `doc/vuepress/src/.vuepress/theme.ts` |
| Global component registration | `doc/vuepress/src/.vuepress/client.ts` |
| Reusable components | `doc/vuepress/src/.vuepress/components/` |
| Site SCSS | `doc/vuepress/src/.vuepress/styles/` |
| Navigation generation | `doc/vuepress/create_menu.py` |

`create_menu.py --root ./src` writes `navbars.yml` and `sidebars.yml` in
`doc/vuepress/`. The TypeScript navbar/sidebar modules read those generated YAML
files. Editing generated YAML alone is temporary; update the generator when a
navigation change must survive the next build.

Keep page-specific layout in its component. Changes to global styles affect all
pages. Inspect actual imports before referring to static JSON, public assets or
custom theme overrides; there is no requirement to invent those directories.

## Local checks and preview

Check Python/YAML snippets, local targets and page addresses before a full build.
Where available, `python ci/check_doc_pages.py` detects duplicate page addresses.
For an end-to-end build in an appropriately configured disposable checkout:

```bash
bash ci/doc.sh
```

For a focused ESPnet3 preview without notebook/API generation:

```bash
mkdir -p doc/vuepress/src/espnet3
cp -R doc/espnet3/. doc/vuepress/src/espnet3/
cd doc/vuepress
python create_menu.py --root ./src
npm ci
npm run docs:build
npm run docs:dev
```

The package scripts are declared in `doc/vuepress/package.json`. There is no
`dev.sh` helper or documented automatic five-second polling wrapper in this
checkout. If your filesystem needs polling, configure the watcher explicitly
rather than assuming it is already enabled.

The focused preview does not generate the whole site's API, notebook, recipe or
legacy pages. Missing routes from those sections need the complete build input;
verify the documentation PRs together when their links depend on sibling sections.
Do not edit `ci/doc.sh` to silently skip notebook rendering and then report that
the complete CI build passed.

Before publishing, check page rendering, internal routes, code examples and
navigation on the generated site. Documentation-only changes normally do not
need model-training tests.

<DocCards :cols="2">
  <DocCard title="Docstring guide" href="./docstring-guide.html" />
  <DocCard title="CI and PR" href="./ci-and-pr.html" />
</DocCards>
