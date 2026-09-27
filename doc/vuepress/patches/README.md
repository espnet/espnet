# VuePress SSR callback patch

`@vuepress+client+2.0.0-rc.31.patch` prevents `onContentUpdated` from
registering DOM-update callbacks during server-side rendering. VuePress keeps
these callbacks in a module-level set and removes them on unmount, but Vue does
not run unmount hooks during SSR. Callbacks used by the theme can therefore
retain previously rendered pages throughout the documentation build.

The patch adds an SSR guard; browser registration and cleanup are unchanged.
It applies to the published JavaScript in the pinned client package. See
[ESPnet #6644](https://github.com/espnet/espnet/issues/6644).

`npm ci` and `npm install` apply the patch through the `postinstall` script.
Patch application errors fail installation. Do not use `--ignore-scripts` when
building the documentation.

This is a temporary local fix, not a released upstream fix. When upgrading
VuePress, check whether `onContentUpdated` skips SSR registration. Once that fix
is included upstream, remove this patch, the `postinstall` script, and the
`patch-package` dependency, then regenerate the lockfile and run the full
documentation build. If the new release still needs a patch, review it against
that release instead of just renaming this patch file.
