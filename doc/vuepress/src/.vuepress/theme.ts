import { hopeTheme } from "vuepress-theme-hope";

import navbar from "./navbar.js"
import sidebar from "./sidebar.js"

export default hopeTheme({

  logo: "/assets/image/espnet_logo1.png",

  favicon: "/assets/image/espnet.png",

  repo: "espnet/espnet",

  docsDir: "src",

  darkmode: "disable",

  // navbar
  navbar,

  // sidebar
  sidebar,

  footer: "Copyright © 2024 ESPnet Community. All rights reserved.",

  displayFooter: true,

  toc: false,

  editLink: false,

  // All features are enabled for demo, only preserve features you need here.
  // Was `plugins.mdEnhance` before theme-hope rc.109, which types that key
  // as `never` ("@deprecated Use `markdown` instead").
  markdown: {
    align: true,
    attrs: true,
    codeTabs: true,
    component: true,
    demo: true,
    figure: true,
    hint: true,
    imgLazyload: true,
    imgSize: true,
    include: true,
    mark: true,
    plantuml: true,
    spoiler: true,
    stylize: [
      {
        matcher: "Recommended",
        replacer: ({ tag }) => {
          if (tag === "em")
            return {
              tag: "Badge",
              attrs: { type: "tip" },
              content: "Recommended",
            };
        },
      },
    ],
    sub: true,
    sup: true,
    tabs: true,
    tasklist: true,
    vPre: true,

    // install chart.js before enabling it
    // chart: true,

    // insert component easily

    // install echarts before enabling it
    // echarts: true,

    // install flowchart.ts before enabling it
    // flowchart: true,

    // gfm requires mathjax-full to provide tex support
    // gfm: true,

    // install katex before enabling it
    // katex: true,

    // install mathjax-full before enabling it
    // mathjax: true,

    // install mermaid before enabling it
    // mermaid: true,

    // playground: {
    //   presets: ["ts", "vue"],
    // },

    // install reveal.js before enabling it
    // revealJs: {
    //   plugins: ["highlight", "math", "search", "notes", "zoom"],
    // },

    // install @vue/repl before enabling it
    // vuePlayground: true,

    // install sandpack-vue3 before enabling it
    // sandpack: true,
  },

  plugins: {

    // iconAssets moved here in theme-hope rc.109
    // ("@deprecated Use `plugins.icon.assets` instead").
    icon: {
      assets: "iconify",
    },


    // Successor to vuepress-plugin-search-pro, which theme-hope now types as
    // `never`.  The theme takes the options object directly, so no import is
    // needed.  `placeholder` is dropped: slimsearch's en locale already
    // defaults it to "Search".  `autoSuggestions` is spelled `suggestion` here.
    slimsearch: {
      indexContent: false,
      suggestion: false,
    },

  },
});
