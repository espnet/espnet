import { hopeTheme } from "vuepress-theme-hope";
import { searchProPlugin } from "vuepress-plugin-search-pro";

import navbar from "./navbar.js"
import sidebar from "./sidebar.js"

export default hopeTheme({

  iconAssets: "fontawesome-with-brands",

  logo: "/assets/image/espnet_logo1.png",

  favicon: "/assets/image/espnet.png",

  repo: "espnet/espnet",

  docsDir: "src",

  // navbar
  navbar,

  // sidebar
  sidebar,

  footer: "Copyright © 2024 ESPnet Community. All rights reserved.",

  displayFooter: true,

  toc: false,
  
  editLink: false,

  plugins: {

    // All features are enabled for demo, only preserve features you need here
    mdEnhance: {
      align: true,
      attrs: true,
      codetabs: true,
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

    searchPro: searchProPlugin({
      placeholder: "Search",
      indexContent: false,
      autoSuggestions: false,
    }),

  },
});
