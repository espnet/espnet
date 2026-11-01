import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";
import { path } from "vuepress/utils";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/espnet/",

  lang: "en-US",
  description: "A documentation for ESPnet",

  plugins: [
    {
      name: "local-normal-page",
      alias: {
        // rc.109 renders a normal page through components/base/PageContent
        // (there is no NormalPage any more); swap in our two-column version.
        "@theme-hope/components/base/PageContent": path.resolve(
          __dirname,
          "./theme/components/NormalPage.vue",
        ),
      },
    },
  ],

  head: [
    [
      "script",
      {
        async: true,
        src: "https://www.googletagmanager.com/gtag/js?id=G-XTTNWSWN9Z",
      },
    ],
    [
      "script",
      {},
      `
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-XTTNWSWN9Z');
      `,
    ],
  ],

  bundler: viteBundler({
    viteOptions: {
      build: {
        sourcemap: false,
        rollupOptions: {
          output: {
            manualChunks() {
              return 'vendor';
            },
          },
        },
      },
    },
    vuePluginOptions: {},
  }),

  theme,

  // Enable it with pwa
  // shouldPrefetch: false,
});
