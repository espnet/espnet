import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/espnet_draft_home_page/",

  lang: "en-US",
  description: "A documentation for ESPnet",

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
