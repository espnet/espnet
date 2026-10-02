import { defineClientConfig } from "vuepress/client";

import DocCard from "./components/DocCard.vue";
import DocCards from "./components/DocCards.vue";
import HomeCapabilities from "./components/HomeCapabilities.vue";
import HomeContribute from "./components/HomeContribute.vue";
import HomeDocLinks from "./components/HomeDocLinks.vue";
import HomeHero from "./components/HomeHero.vue";
import HomePipeline from "./components/HomePipeline.vue";
import HomeQuickStart from "./components/HomeQuickStart.vue";
import HomeRecipe from "./components/HomeRecipe.vue";
import MetricsExplorer from "./components/MetricsExplorer.vue";
import XVectorAnimation from "./components/XVectorAnimation.vue";
import XVectorConfig from "./components/XVectorConfig.vue";

export default defineClientConfig({
  enhance({ app }) {
    app.component("DocCard", DocCard);
    app.component("DocCards", DocCards);
    app.component("HomeCapabilities", HomeCapabilities);
    app.component("HomeContribute", HomeContribute);
    app.component("HomeDocLinks", HomeDocLinks);
    app.component("HomeHero", HomeHero);
    app.component("HomePipeline", HomePipeline);
    app.component("HomeQuickStart", HomeQuickStart);
    app.component("HomeRecipe", HomeRecipe);
    app.component("MetricsExplorer", MetricsExplorer);
    app.component("XVectorAnimation", XVectorAnimation);
    app.component("XVectorConfig", XVectorConfig);
  },
});
