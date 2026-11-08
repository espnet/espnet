import { defineClientConfig } from "vuepress/client";
import HomeCapabilities from "./components/HomeCapabilities.vue";
import HomeContribute from "./components/HomeContribute.vue";
import HomeDocLinks from "./components/HomeDocLinks.vue";
import HomeHero from "./components/HomeHero.vue";
import HomePipeline from "./components/HomePipeline.vue";
import HomeRecipe from "./components/HomeRecipe.vue";
import HomeQuickStart from "./components/HomeQuickStart.vue";
import DocCard from "./components/DocCard.vue";
import DocCards from "./components/DocCards.vue";
import FileStageMappingWidget from "./components/FileStageMappingWidget.vue";
import DataloaderDemo from "./components/DataloaderDemo.vue";
import MetricsExplorer from "./components/MetricsExplorer.vue";
import SectionConfigBuilder from "./components/SectionConfigBuilder.vue";
import SectionExecutionModes from "./components/SectionExecutionModes.vue";
import SectionOverview from "./components/SectionOverview.vue";
import SectionProvider from "./components/SectionProvider.vue";
import SectionShardLifecycle from "./components/SectionShardLifecycle.vue";
import XVectorAnimation from "./components/XVectorAnimation.vue";
import XVectorConfig from "./components/XVectorConfig.vue";
import DemoConfigEditor from "./components/DemoConfigEditor.vue";
import ShardingDemo from "./components/ShardingDemo.vue";

export default defineClientConfig({
  enhance({ app }) {
    app.component("HomeCapabilities", HomeCapabilities);
    app.component("HomeContribute", HomeContribute);
    app.component("HomeDocLinks", HomeDocLinks);
    app.component("HomeHero", HomeHero);
    app.component("HomePipeline", HomePipeline);
    app.component("HomeRecipe", HomeRecipe);
    app.component("HomeQuickStart", HomeQuickStart);
    app.component("DocCard", DocCard);
    app.component("DocCards", DocCards);
    app.component("FileStageMappingWidget", FileStageMappingWidget);
    app.component("DataloaderDemo", DataloaderDemo);
    app.component("MetricsExplorer", MetricsExplorer);
    app.component("DemoConfigEditor", DemoConfigEditor);
    app.component("SectionConfigBuilder", SectionConfigBuilder);
    app.component("SectionExecutionModes", SectionExecutionModes);
    app.component("SectionOverview", SectionOverview);
    app.component("SectionProvider", SectionProvider);
    app.component("SectionShardLifecycle", SectionShardLifecycle);
    app.component("XVectorAnimation", XVectorAnimation);
    app.component("XVectorConfig", XVectorConfig);
    app.component("ShardingDemo", ShardingDemo);
  },
});
