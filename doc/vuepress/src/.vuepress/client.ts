import { defineClientConfig } from "vuepress/client";

import DocCard from "./components/DocCard.vue";
import DocCards from "./components/DocCards.vue";
import HomeQuickStart from "./components/HomeQuickStart.vue";

export default defineClientConfig({
  enhance({ app }) {
    app.component("DocCard", DocCard);
    app.component("DocCards", DocCards);
    app.component("HomeQuickStart", HomeQuickStart);
  },
});
