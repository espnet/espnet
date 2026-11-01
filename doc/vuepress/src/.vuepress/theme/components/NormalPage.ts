import { RenderDefault, hasGlobalComponent } from "@vuepress/helper/client";
import { defineComponent } from "vue";
import { withBase } from "vuepress/client";

// Mirrors vuepress-theme-hope rc.109's components/base/PageContent, which
// config.ts aliases to this file, with a two-column shell added around it.
import BreadCrumb from "@theme-hope/components/base/BreadCrumb";
import MarkdownContent from "@theme-hope/components/base/MarkdownContent";
import PageNav from "@theme-hope/components/base/PageNav";
import PageTitle from "@theme-hope/components/base/PageTitle";
import TOC from "@theme-hope/components/base/TOC";
import PageMeta from "@theme-hope/components/info/PageMeta";
import { useDarkMode } from "@theme-hope/composables/useDarkMode";
import { useData } from "@theme-hope/composables/useData";

import DocsAiSidebar from "../../components/DocsAiSidebar.vue";
import DocsFeedback from "../../components/DocsFeedback.vue";

export default defineComponent({
  name: "PageContent",
  components: {
    BreadCrumb,
    MarkdownContent,
    PageNav,
    PageTitle,
    PageMeta,
    TOC,
    DocsAiSidebar,
    DocsFeedback,
  },
  setup() {
    const { frontmatter } = useData();
    const isDarkMode = useDarkMode();

    return {
      RenderDefault,
      frontmatter,
      hasGlobalComponent,
      isDarkMode,
      withBase,
    };
  },
});
