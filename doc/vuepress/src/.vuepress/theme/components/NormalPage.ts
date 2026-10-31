import { hasGlobalComponent } from "@vuepress/helper/client";
import { computed, defineComponent } from "vue";
import { usePageData, usePageFrontmatter, withBase } from "vuepress/client";
import { RenderDefault } from "vuepress-shared/client";

import BreadCrumb from "@theme-hope/components/BreadCrumb";
import MarkdownContent from "@theme-hope/components/MarkdownContent";
import PageNav from "@theme-hope/components/PageNav";
import PageTitle from "@theme-hope/components/PageTitle";
import { useThemeLocaleData } from "@theme-hope/composables/index";
import PageMeta from "@theme-hope/modules/info/components/PageMeta";
import TOC from "@theme-hope/modules/info/components/TOC";
import { useDarkmode } from "@theme-hope/modules/outlook/composables/index";

import DocsAiSidebar from "../../components/DocsAiSidebar.vue";
import DocsFeedback from "../../components/DocsFeedback.vue";

export default defineComponent({
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
    const page = usePageData();
    const frontmatter = usePageFrontmatter();
    const { isDarkmode } = useDarkmode();
    const themeLocale = useThemeLocaleData();

    const tocEnable = computed(() => frontmatter.value.toc ?? true);
    const headerDepth = computed(
      () => frontmatter.value.headerDepth ?? themeLocale.value.headerDepth ?? 2,
    );
    const hasTocItems = computed(() => (page.value.headers ?? []).length > 0);

    return {
      RenderDefault,
      frontmatter,
      hasGlobalComponent,
      hasTocItems,
      headerDepth,
      isDarkmode,
      tocEnable,
      withBase,
    };
  },
});
