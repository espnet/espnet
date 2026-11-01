<script lang="ts" src="./NormalPage.ts"></script>

<template>
  <main id="main-content" class="vp-page">
    <component :is="hasGlobalComponent('LocalEncrypt') ? 'LocalEncrypt' : RenderDefault">
      <div class="docs-page-shell">
        <div class="docs-page-shell__main">
          <div v-if="frontmatter.cover" class="page-cover">
            <img :src="withBase(frontmatter.cover)" alt="" no-view />
          </div>

          <BreadCrumb />
          <PageTitle />
          <MarkdownContent />
          <PageMeta />
          <PageNav />

          <component
            :is="'CommentService'"
            v-if="hasGlobalComponent('CommentService')"
            :darkmode="isDarkMode"
          />
        </div>

        <div class="docs-page-shell__sidebar">
          <DocsFeedback />
          <DocsAiSidebar />
          <!-- rc.109's TOC collects headers from the rendered DOM on the client
               (useHeaders -> onContentUpdated), so nothing is known at SSR time;
               always mount the wrapper and let :empty hide it in CSS. -->
          <div class="docs-page-shell__toc">
            <TOC />
          </div>
        </div>
      </div>
    </component>
  </main>
</template>
