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
            :darkmode="isDarkmode"
          />
        </div>

        <div class="docs-page-shell__sidebar">
          <DocsFeedback />
          <DocsAiSidebar />
          <div v-if="tocEnable && hasTocItems" class="docs-page-shell__toc">
            <div class="docs-page-shell__toc-title">On this page</div>
            <TOC :header-depth="headerDepth" />
          </div>
        </div>
      </div>
    </component>
  </main>
</template>
