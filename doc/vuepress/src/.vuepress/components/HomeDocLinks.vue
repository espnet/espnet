<template>
  <section class="home-doclinks">
    <div class="home-section-inner">
      <div class="dl-grid">
        <div v-for="col in columns" :key="col.heading" class="dl-col">
          <div class="dl-heading">{{ col.heading }}</div>
          <ul class="dl-list">
            <li v-for="link in col.links" :key="link.label">
              <a :href="link.href" class="dl-link">{{ link.label }}</a>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { withBase } from 'vuepress/client'

const columns = [
  {
    heading: 'Docs',
    links: [
      { label: 'Core Reference',    href: './core/index.html' },
      { label: 'Config Reference',  href: './config/index.html' },
      { label: 'System and Stages', href: './stages/index.html' },
      { label: 'Components',        href: './core/components/index.html' },
      { label: 'Parallel & Distributed', href: './core/parallel/index.html' },
      { label: 'Generated Python API', href: './api-reference.html' },
    ],
  },
  {
    heading: 'Tutorials',
    links: [
      { label: 'Quick Start',              href: './get-started/index.html' },
      { label: 'What a Recipe Is',         href: './get-started/what-is-a-recipe.html' },
      { label: 'Guides hub',               href: './guides/index.html' },
      { label: 'Finetuning',               href: './guides/finetuning/index.html' },
      { label: 'Migrating from ESPnet2',   href: './guides/migrating-from-espnet2/index.html' },
      { label: 'Coming from PyTorch',      href: './guides/coming-from-pytorch/index.html' },
      { label: 'Coming from other tools',  href: './guides/coming-from-other-toolkits/index.html' },
    ],
  },
  {
    heading: 'Resources',
    links: [
      { label: 'GitHub',           href: 'https://github.com/espnet/espnet' },
      { label: 'Citation',         href: withBase('/citations.html') },
      { label: 'Contributing',     href: './contributing/index.html' },
      { label: 'Troubleshooting',  href: './get-started/installation-troubleshooting.html' },
    ],
  },
]
</script>

<style scoped>
.dl-citation {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid #e0e0e0;
}

.dl-citation-desc {
  font-size: 13px;
  color: var(--vp-c-text-2);
  margin: 8px 0 10px;
}

.dl-bibtex {
  background: var(--vp-c-bg-soft);
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 14px 16px;
  font-size: 12px;
  line-height: 1.6;
  overflow-x: auto;
  margin: 0 0 10px;
}

.dl-bibtex code {
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
  background: none;
  padding: 0;
}

.dl-citation-more {
  font-size: 12px;
  color: var(--vp-c-brand-2, #3ecf8e);
  text-decoration: none;
}

.dl-citation-more:hover {
  text-decoration: underline;
}

</style>
