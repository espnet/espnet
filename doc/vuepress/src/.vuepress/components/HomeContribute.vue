<template>
  <section class="home-contribute">
    <div class="home-section-inner">
      <p class="home-label">Community</p>
      <h2 class="home-section-title">Contribute to ESPnet3</h2>
      <p class="home-section-desc">
        Add a recipe, extend the framework, or improve the docs — contributions of all sizes are welcome.
      </p>

      <div class="cap-grid">
        <a
          v-for="item in items"
          :key="item.title"
          class="cap-card cap-card-link contrib-card"
          :href="item.href"
        >
          <span class="cap-icon" :style="{ color: item.color }">
            <component :is="item.icon" />
          </span>
          <div class="cap-name">{{ item.title }}</div>
          <div class="cap-desc">{{ item.desc }}</div>
          <span class="contrib-badge">{{ item.badge }}</span>
        </a>
      </div>

    </div>
  </section>
</template>

<script setup>
import { h } from 'vue'

const IconGitBranch = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('line', { x1: 6, y1: 3, x2: 6, y2: 15 }),
  h('circle', { cx: 18, cy: 6, r: 3 }),
  h('circle', { cx: 6, cy: 18, r: 3 }),
  h('path', { d: 'M18 9a9 9 0 0 1-9 9' }),
])

const IconTerminal = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('polyline', { points: '4 17 10 11 4 5' }),
  h('line', { x1: 12, y1: 19, x2: 20, y2: 19 }),
])

const IconGitPR = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('circle', { cx: 18, cy: 18, r: 3 }),
  h('circle', { cx: 6, cy: 6, r: 3 }),
  h('path', { d: 'M13 6h3a2 2 0 0 1 2 2v7' }),
  h('line', { x1: 6, y1: 9, x2: 6, y2: 21 }),
])

const IconFileText = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('path', { d: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z' }),
  h('polyline', { points: '14 2 14 8 20 8' }),
  h('line', { x1: 16, y1: 13, x2: 8, y2: 13 }),
  h('line', { x1: 16, y1: 17, x2: 8, y2: 17 }),
  h('polyline', { points: '10 9 9 9 8 9' }),
])

const IconBookUpload = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('path', { d: 'M4 19.5A2.5 2.5 0 0 1 6.5 17H20' }),
  h('path', { d: 'M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z' }),
  h('path', { d: 'M12 13V7' }),
  h('path', { d: 'M9.5 9.5 12 7l2.5 2.5' }),
])

const IconLayers = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('path', { d: 'm12 3 9 4.5-9 4.5-9-4.5L12 3Z' }),
  h('path', { d: 'm3 12 9 4.5 9-4.5' }),
  h('path', { d: 'm3 16.5 9 4.5 9-4.5' }),
])

const items = [
  {
    title: 'Dev setup',
    desc: 'Get your local environment ready — editable install, linting, and test runner.',
    badge: 'dev-setup',
    href: './contributing/dev-setup.html',
    icon: IconTerminal,
    color: '#0f7a5a',
  },
  {
    title: 'CI & pull requests',
    desc: 'Understand the CI pipeline and what reviewers look for in a PR.',
    badge: 'ci-and-pr',
    href: './contributing/ci-and-pr.html',
    icon: IconGitPR,
    color: '#4254c8',
  },
  {
    title: 'Add a stage',
    desc: 'Extend the research pipeline with a new processing or evaluation stage.',
    badge: 'adding-a-stage',
    href: './contributing/adding-a-stage.html',
    icon: IconGitBranch,
    color: '#b0305a',
  },
  {
    title: 'Core overview',
    desc: 'Read the architecture docs before changing Systems, components, parallel, or config behavior.',
    badge: 'core',
    href: './core/index.html',
    icon: IconLayers,
    color: '#4254c8',
  },
  {
    title: 'Writing docs',
    desc: 'Edit Markdown, use Vue components, and understand the current VuePress setup.',
    badge: 'writing-docs',
    href: './contributing/writing-docs.html',
    icon: IconBookUpload,
    color: '#b06a00',
  },
  {
    title: 'Docstring guide',
    desc: 'Write docs that are consistent, searchable, and auto-rendered.',
    badge: 'docstring-guide',
    href: './contributing/docstring-guide.html',
    icon: IconFileText,
    color: '#0f7a5a',
  },
]
</script>

<style scoped>
.home-contribute {
  border-top: 0.5px solid #e2e6ec;
}

.contrib-card {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-bottom: 16px;
}

.contrib-badge {
  margin-top: 8px;
  display: inline-block;
  font-size: 10px;
  font-family: var(--vp-font-family-mono);
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 2px 8px;
  border-radius: 4px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  align-self: flex-start;
}

.contrib-cta {
  margin-top: 8px;
  padding: 14px 18px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 0.5px solid var(--vp-c-divider);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.contrib-cta-text {
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.contrib-cta-text strong {
  color: var(--vp-c-text-1);
  font-weight: 600;
}
.contrib-cta-text code {
  font-size: 11px;
  font-family: var(--vp-font-family-mono);
  background: var(--vp-c-bg-mute);
  padding: 1px 5px;
  border-radius: 3px;
  border: 0.5px solid var(--vp-c-divider);
}

.contrib-cta-link {
  font-size: 12px;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  text-decoration: none;
  white-space: nowrap;
}
.contrib-cta-link:hover {
  text-decoration: underline;
}
</style>
