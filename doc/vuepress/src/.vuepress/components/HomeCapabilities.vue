<template>
  <section class="home-capabilities">
    <div class="home-section-inner">
      <p class="home-label">Research at Scale</p>
      <h2 class="home-section-title">Key features</h2>
      <p class="home-section-desc">
        Built for the foundation model era — distributed, multilingual, reproducible.
      </p>

      <div class="cap-grid">
        <a
          v-for="cap in capabilities"
          :key="cap.name"
          class="cap-card cap-card-link"
          :href="cap.href"
        >
          <span class="cap-icon" :style="{ color: cap.color }">
            <component :is="cap.icon" />
          </span>
          <div class="cap-name">{{ cap.name }}</div>
          <div class="cap-desc">{{ cap.desc }}</div>
        </a>
      </div>
    </div>
  </section>
</template>

<script setup>
import { h } from 'vue'

// Minimal inline SVG icons (no external dependency)
const IconServer = () => h('svg', { width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('rect', { x: 2, y: 3, width: 20, height: 6, rx: 1 }),
  h('rect', { x: 2, y: 12, width: 20, height: 6, rx: 1 }),
  h('circle', { cx: 18, cy: 6, r: 1, fill: 'currentColor' }),
  h('circle', { cx: 18, cy: 15, r: 1, fill: 'currentColor' }),
])

const IconActivity = () => h('svg', { width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('polyline', { points: '22 12 18 12 15 21 9 3 6 12 2 12' }),
])

const IconPuzzle = () => h('svg', { width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('path', { d: 'M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.61a2.404 2.404 0 0 1-3.408 0l-1.569-1.568a.877.877 0 0 0-.878-.288c-.322.049-.622.239-.801.52-.479.795-1.422 1.222-2.29 1.018-.868-.204-1.554-.89-1.758-1.758-.204-.868.223-1.811 1.018-2.29.281-.179.471-.479.52-.801a.877.877 0 0 0-.288-.877l-1.568-1.57a2.404 2.404 0 0 1 0-3.407l1.61-1.61a.98.98 0 0 1 .837-.277c.47.07.802.48.968.926a2.501 2.501 0 1 0 3.214-3.215c-.446-.166-.855-.497-.925-.968a.979.979 0 0 1 .276-.837l1.61-1.61a2.404 2.404 0 0 1 3.408 0l1.568 1.568c.23.23.556.338.878.29z' }),
])

const IconDatabase = () => h('svg', { width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('ellipse', { cx: 12, cy: 5, rx: 8, ry: 3 }),
  h('path', { d: 'M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5' }),
  h('path', { d: 'M4 11v8c0 1.7 3.6 3 8 3s8-1.3 8-3v-8' }),
])

const IconRoute = () => h('svg', { width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8 }, [
  h('circle', { cx: 6, cy: 19, r: 2 }),
  h('circle', { cx: 18, cy: 5, r: 2 }),
  h('path', { d: 'M8 19h3a4 4 0 0 0 4-4V7' }),
])

const capabilities = [
  {
    name: 'Parallel',
    desc: 'See providers, runners, and Dask-backed execution patterns.',
    color: '#0f7a5a',
    icon: IconServer,
    href: 'core/parallel/',
  },
  {
    name: 'Config',
    desc: 'Understand the YAML surface for training, inference, metrics, demo, and parallel.',
    color: '#4254c8',
    icon: IconServer,
    href: 'core/config/',
  },
  {
    name: 'Datasets',
    desc: 'Read how recipe-local datasets, builders, and DataOrganizer fit together.',
    color: '#0f766e',
    icon: IconDatabase,
    href: 'core/datasets',
  },
  {
    name: 'Demo',
    desc: 'Follow the packaged demo flow from UI definition to runtime and packing.',
    color: '#b06a00',
    icon: IconActivity,
    href: 'core/demo/',
  },
  {
    name: 'Components',
    desc: 'Read the reusable data, trainer, model, metrics, and optimizer layers.',
    color: '#b0305a',
    icon: IconPuzzle,
    href: 'core/components/',
  },
  {
    name: 'Stages',
    desc: 'Start from the stage map for training, inference, metrics, and publication.',
    color: '#9a3412',
    icon: IconRoute,
    href: 'stages/',
  },
]
</script>
