<template>
  <section class="home-recipes">
    <div class="home-section-inner">

      <p class="home-label">Recipes</p>
      <h2 class="home-section-title">Clone a recipe, run in minutes</h2>
      <p class="home-section-desc">
        Pick a built-in recipe and clone it locally — configs, data pipeline, and training script all included.
      </p>

      <!-- Command box -->
      <div class="recipe-cmd">
        <code class="recipe-cmd-code">
          <span class="recipe-cmd-prefix">$</span>
          <span class="recipe-cmd-text">
            espnet3 clone
            <span class="recipe-cmd-arg">{{ activeRecipe }}</span>
            <span class="recipe-cmd-opt">&nbsp;--project my_project</span>
          </span>
        </code>
        <button class="recipe-copy-btn" :class="{ copied }" @click="copy">
          <Transition name="cmd-fade" mode="out-in">
            <span :key="copied ? 'copied' : 'copy'">{{ copied ? '✓ Copied' : 'Copy' }}</span>
          </Transition>
        </button>
      </div>
      <p class="recipe-cmd-hint">
        Creates <span class="recipe-cmd-hint-path">my_project/</span> with the full recipe — edit &amp; run immediately.
      </p>

      <!-- Recipe chips -->
      <div class="recipe-shelf">
        <div class="recipe-shelf-label">Available recipes</div>
        <div class="recipe-chips">
          <button
            v-for="r in recipes"
            :key="r"
            class="recipe-chip"
            :class="{ active: activeRecipe === r }"
            @click="activeRecipe = r"
          >{{ r }}</button>
        </div>
        <a class="recipe-browse-link" href="https://github.com/espnet/espnet/tree/master/egs3">Browse all recipes</a>
      </div>

      <!-- Guide cards -->
      <template v-if="showGuides">
        <p class="recipe-guides-label">Guides</p>
        <div class="recipe-guide-grid">
          <article
            v-for="guide in guides"
            :key="guide.name"
            class="recipe-guide-card"
          >
            <div class="recipe-guide-head">
              <span class="recipe-guide-icon" :style="{ background: guide.iconBg }">
                <component :is="guide.icon" :color="guide.iconColor" />
              </span>
              <a class="recipe-guide-name" :href="guide.href">{{ guide.name }}</a>
              <span class="recipe-guide-badge" :style="{ background: guide.iconBg, color: guide.iconColor }">
                {{ guide.badge }}
              </span>
            </div>
            <p class="recipe-guide-desc">{{ guide.desc }}</p>
            <ul class="recipe-guide-links">
              <li v-for="link in guide.links" :key="link.label">
                <a :href="link.href" @click.stop class="recipe-guide-link">
                  {{ link.label }}
                </a>
              </li>
            </ul>
          </article>
        </div>
      </template>

    </div>
  </section>
</template>

<script setup>
import { ref, h } from 'vue'

const props = defineProps({
  showGuides: { type: Boolean, default: true },
})

const copied = ref(false)
const activeRecipe = ref('librispeech/asr')

const recipes = [
  'librispeech/asr',
  'commonvoice/asr',
  'voxceleb/sv',
  'ljspeech/tts',
  'must-c/st',
  'wsj/enh',
]

function copy() {
  navigator.clipboard.writeText(`espnet3 clone ${activeRecipe.value} --project my_project`).then(() => {
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  })
}

const IconBook = ({ color }) => h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('path', { d: 'M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z' }),
  h('path', { d: 'M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z' }),
])

const IconCirclePlus = ({ color }) => h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('circle', { cx: 12, cy: 12, r: 10 }),
  h('line', { x1: 12, y1: 8, x2: 12, y2: 16 }),
  h('line', { x1: 8, y1: 12, x2: 16, y2: 12 }),
])

const IconActivity = ({ color }) => h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('polyline', { points: '22 12 18 12 15 21 9 3 6 12 2 12' }),
])

const IconBox = ({ color }) => h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('path', { d: 'M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z' }),
])

const IconScaling = ({ color }) => h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('polyline', { points: '23 6 13.5 15.5 8.5 10.5 1 18' }),
  h('polyline', { points: '17 6 23 6 23 12' }),
])

const IconConfig = ({ color }) => h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('circle', { cx: 12, cy: 12, r: 3 }),
  h('path', { d: 'M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z' }),
])

const guides = [
  {
    name: 'What is a recipe',
    badge: 'Guide',
    desc: 'See how one ESPnet3 recipe holds configs, stages, dataset code, and outputs in one project.',
    href: './get-started/what-is-a-recipe.html',
    icon: IconBook,
    iconBg: '#edf8f3',
    iconColor: '#0f7a5a',
    links: [
      { label: 'Files and stages',         href: './get-started/what-is-a-recipe.html#how-files-and-stages-connect' },
      { label: 'Clone a recipe',           href: './get-started/what-is-a-recipe.html#clone-a-recipe-and-keep-working-in-it' },
      { label: 'Typical flow',             href: './get-started/what-is-a-recipe.html#typical-flow' },
    ],
  },
  {
    name: 'Scaling',
    badge: 'Guide',
    desc: 'Scale training to multiple nodes, handle large datasets, and run batch inference.',
    href: './guides/scaling/index.html',
    icon: IconScaling,
    iconBg: '#f3f0fb',
    iconColor: '#6a30b0',
    links: [
      { label: 'Multi-node training',      href: './guides/scaling/multi-node.html' },
      { label: 'Large-scale data',         href: './guides/scaling/data-pipeline.html' },
      { label: 'Inference at scale',       href: './guides/scaling/inference.html' },
    ],
  },
  {
    name: 'Finetuning',
    badge: 'Guide',
    desc: 'Adapt a cloned recipe to your own dataset, model, and training loop.',
    href: './guides/finetuning/index.html',
    icon: IconActivity,
    iconBg: '#fdf5e6',
    iconColor: '#b06a00',
    links: [
      { label: 'Custom dataset',           href: './guides/finetuning/custom-dataset.html' },
      { label: 'Customize the model',      href: './guides/finetuning/custom-model.html' },
      { label: 'Customize training loop',  href: './guides/finetuning/training-loop.html' },
    ],
  },
  {
    name: 'Config',
    badge: 'Reference',
    desc: 'Understand the YAML config slots for training, inference, dataset, resolvers, and publication.',
    href: './core/config/index.html',
    icon: IconConfig,
    iconBg: '#e6f7f7',
    iconColor: '#0f7a7a',
    links: [
      { label: 'Training config',          href: './core/config/training.html' },
      { label: 'Resolvers',                href: './core/config/resolvers.html' },
      { label: 'Dataset config',           href: './core/config/dataset.html' },
    ],
  },
  {
    name: 'Migrating from ESPnet2',
    badge: 'Guide',
    desc: 'Move an ESPnet2 recipe to ESPnet3 and understand the structure, config, and parallel changes.',
    href: './guides/migrating-from-espnet2/index.html',
    icon: IconCirclePlus,
    iconBg: '#e6f1fb',
    iconColor: '#185fa5',
    links: [
      { label: 'Recipe structure',         href: './guides/migrating-from-espnet2/recipe-structure.html' },
      { label: 'Task to System',           href: './guides/migrating-from-espnet2/task-to-system.html' },
      { label: 'Config diff',              href: './guides/migrating-from-espnet2/config-diff.html' },
    ],
  },
  {
    name: 'From other toolkits',
    badge: 'Guide',
    desc: 'Read the high-level migration notes if your reference point is Hugging Face, NeMo, or SpeechBrain.',
    href: './guides/coming-from-other-toolkits/index.html',
    icon: IconBox,
    iconBg: '#fbeaf0',
    iconColor: '#b0305a',
    links: [
      { label: 'From Hugging Face',        href: './guides/coming-from-other-toolkits/from-huggingface.html' },
      { label: 'From NeMo',                href: './guides/coming-from-other-toolkits/from-nemo.html' },
      { label: 'From SpeechBrain',         href: './guides/coming-from-other-toolkits/from-speechbrain.html' },
    ],
  },
]
</script>

<style scoped>
/* ── Section wrapper ───────────────────────────────── */
.home-recipes {
  border-top: 0.5px solid #e2e6ec;
  background: #ffffff;
}

/* ── Command box ───────────────────────────────────── */
.recipe-cmd {
  display: flex;
  align-items: center;
  gap: 10px;
  background: #0d1520;
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 8px;
}

.recipe-cmd-code {
  flex: 1;
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 13px;
  color: #e2e8f0;
  background: transparent;
  display: flex;
  align-items: center;
  gap: 8px;
  overflow: hidden;
  white-space: nowrap;
}

.recipe-cmd-prefix {
  color: #0f7a5a;
  user-select: none;
  flex-shrink: 0;
}

.recipe-cmd-arg  { color: #7dd3b8; }
.recipe-cmd-opt  { color: #6b7a8d; }

.recipe-copy-btn {
  font-size: 11px;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: 5px;
  border: 0.5px solid #2a3a50;
  background: transparent;
  color: #7a8898;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
  flex-shrink: 0;
  min-width: 64px;
  text-align: center;
}
.recipe-copy-btn:hover,
.recipe-copy-btn.copied {
  border-color: #0f7a5a;
  color: #0f7a5a;
}

.recipe-cmd-hint {
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 11px;
  color: #9aaabb;
  margin-bottom: 20px;
  padding-left: 2px;
}
.recipe-cmd-hint-path { color: #0f7a5a; }

/* ── Recipe shelf ──────────────────────────────────── */
.recipe-shelf {
  background: #ffffff;
  border: 0.5px solid #e2e6ec;
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 28px;
}

.recipe-shelf-label {
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #9aaabb;
  margin-bottom: 10px;
}

.recipe-chips {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.recipe-chip {
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 5px;
  background: #f7f8fa;
  color: #3a4a5c;
  border: 0.5px solid #e2e6ec;
  cursor: pointer;
  transition: all 0.15s;
}
.recipe-chip:hover {
  border-color: rgba(15, 122, 90, 0.3);
  color: #0f7a5a;
}
.recipe-chip.active {
  background: #edf8f3;
  color: #0f7a5a;
  border-color: rgba(15, 122, 90, 0.35);
}

.recipe-browse-link {
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 11px;
  color: #0f7a5a;
  text-decoration: none;
}
.recipe-browse-link:hover { text-decoration: underline; }

/* ── Guide cards ───────────────────────────────────── */
.recipe-guides-label {
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #c0cad8;
  margin-bottom: 12px;
}

.recipe-guide-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.recipe-guide-card {
  background: #ffffff;
  border: 0.5px solid #e2e6ec;
  border-radius: 10px;
  padding: 15px 16px;
  color: inherit;
  display: block;
  transition: border-color 0.18s;
}
.recipe-guide-card:hover { border-color: rgba(15, 122, 90, 0.35); }

.recipe-guide-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.recipe-guide-icon {
  width: 26px;
  height: 26px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.recipe-guide-name {
  font-weight: 600;
  font-size: 14px;
  color: #0d1520;
  flex: 1;
  line-height: 1.3;
  text-decoration: none;
}

.recipe-guide-name:hover {
  color: #0f7a5a;
}

.recipe-guide-badge {
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 4px;
  flex-shrink: 0;
}

.recipe-guide-desc {
  font-size: 13px;
  color: #7b8795;
  line-height: 1.55;
  margin-bottom: 9px;
}

.recipe-guide-links {
  list-style: none;
  margin: 0;
  padding: 0;
  border-top: 0.5px solid #f0f3f6;
  padding-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.recipe-guide-link {
  font-size: 13px;
  color: #3a4a5c;
  text-decoration: none;
  padding: 2px 0;
  display: flex;
  align-items: baseline;
  gap: 5px;
  line-height: 1.45;
  transition: color 0.15s;
}
.recipe-guide-link::before {
  content: '›';
  color: rgba(15, 122, 90, 0.45);
  flex-shrink: 0;
}
.recipe-guide-link:hover { color: #0f7a5a; }


/* ── Responsive ────────────────────────────────────── */
@media (max-width: 520px) {
  .recipe-guide-grid {
    grid-template-columns: 1fr;
  }
  .recipe-cmd-code {
    font-size: 11px;
  }
}

/* ── Transition (shared with HomeHero) ─────────────── */
.cmd-fade-enter-active,
.cmd-fade-leave-active {
  transition: opacity 0.15s, transform 0.15s;
}
.cmd-fade-enter-from,
.cmd-fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
