<template>
  <div class="ce-root">
    <div class="ce-card">
      <div class="ce-tabs">
        <button
          v-for="key in presetKeys"
          :key="key"
          :class="['ce-tab', { active: activePreset === key }]"
          @click="selectPreset(key)"
        >
          {{ PRESETS[key].label }}
        </button>
      </div>

      <div class="ce-layout">
        <!-- left: yaml editor -->
        <div class="ce-pane">
          <div class="ce-pane-header">
            <span>demo.yaml</span>
            <span class="ce-badge ce-badge--blue">editable</span>
          </div>
          <textarea
            v-model="yamlText"
            class="ce-textarea"
            spellcheck="false"
          />
        </div>

        <!-- right: gradio preview -->
        <div class="ce-pane">
          <div class="ce-pane-header">
            <span>Gradio preview</span>
            <span class="ce-badge ce-badge--green">live</span>
          </div>
          <div class="ce-preview">
            <p v-if="!hasContent" class="ce-empty">
              Add inputs/outputs in the YAML to preview
            </p>

            <!-- gradio window -->
            <div v-else class="gw">
              <div class="gw__header">
                <div class="gw__title">{{ parsedUI.title || 'Gradio Demo' }}</div>
                <div v-if="parsedUI.description" class="gw__desc">{{ parsedUI.description }}</div>
              </div>
              <div class="gw__body">
                <template v-for="(spec, i) in parsedUI.inputs" :key="`in-${i}`">
                  <GradioComponent :spec="spec" :is-input="true" />
                </template>
                <button class="gw__btn">Run</button>
                <template v-for="(spec, i) in parsedUI.outputs" :key="`out-${i}`">
                  <GradioComponent :spec="spec" :is-input="false" />
                </template>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="ce-callout">
      <strong>Built-in types:</strong> only <code>audio</code> and <code>text</code>
      are registered by default in <code>UIAssetRegistry</code>.
      Any other value in <code>type:</code> will cause a <code>KeyError</code>
      at runtime — unless you register a custom <code>UIAsset</code> in your
      <code>app.py</code> first.
    </div>
  </div>
</template>

<script setup>
import { ref, computed, defineComponent, h } from 'vue'

// ── GradioComponent (inline) ───────────────────────────────────────────────

const GradioComponent = defineComponent({
  props: {
    spec: { type: Object, required: true },
    isInput: { type: Boolean, required: true },
  },
  setup(props) {
    const label = computed(() => props.spec.label ?? props.spec.key ?? props.spec.type ?? '?')
    return () => {
      const labelEl = h('div', { class: 'gc__label' }, label.value)

      let body
      if (props.spec.type === 'audio') {
        body = h('div', { class: ['gc__audio', props.isInput && 'gc__audio--input'] }, [
          h('div', { class: 'gc__audio-icon' }, props.isInput ? '🎤' : '🔊'),
          h('div', props.isInput ? 'Drop audio or record' : 'Audio output'),
        ])
      } else if (props.spec.type === 'text') {
        body = h('div', { class: 'gc__textbox' }, props.isInput ? 'Type here…' : 'Output will appear here')
      } else {
        body = h('div', { class: 'gc__error' }, `Unknown type: "${props.spec.type}" — register a custom UIAsset`)
      }

      return h('div', { class: 'gc' }, [labelEl, body])
    }
  },
})

// ── Presets ────────────────────────────────────────────────────────────────

const PRESETS = {
  basic: {
    label: 'basic ASR',
    yaml: `model:
  dir_or_tag: exp/training_asr/model_pack
  trust_user_code: true

ui:
  app_script: src/app.py
  inputs:
    - type: audio
      key: speech
      label: "Input speech"
  outputs:
    - type: text
      key: transcription
      label: "Transcription"

upload_demo:
  hf_repo: your-org/your-demo`,
  },

  titled: {
    label: 'with title & description',
    yaml: `model:
  dir_or_tag: exp/training_asr/model_pack
  trust_user_code: true

ui:
  app_script: src/app.py
  title: "ASR Demo"
  description: "Upload audio to get a transcription."
  inputs:
    - type: audio
      key: speech
      label: "Input speech"
  outputs:
    - type: text
      key: transcription
      label: "Transcription"

upload_demo:
  hf_repo: your-org/your-demo`,
  },

  multi: {
    label: 'multiple outputs',
    yaml: `model:
  dir_or_tag: exp/translation/model_pack
  trust_user_code: true

ui:
  app_script: src/app.py
  title: "Speech Translation"
  inputs:
    - type: audio
      key: speech
      label: "Input speech"
    - type: text
      key: prompt
      label: "Translation prompt"
  outputs:
    - type: text
      key: transcription
      label: "Transcription"
    - type: text
      key: translation
      label: "Translation"

upload_demo:
  hf_repo: your-org/your-demo`,
  },
}

// ── State ──────────────────────────────────────────────────────────────────

const presetKeys = Object.keys(PRESETS)
const activePreset = ref('basic')
const yamlText = ref(PRESETS['basic'].yaml)

// ── YAML parser ────────────────────────────────────────────────────────────

function parseYAML(text) {
  const ui = { inputs: [], outputs: [] }
  let section = null
  let list = null
  let item = null

  for (const raw of text.split('\n')) {
    const line = raw.trimEnd()
    if (!line || line.trim().startsWith('#')) continue

    const indent = line.length - line.trimStart().length
    const content = line.trim()

    if (indent === 0) {
      section = content.replace(':', '')
      list = null
      item = null
      continue
    }

    if (section !== 'ui') continue

    if (indent === 2) {
      if (content.startsWith('title:')) {
        ui.title = content.replace('title:', '').trim().replace(/['"]/g, '')
        continue
      }
      if (content.startsWith('description:')) {
        ui.description = content.replace('description:', '').trim().replace(/['"]/g, '')
        continue
      }
      if (content === 'inputs:')  { list = 'inputs';  item = null; continue }
      if (content === 'outputs:') { list = 'outputs'; item = null; continue }
    }

    if (list && indent >= 4) {
      if (content.startsWith('- type:')) {
        item = { type: content.replace('- type:', '').trim() }
        ui[list].push(item)
      } else if (item) {
        if (content.startsWith('type:'))  item.type  = content.replace('type:', '').trim()
        if (content.startsWith('key:'))   item.key   = content.replace('key:', '').trim()
        if (content.startsWith('label:')) item.label = content.replace('label:', '').trim().replace(/['"]/g, '')
      }
    }
  }

  return ui
}

// ── Computed ───────────────────────────────────────────────────────────────

const parsedUI = computed(() => parseYAML(yamlText.value))
const hasContent = computed(
  () => parsedUI.value.inputs.length > 0 || parsedUI.value.outputs.length > 0,
)

// ── Actions ────────────────────────────────────────────────────────────────

function selectPreset(key) {
  activePreset.value = key
  yamlText.value = PRESETS[key].yaml
}
</script>

<style lang="scss">
$border:      rgba(26, 25, 23, 0.10);
$border2:     rgba(26, 25, 23, 0.18);
$accent:      #2563eb;
$accent-soft: #eff4ff;
$accent-mid:  #bfcfff;
$green:       #16a34a;
$green-soft:  #f0fdf4;
$green-mid:   #86efac;
$amber-soft:  #fffbeb;
$amber-mid:   #fcd34d;
$ink:         #1a1917;
$ink2:        #4a4845;
$ink3:        #8a8784;
$bg:          #f9f8f6;
$radius:      10px;
$radius-lg:   14px;
$mono:        'JetBrains Mono', monospace;
$sans:        'Sora', sans-serif;

.ce-root {
  // ── card shell ──────────────────────────────────────────────────────────
  .ce-card {
    border: 1px solid $border;
    border-radius: $radius-lg;
    overflow: hidden;
  }

  // ── preset tabs ─────────────────────────────────────────────────────────
  .ce-tabs {
    display: flex;
    gap: 2px;
    padding: 10px 16px 0;
    border-bottom: 1px solid $border;
    background: $bg;
  }

  .ce-tab {
    font-family: $sans;
    font-size: 11px;
    font-weight: 500;
    padding: 5px 11px;
    border-radius: 5px 5px 0 0;
    border: 1px solid transparent;
    border-bottom: none;
    background: transparent;
    color: $ink3;
    cursor: pointer;
    transition: all 0.15s;

    &.active {
      color: $ink;
      background: #fff;
      border-color: $border;
      margin-bottom: -1px;
      position: relative;
      z-index: 1;
    }
  }

  // ── split layout ────────────────────────────────────────────────────────
  .ce-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: $border;
  }

  .ce-pane { background: #fff; }

  .ce-pane-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    border-bottom: 1px solid $border;
    font-size: 11px;
    font-weight: 500;
    color: $ink2;
  }

  // ── badge ───────────────────────────────────────────────────────────────
  .ce-badge {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 2px 8px;
    border-radius: 100px;
    border: 1px solid;

    &--blue  { color: $accent; background: $accent-soft; border-color: $accent-mid; }
    &--green { color: $green;  background: $green-soft;  border-color: $green-mid;  }
  }

  // ── textarea ────────────────────────────────────────────────────────────
  .ce-textarea {
    display: block;
    width: 100%;
    height: 340px;
    padding: 16px;
    font-family: $mono;
    font-size: 12px;
    line-height: 1.75;
    color: $ink;
    background: #fff;
    border: none;
    outline: none;
    resize: none;
  }

  // ── preview pane ────────────────────────────────────────────────────────
  .ce-preview {
    padding: 16px;
    min-height: 340px;
  }

  .ce-empty {
    text-align: center;
    color: $ink3;
    padding: 40px 0;
    font-size: 12px;
  }

  // ── callout ─────────────────────────────────────────────────────────────
  .ce-callout {
    display: flex;
    gap: 10px;
    padding: 12px 16px;
    margin-top: 16px;
    background: $amber-soft;
    border: 1px solid $amber-mid;
    border-radius: $radius;
    font-size: 12px;
    color: #78350f;
    line-height: 1.55;

    code {
      font-family: $mono;
      font-size: 11px;
      background: rgba(217, 119, 6, 0.12);
      padding: 1px 5px;
      border-radius: 3px;
    }
  }

  // ── gradio window ───────────────────────────────────────────────────────
  .gw {
    border: 1px solid $border2;
    border-radius: $radius;
    overflow: hidden;

    &__header {
      background: #f8faff;
      border-bottom: 1px solid $border;
      padding: 10px 14px;
    }

    &__title {
      font-size: 13px;
      font-weight: 600;
      color: $ink;
    }

    &__desc {
      font-size: 11px;
      color: $ink3;
      margin-top: 2px;
    }

    &__body {
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 14px;
    }

    &__btn {
      width: 100%;
      padding: 8px 18px;
      background: $accent;
      color: #fff;
      border: none;
      border-radius: 7px;
      font-family: $sans;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s;

      &:hover { background: #1d4ed8; }
    }
  }

  // ── gradio component ────────────────────────────────────────────────────
  .gc {
    display: flex;
    flex-direction: column;
    gap: 4px;

    &__label {
      font-size: 10px;
      font-weight: 500;
      color: $ink2;
    }

    &__audio {
      background: $bg;
      border: 1.5px dashed $border2;
      border-radius: 7px;
      padding: 14px;
      text-align: center;
      color: $ink3;
      font-size: 11px;
      transition: all 0.15s;

      &--input {
        cursor: pointer;
        &:hover {
          border-color: $accent;
          color: $accent;
          background: $accent-soft;
        }
      }

      &-icon {
        font-size: 18px;
        margin-bottom: 2px;
      }
    }

    &__textbox {
      border: 1px solid $border2;
      border-radius: 7px;
      padding: 8px 10px;
      font-size: 12px;
      color: $ink3;
      background: #fff;
      min-height: 48px;
    }

    &__error {
      border: 1px solid #fecaca;
      border-radius: 7px;
      padding: 8px 10px;
      font-family: $mono;
      font-size: 11px;
      color: #b91c1c;
      background: #fef2f2;
    }
  }
}
</style>
