<template>
  <section class="home-pipeline">
    <div class="home-section-inner">
      <p class="home-label">Research Pipeline</p>
      <h2 class="home-section-title">Stages</h2>
      <p class="home-section-desc">
        Click a stage, to see the details
      </p>

      <div class="pipeline">
        <div
          v-for="(step, si) in steps"
          :key="step.name"
          class="pipe-step"
          :class="{ active: activeIndex === si }"
          role="button"
          tabindex="0"
          :aria-expanded="activeIndex === si"
          :aria-controls="`pipe-detail-${si}`"
          @click="toggle(si)"
          @keydown.enter.prevent="toggle(si)"
          @keydown.space.prevent="toggle(si)"
        >
          <div class="pipe-left">
            <div class="pipe-dot" />
            <div v-if="si < steps.length - 1" class="pipe-line" />
          </div>
          <div class="pipe-card">
            <div class="pipe-name">{{ step.name }}</div>
            <div class="pipe-meta-row">
              <span class="pipe-meta">{{ step.desc }}</span>
              <a class="pipe-link" :href="step.link" @click.stop>View docs →</a>
            </div>

            <div
              :id="`pipe-detail-${si}`"
              class="pipe-detail"
              :class="{ 'pipe-detail-open': activeIndex === si }"
            >
              <div class="pipe-detail-inner">
                <div v-if="step.tasks?.length" class="pipe-detail-row">
                  <button
                    v-for="(t, ti) in step.tasks"
                    :key="t.label"
                    type="button"
                    class="tag tag-green"
                    :class="{ 'tag-selected': activeTasks[si] === ti }"
                    :aria-pressed="activeTasks[si] === ti"
                    @click.stop="activeTasks[si] = ti"
                  >{{ t.label }}</button>
                </div>
                <pre v-if="step.tasks?.length" class="pipe-yaml"><code v-html="step.tasks[activeTasks[si]].yaml" /></pre>
                <div v-else class="pipe-yaml">No configuration required</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, reactive } from 'vue'

const activeIndex = ref(0)
const activeTasks = reactive(Array(8).fill(0))

function toggle(i) {
  activeIndex.value = i
}

function yaml(lines) {
  return lines
    .map(l =>
      l
        .replace(/^(\s*)(-\s+)([^:]+)$/, (_, sp, dash, v) => (
          `${sp}${dash}<span class="v">${v.trim()}</span>`
        ))
        .replace(/^(\s*(?:-\s+)?)?([\w_-]+):(.*)?$/, (_, sp, k, v) => {
          const prefix = sp || ''
          const val = (v || '').trim()
          return val
            ? `${prefix}<span class="k">${k}:</span> <span class="v">${val}</span>`
            : `${prefix}<span class="k">${k}:</span>`
        })
    )
    .join('\n')
}

const steps = [
  {
    name: "Creating Dataset",
    desc: "Prepare source data and manifests",
    link: "./stages/create-dataset.html",
    tasks: [
      {
        label: "LibriSpeech",
        yaml: yaml([
          "dataset:",
          "  train:",
          "    - data_src: librispeech_100/esp2_asr",
          "      data_src_args:",
          "        split: train-clean-100",
          "  valid:",
          "    - data_src: librispeech_100/esp2_asr",
          "      data_src_args:",
          "        split: dev-clean",
        ]),
      },
      {
        label: "Custom",
        yaml: yaml([
          "dataset:",
          "  recipe_dir: ${recipe_dir}",
          "  train:",
          "    - data_src_args:",
          "        split: train",
          "  valid:",
          "    - data_src_args:",
          "        split: valid",
        ]),
      },
    ],
  },
  {
    name: "Train Tokenizer",
    desc: "ASR-specific declared stage",
    link: "./stages/train-tokenizer.html",
    tasks: [
      {
        label: "tokenizer",
        yaml: yaml([
          "tokenizer:",
          "  model_type: bpe",
          "  vocab_size: 30",
          "  save_path: ${data_dir}/bpe_30",
        ]),
      },
    ],
  },
  {
    name: "Collect Stats",
    desc: "Feature statistics and shape collection",
    link: "./stages/collect-stats.html",
    tasks: [
    ],
  },
  {
    name: "Training",
    desc: "Training model loss protocol and Lightning settings",
    link: "./stages/train.html",
    tasks: [
      {
        label: "model",
        yaml: yaml([
          "task: espnet3.systems.esp2_asr.task.ASRTask",
          "model:",
          "  encoder: transformer",
          "  decoder: transformer",
          "  model_conf:",
          "    ctc_weight: 0.3",
        ]),
      },
      {
        label: "optimizer",
        yaml: yaml([
          "optimizer:",
          "  _target_: torch.optim.Adam",
          "  lr: 0.001",
        ]),
      },
      {
        label: "trainer",
        yaml: yaml([
          "num_device: 4",
          "trainer:",
          "  accelerator: gpu",
          "  devices: ${num_device}",
          "  strategy: ddp",
          "  max_epochs: 50",
        ]),
      },
      {
        label: "dataloader",
        yaml: yaml([
          "dataloader:",
          "  train:",
          "    iter_factory: null",
          "    batch_size: 32",
          "    shuffle: true",
          "  valid:",
          "    iter_factory: null",
          "    batch_size: 32",
          "    shuffle: false",
        ]),
      },
    ],
  },
  {
    name: "Inference",
    desc: "Checked audio inputs and named text outputs",
    link: "./stages/inference.html",
    tasks: [
      {
        label: "dataset",
        yaml: yaml([
          "dataset:",
          "  test:",
          "    - name: eval1",
          "      data_src: librispeech_100/esp2_asr",
          "      data_src_args:",
          "        split: test-clean",
          "    - name: eval2",
          "      data_src: librispeech_100/esp2_asr",
          "      data_src_args:",
          "        split: test-other",
        ]),
      },
      {
        label: "runner and contract",
        yaml: yaml([
          "input_key: speech",
          "runner:",
          "  _target_: espnet3.systems.esp2_asr.inference.ASRInferenceRunner",
          "contract:",
          "  inputs:",
          "    speech:",
          "      kind: audio",
          "      sample_rate: 16000",
          "  outputs:",
          "    text:",
          "      kind: text",
        ]),
      },
    ],
  },
  {
    name: "Metrics",
    desc: "Bind metric contract inputs to artifacts or dataset columns",
    link: "./stages/metrics.html",
    tasks: [
      {
        label: "WER",
        yaml: yaml([
          "metrics:",
          "  - metric:",
          "      _target_: espnet3.systems.esp2_asr.metrics.wer.WER",
          "    inputs:",
          "      ref: dataset:text",
          "      hyp: text",
        ]),
      },
    ],
  },
  {
    name: "Publication",
    desc: "Pack local artifacts; uploading is explicit",
    link: "./stages/publish.html",
    tasks: [
      {
        label: "pack-model",
        yaml: yaml([
          "pack_model:",
          "  out_dir: ${exp_dir}/model_pack",
          "  include:",
          "    - src",
          "    - dataset",
          "  exclude:",
          "    - last.ckpt",
        ]),
      },
      {
        label: "upload-model",
        yaml: yaml([
          "upload_model:",
          "  hf_repo: your-user/my-asr",
          "  private: false",
          "  update: false",
        ]),
      },
    ],
  },
  {
    name: "Demo",
    desc: "UI fields derive from packed inference contract",
    link: "./stages/demo.html",
    tasks: [
      {
        label: "model and UI",
        yaml: yaml([
          "model:",
          "  dir_or_tag: exp/${exp_tag}/model_pack",
          "  device: auto",
          "ui:",
          "  app_script: ${config_path:../src/app.py}",
          "  title: ASR demo",
        ]),
      },
      {
        label: "pack",
        yaml: yaml([
          "pack:",
          "  out_dir: demo",
          "  include: []",
        ]),
      },
    ],
  },
]
</script>
