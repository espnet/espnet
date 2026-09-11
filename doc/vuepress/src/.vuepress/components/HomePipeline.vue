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
          @click="toggle(si)"
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
              class="pipe-detail"
              :class="{ 'pipe-detail-open': activeIndex === si }"
            >
              <div class="pipe-detail-inner">
                <div v-if="step.tasks?.length" class="pipe-detail-row">
                  <span
                    v-for="(t, ti) in step.tasks"
                    :key="t.label"
                    class="tag tag-green"
                    :class="{ 'tag-selected': activeTasks[si] === ti }"
                    @click.stop="activeTasks[si] = ti"
                  >{{ t.label }}</span>
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
const activeTasks = reactive(Array(7).fill(0))

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
    name: 'Creating Dataset',
    desc: 'Data loading & preprocessing',
    link: './stages/create-dataset.html',
    tasks: [
      {
        label: 'LibriSpeech',
        yaml: yaml([
          'dataset:',
          '  train:',
          '    data_src: librispeech/asr',
          '    data_src_args:',
          '      split: train-clean-100',
          '  valid:',
          '    data_src: librispeech/asr',
          '    data_src_args:',
          '      split: dev-clean',
        ]),
      },
      {
        label: 'Custom',
        yaml: yaml([
          'dataset:',
          '  train:',
          '    data_src: src.dataset.MyDataset',
          '    data_src_args:',
          '      split: train',
          '  valid:',
          '    data_src: src.dataset.MyDataset',
          '    data_src_args:',
          '      split: valid',
        ]),
      },
    ],
  },
  {
    name: 'Collect Stats',
    desc: 'Feature statistics & shape collection',
    link: './stages/collect-stats.html',
    tasks: [],
  },
  {
    name: 'Training',
    desc: 'Distributed model training',
    link: './stages/train.html',
    tasks: [
      {
        label: 'model',
        yaml: yaml([
          'model:',
          '  _target_: espnet3.systems.asr.task.ASRTask',
          '  token_type: bpe',
          '  ctc_weight: 0.3',
        ]),
      },
      {
        label: 'optimizers',
        yaml: yaml([
          'optimizer:',
          '  _target_: torch.optim.Adam',
          '  lr: 0.001',
        ]),
      },
      {
        label: 'trainer',
        yaml: yaml([
          'trainer:',
          '  accelerator: gpu',
          '  devices: 4',
          '  strategy: ddp',
          '  max_epochs: 50',
        ]),
      },
      {
        label: 'dataloader',
        yaml: yaml([
          'dataloader:',
          '  train:',
          '    batch_size: 32',
          '    shuffle: true',
          '  valid:',
          '    batch_size: 32',
          '    shuffle: false',
        ]),
      },
    ],
  },
  {
    name: 'Inference',
    desc: 'Decoding & beam search',
    link: './stages/inference.html',
    tasks: [
      {
        label: 'dataset (single)',
        yaml: yaml([
          'dataset:',
          '  test:',
          '    eval1:',
          '      data_src: librispeech/asr',
          '      data_src_args:',
          '        split: test-clean',
        ]),
      },
      {
        label: 'dataset (multi)',
        yaml: yaml([
          'dataset:',
          '  test:',
          '    eval1:',
          '      data_src: librispeech/asr',
          '      data_src_args:',
          '        split: test-clean',
          '    eval2:',
          '      data_src: librispeech/asr',
          '      data_src_args:',
          '        split: test-other',
        ]),
      },
      {
        label: 'model',
        yaml: yaml([
          'model:',
          '  _target_: espnet3.publication.inference_model.InferenceModel.from_pretrained',
          '  model_tag: espnet/asr_conformer',
          'provider:',
          '  _target_: espnet3.systems.base.inference_provider.InferenceProvider',
          'runner:',
          '  batch_size: 32',
          'input_key: speech',
          'inference_dir: exp/inference',
        ]),
      },
    ],
  },
  {
    name: 'Metrics',
    desc: 'Metrics & benchmark scoring',
    link: './stages/metrics.html',
    tasks: [
      {
        label: 'WER',
        yaml: yaml([
          'metrics:',
          '  - _target_: espnet3.systems.asr.metrics.wer.WER',
          '    normalize: true',
          '    remove_whitespace: false',
        ]),
      },
      {
        label: 'Custom',
        yaml: yaml([
          'metrics:',
          '  - _target_: src.metrics.MyMetrics',
          '    output_key: text',
          '    reference_key: text_ref',
        ]),
      },
    ],
  },
  {
    name: 'Publication',
    desc: 'Pack & upload model and results',
    link: './stages/publish.html',
    tasks: [
      {
        label: 'pack-model',
        yaml: yaml([
          'pack_model:',
          '  out_dir: ${exp_dir}/model_pack',
          '  include:',
          '    - ${recipe_dir}/src',
          '    - ${recipe_dir}/src',
          '  exclude:',
          '    - last.ckpt',

        ]),
      },
      {
        label: 'upload-model',
        yaml: yaml([
          'upload_model:',
          '  repo_id: espnet/my_asr',
        ]),
      },
    ],
  },
  {
    name: 'Demo',
    desc: 'Pack & upload interactive demo',
    link: './stages/demo.html',
    tasks: [
      {
        label: 'inputs',
        yaml: yaml([
          'inputs:',
          '  - name: speech',
          '    type: audio',
          '  - name: prompt',
          '    type: text',
        ]),
      },
      {
        label: 'outputs',
        yaml: yaml([
          'outputs:',
          '  - name: text',
          '    type: textbox',
          '  - name: score',
          '    type: label',
        ]),
      },
    ],
  },
]
</script>
