<template>
  <div class="fm">
    <div class="layout">
      <!-- Files column -->
      <div>
        <div class="col-label">files</div>
        <template v-for="file in files" :key="file.id">
          <div
            v-if="file.isDir"
            class="tree-dir"
          >{{ file.name }}</div>
          <div
            v-else
            class="tree-node"
            :class="[
              file.indent ? `indent${file.indent}` : '',
              fileHighlightClass(file.id),
              activeFile === file.id ? 'active-sel' : '',
            ]"
            @click="selectFile(file.id)"
          >
            <span class="ficon" aria-hidden="true"><HopeIcon :icon="file.icon" /></span>
            <span class="fname">{{ file.name }}</span>
            <span class="fbadge">{{ file.badge }}</span>
          </div>
        </template>
      </div>

      <!-- Arrow -->
      <div class="arrow-col">
        <svg width="40" height="20" viewBox="0 0 40 20" fill="none">
          <line x1="2" y1="10" x2="32" y2="10" stroke="var(--vp-c-divider)" stroke-width="1" />
          <path d="M26 5L34 10L26 15" stroke="var(--vp-c-divider)" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </div>

      <!-- Stages column -->
      <div>
        <div class="col-label">stages</div>
        <div
          v-for="stage in stages"
          :key="stage.id"
          class="stage-node"
          :class="[
            stage.optional ? 'optional-stage' : '',
            activeStage === stage.id ? 'active-sel' : '',
            stageHighlightClass(stage.id),
          ]"
          @click="selectStage(stage.id)"
        >
          <span class="sicon" aria-hidden="true"><HopeIcon :icon="stage.icon" /></span>
          <span class="sname">{{ stage.label }}</span>
          <span v-if="stage.optional" class="opt-label">optional</span>
        </div>
      </div>
    </div>

    <!-- Detail panel -->
    <div class="detail" v-html="detailHtml" />

    <!-- Legend -->
    <div class="legend">
      <div class="leg"><div class="leg-dot" style="background:#9FE1CB" />reads</div>
      <div class="leg"><div class="leg-dot" style="background:#CECBF6" />writes</div>
      <div class="leg"><div class="leg-dot" style="background:#FAC775" />entry point</div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import HopeIcon from "vuepress-theme-hope/components/HopeIcon.js"

const DEFAULT_DETAIL = 'Click a file or stage to see what it connects to.'

const STAGE_MAP = {
  data_prep: {
    reads: ['run_py', 'builder_py'],
    writes: [],
    detail: '<strong>Data prep</strong> — <code>python run.py --stages data_prep</code><br>Runs <code>dataset/builder.py</code> to download and build the dataset. No output written to <code>exp/</code>; the builder manages its own output internally.',
  },
  collect_stats: {
    reads: ['run_py', 'training_yaml', 'dataset_py'],
    writes: ['exp_stats'],
    detail: '<strong>Collect stats</strong> (optional) — reads <code>conf/training.yaml</code> and <code>dataset/dataset.py</code> to collect feature statistics and shape metadata. Writes to <code>exp/{expdir}/stats/</code>.',
  },
  training: {
    reads: ['run_py', 'training_yaml', 'dataset_py', 'src'],
    writes: ['exp_ckpt'],
    detail: '<strong>Training</strong> — PyTorch Lightning loop. Reads <code>conf/training.yaml</code>, <code>dataset/dataset.py</code>, and any custom code in <code>src/</code>. Writes <code>*.ckpt</code> to <code>exp/{expdir}/</code>. Logs are excluded from this view.<br>Multi-GPU, DeepSpeed, and profiling are config-level flags.',
  },
  inference: {
    reads: ['run_py', 'inference_yaml', 'exp_ckpt'],
    writes: ['exp_inference'],
    detail: '<strong>Inference</strong> — provider/runner API handles batching and device placement. Reads <code>conf/inference.yaml</code> and checkpoints from <code>exp/{expdir}/</code>. Writes decoded outputs to <code>exp/{expdir}/inference/</code>.',
  },
  metrics: {
    reads: ['run_py', 'metrics_yaml', 'exp_inference'],
    writes: ['exp_metrics'],
    detail: '<strong>Metrics</strong> — reads <code>conf/metrics.yaml</code> and decoded outputs from <code>exp/{expdir}/inference/</code>. Supports <code>torchmetrics</code> and custom metric modules. Writes scores to <code>exp/{expdir}/metrics/</code>.',
  },
  demo_pub: {
    reads: ['run_py', 'demo_yaml', 'inference_yaml', 'exp_ckpt'],
    writes: ['demo_out'],
    detail: '<strong>Demo / publication</strong> — reads <code>conf/demo.yaml</code>, <code>conf/inference.yaml</code>, and <code>exp/{expdir}/</code>. Auto-generates a Gradio <code>app.py</code> from your <code>InferenceInterface</code> and packs everything into <code>demo/</code> in one command.',
  },
}

const FILE_MAP = {
  run_py:         { stages: { data_prep: 'entry', collect_stats: 'entry', training: 'entry', inference: 'entry', metrics: 'entry', demo_pub: 'entry' }, detail: '<strong>run.py</strong> — entry point for all stages.<br><code>python run.py --stages train infer publish --training_config conf/training.yaml --inference_config conf/inference.yaml</code>' },
  training_yaml:  { stages: { collect_stats: 'read', training: 'read' }, detail: '<strong>conf/training.yaml</strong> — model architecture, optimizer, scheduler, dataset splits. Read by Collect stats and Training.' },
  inference_yaml: { stages: { inference: 'read', demo_pub: 'read' }, detail: '<strong>conf/inference.yaml</strong> — model weights path, decoding params, input/output keys. Read by Inference and Demo/pub.' },
  metrics_yaml:   { stages: { metrics: 'read' }, detail: '<strong>conf/metrics.yaml</strong> — which metrics to compute and how. Read by the Metrics stage.' },
  demo_yaml:      { stages: { demo_pub: 'read' }, detail: '<strong>conf/demo.yaml</strong> — Gradio UI config, requirements, pack.files list. Read by Demo/publication.' },
  builder_py:     { stages: { data_prep: 'read' }, detail: '<strong>dataset/builder.py</strong> — user-written. Handles download, preprocessing, and build. Called by Data prep.' },
  dataset_py:     { stages: { collect_stats: 'read', training: 'read' }, detail: '<strong>dataset/dataset.py</strong> — user-written. Loads the built dataset at training time. Used by Collect stats and Training.' },
  src:            { stages: { training: 'read' }, detail: '<strong>src/*.py</strong> — user-written custom code: system definition, model, components, etc. Loaded by Training.' },
  exp_ckpt:       { stages: { training: 'write', inference: 'read', demo_pub: 'read' }, detail: '<strong>exp/{expdir}/*.ckpt</strong> — checkpoint files written by Training, read by Inference and Demo/pub. Logs are not shown here.' },
  exp_stats:      { stages: { collect_stats: 'write' }, detail: '<strong>exp/{expdir}/stats/</strong> — feature statistics written by Collect stats.' },
  exp_inference:  { stages: { inference: 'write', metrics: 'read' }, detail: '<strong>exp/{expdir}/inference/</strong> — decoded outputs written by Inference, read by Metrics.' },
  exp_metrics:    { stages: { metrics: 'write' }, detail: '<strong>exp/{expdir}/metrics/</strong> — metric scores written by the Metrics stage.' },
  demo_out:       { stages: { demo_pub: 'write' }, detail: '<strong>demo/</strong> — auto-generated Gradio app bundle written by Demo/publication.' },
}

const files = [
  { id: 'run_py',         name: 'run.py',           icon: 'tabler:terminal',        badge: 'entry',     indent: 0 },
  { id: '_conf',          name: 'conf/',             isDir: true },
  { id: 'training_yaml',  name: 'training.yaml',    icon: 'tabler:file-text',       badge: 'config',    indent: 1 },
  { id: 'inference_yaml', name: 'inference.yaml',   icon: 'tabler:file-text',       badge: 'config',    indent: 1 },
  { id: 'metrics_yaml',   name: 'metrics.yaml',     icon: 'tabler:file-text',       badge: 'config',    indent: 1 },
  { id: 'demo_yaml',      name: 'demo.yaml',        icon: 'tabler:file-text',       badge: 'config',    indent: 1 },
  { id: '_dataset',       name: 'dataset/',          isDir: true },
  { id: 'builder_py',     name: 'builder.py',       icon: 'tabler:brand-python',    badge: 'user',      indent: 1 },
  { id: 'dataset_py',     name: 'dataset.py',       icon: 'tabler:brand-python',    badge: 'user',      indent: 1 },
  { id: '_src',           name: 'src/',              isDir: true },
  { id: 'src',            name: '*.py (custom)',    icon: 'tabler:brand-python',    badge: 'user',      indent: 1 },
  { id: '_exp',           name: 'exp/{expdir}/',     isDir: true },
  { id: 'exp_ckpt',       name: '*.ckpt',           icon: 'tabler:device-floppy',   badge: 'generated', indent: 1 },
  { id: 'exp_stats',      name: 'stats/',           icon: 'tabler:chart-histogram', badge: 'generated', indent: 1 },
  { id: 'exp_inference',  name: 'inference/',       icon: 'tabler:folder',          badge: 'generated', indent: 1 },
  { id: 'exp_metrics',    name: 'metrics/',         icon: 'tabler:folder',          badge: 'generated', indent: 1 },
  { id: '_demo',          name: 'demo/',             isDir: true },
  { id: 'demo_out',       name: 'app.py + bundle',  icon: 'tabler:device-desktop',  badge: 'generated', indent: 1 },
]

const stages = [
  { id: 'data_prep',     label: 'Data prep',          icon: 'tabler:database',        optional: false },
  { id: 'collect_stats', label: 'Collect stats',      icon: 'tabler:chart-histogram', optional: true  },
  { id: 'training',      label: 'Training',           icon: 'tabler:bolt',            optional: false },
  { id: 'inference',     label: 'Inference',          icon: 'tabler:cpu',             optional: false },
  { id: 'metrics',       label: 'Metrics',            icon: 'tabler:chart-bar',       optional: false },
  { id: 'demo_pub',      label: 'Demo / publication', icon: 'tabler:device-desktop',  optional: false },
]

const activeFile = ref(null)
const activeStage = ref('training')
const detailHtml = ref(STAGE_MAP.training.detail)

function selectFile(fid) {
  if (activeFile.value === fid) {
    activeFile.value = null
    activeStage.value = null
    detailHtml.value = DEFAULT_DETAIL
    return
  }
  activeFile.value = fid
  activeStage.value = null
  detailHtml.value = FILE_MAP[fid]?.detail ?? DEFAULT_DETAIL
}

function selectStage(sid) {
  if (activeStage.value === sid) {
    activeFile.value = null
    activeStage.value = null
    detailHtml.value = DEFAULT_DETAIL
    return
  }
  activeStage.value = sid
  activeFile.value = null
  detailHtml.value = STAGE_MAP[sid]?.detail ?? DEFAULT_DETAIL
}

function fileHighlightClass(fid) {
  if (!FILE_MAP[fid]) return ''
  if (activeFile.value) return ''
  if (activeStage.value) {
    const sm = STAGE_MAP[activeStage.value]
    if (!sm) return ''
    if (fid === 'run_py' && sm.reads.includes('run_py')) return 'hl-entry'
    if (sm.reads.includes(fid)) return 'hl-read'
    if (sm.writes.includes(fid)) return 'hl-write'
  }
  return ''
}

function stageHighlightClass(sid) {
  if (activeStage.value) return ''
  if (activeFile.value) {
    const fm = FILE_MAP[activeFile.value]
    if (!fm) return ''
    return fm.stages[sid] ? 'hl-stage' : ''
  }
  return ''
}
</script>

<style scoped>
.fm { padding: 1.5rem 0; font-size: 13px; }

.col-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--vp-c-text-2);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 8px;
}

.layout {
  display: grid;
  grid-template-columns: 1fr 56px 1fr;
  align-items: start;
}

.tree-dir {
  font-size: 11px;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-base);
  padding: 4px 8px 2px;
  margin-top: 4px;
}

.tree-node {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  margin: 2px 0;
  border-radius: 6px;
  cursor: pointer;
  border: 0.5px solid transparent;
  transition: background 0.15s, border-color 0.15s;
  white-space: nowrap;
  overflow: hidden;
}
.tree-node:hover { background: var(--vp-c-bg-soft); }

.ficon {
  font-size: 14px;
  color: var(--vp-c-text-2);
  flex-shrink: 0;
  display: flex;
  align-items: center;
}

.fname {
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  color: var(--vp-c-text-1);
  overflow: hidden;
  text-overflow: ellipsis;
}
.fbadge {
  margin-left: auto;
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 4px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  flex-shrink: 0;
  font-family: var(--vp-font-family-base);
}

.indent1 { padding-left: 20px; }
.indent2 { padding-left: 36px; }

.hl-read  { background: #E1F5EE; border-color: #1D9E75; }
.hl-read .fname  { color: #085041; }
.hl-read .fbadge { background: #9FE1CB; color: #085041; }
.hl-read .ficon  { color: #1D9E75; }

.hl-write  { background: #EEEDFE; border-color: #7F77DD; }
.hl-write .fname  { color: #3C3489; }
.hl-write .fbadge { background: #CECBF6; color: #3C3489; }
.hl-write .ficon  { color: #7F77DD; }

.hl-entry  { background: #FAEEDA; border-color: #BA7517; }
.hl-entry .fname  { color: #633806; }
.hl-entry .fbadge { background: #FAC775; color: #633806; }
.hl-entry .ficon  { color: #BA7517; }

.active-sel { background: #dbeafe; border-color: #3b82f6; }
.active-sel .fname { color: #1e40af; }
.active-sel .ficon { color: #3b82f6; }

.arrow-col {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 28px;
}

.stage-node {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  margin: 2px 0;
  border-radius: 6px;
  cursor: pointer;
  border: 0.5px solid #e0e0e0;
  background: var(--vp-c-bg-soft);
  transition: background 0.15s, border-color 0.15s;
}
.stage-node:hover { border-color: var(--vp-c-divider); }
.optional-stage { border-style: dashed; }

.sname { font-family: var(--vp-font-family-base); font-size: 12px; font-weight: 500; color: var(--vp-c-text-1); }
.sicon { font-size: 15px; color: var(--vp-c-text-2); display: flex; align-items: center; }
.opt-label { margin-left: auto; font-size: 10px; color: var(--vp-c-text-3); font-family: var(--vp-font-family-base); }

.hl-stage { background: #EEEDFE; border-color: #7F77DD; }
.hl-stage .sname { color: #3C3489; }
.hl-stage .sicon { color: #534AB7; }

.stage-node.active-sel .sname { color: #1e40af; }
.stage-node.active-sel .sicon { color: #3b82f6; }

.detail {
  margin-top: 1rem;
  padding: 10px 14px;
  border-radius: 8px;
  border: 0.5px solid #e0e0e0;
  background: var(--vp-c-bg-soft);
  font-family: var(--vp-font-family-base);
  font-size: 12px;
  line-height: 1.6;
  min-height: 56px;
  color: var(--vp-c-text-2);
}
.detail :deep(strong) { color: var(--vp-c-text-1); font-weight: 500; }
.detail :deep(code) {
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  background: var(--vp-c-bg);
  padding: 1px 4px;
  border-radius: 3px;
  border: 0.5px solid #e0e0e0;
}

.legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
.leg { display: flex; align-items: center; gap: 4px; font-family: var(--vp-font-family-base); font-size: 11px; color: var(--vp-c-text-2); }
.leg-dot { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }
</style>