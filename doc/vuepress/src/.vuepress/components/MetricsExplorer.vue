<template>
  <div class="fm">

    <p class="subtitle">Click a file or stage to highlight reads / writes relationships.</p>

    <div class="layout">

      <!-- Files -->
      <div>
        <div class="col-label">files</div>

        <!-- run.py -->
        <div
          class="tree-node"
          :class="fileNodeClass('run_py')"
          @click="selectFile('run_py')"
        >
          <span class="ficon">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m5 7l5 5l-5 5m7 2h7"/>
            </svg>
          </span>
          <span class="fname">run.py</span>
          <span class="fbadge">entry</span>
        </div>

        <!-- conf/ -->
        <div class="tree-dir" :class="{ open: openDirs.conf }" @click="toggleDir('conf')">
          <span class="dir-chevron">▶</span>📁 conf/
        </div>
        <div class="dir-children" :class="{ open: openDirs.conf }">
          <div
            class="tree-node indent1"
            :class="fileNodeClass('metrics_yaml')"
            @click="selectFile('metrics_yaml')"
          >
            <span class="ficon">≡</span>
            <span class="fname">metrics.yaml</span>
            <span class="fbadge">config</span>
          </div>
        </div>

        <!-- exp/{expdir}/inference/ -->
        <div class="tree-dir" :class="{ open: openDirs.inference }" @click="toggleDir('inference')">
          <span class="dir-chevron">▶</span>📁 exp/{expdir}/inference/
        </div>
        <div class="dir-children" :class="{ open: openDirs.inference }">

          <!-- test_1/ -->
          <div class="tree-dir indent1" :class="{ open: openDirs.test1 }" @click.stop="toggleDir('test1')">
            <span class="dir-chevron">▶</span>📁 test_1/
          </div>
          <div class="dir-children" :class="{ open: openDirs.test1 }">

            <div
              class="tree-node indent2"
              :class="fileNodeClass('ref_wav_scp')"
              @click.stop="selectFile('ref_wav_scp')"
            >
              <span class="ficon">≡</span>
              <span class="fname">ref_wav.scp</span>
              <span class="fbadge">required</span>
            </div>

            <div
              class="tree-node indent2"
              :class="fileNodeClass('hyp_wav_scp')"
              @click.stop="selectFile('hyp_wav_scp')"
            >
              <span class="ficon">≡</span>
              <span class="fname">hyp_wav.scp</span>
              <span class="fbadge">required</span>
            </div>

            <!-- ref_wav/ -->
            <div class="tree-dir indent2" :class="{ open: openDirs.ref_wav }" @click.stop="toggleDir('ref_wav')">
              <span class="dir-chevron">▶</span>📁 ref_wav/
            </div>
            <div class="dir-children" :class="{ open: openDirs.ref_wav }">
              <div class="tree-node indent3 no-click"><span class="ficon">♪</span><span class="fname">utt001.wav</span></div>
              <div class="tree-node indent3 no-click"><span class="ficon">♪</span><span class="fname">utt002.wav</span></div>
            </div>

            <!-- hyp_wav/ -->
            <div class="tree-dir indent2" :class="{ open: openDirs.hyp_wav }" @click.stop="toggleDir('hyp_wav')">
              <span class="dir-chevron">▶</span>📁 hyp_wav/
            </div>
            <div class="dir-children" :class="{ open: openDirs.hyp_wav }">
              <div class="tree-node indent3 no-click"><span class="ficon">♪</span><span class="fname">utt001.wav</span></div>
              <div class="tree-node indent3 no-click"><span class="ficon">♪</span><span class="fname">utt002.wav</span></div>
            </div>

          </div><!-- /test_1 children -->

          <!-- test_2/ -->
          <div class="tree-dir indent1" :class="{ open: openDirs.test2 }" @click.stop="toggleDir('test2')">
            <span class="dir-chevron">▶</span>📁 test_2/
          </div>
          <div class="dir-children" :class="{ open: openDirs.test2 }"></div>

          <!-- metrics.json -->
          <div
            class="tree-node indent1"
            :class="fileNodeClass('metrics_json')"
            @click="selectFile('metrics_json')"
          >
            <span class="ficon">{}</span>
            <span class="fname">metrics.json</span>
            <span class="fbadge">generated</span>
          </div>

        </div><!-- /inference children -->
      </div><!-- /files col -->

      <!-- Arrow -->
      <div class="arrow-col">
        <svg width="40" height="20" viewBox="0 0 40 20" fill="none">
          <line x1="2" y1="10" x2="32" y2="10" stroke="#d1d5db" stroke-width="1"/>
          <path d="M26 5L34 10L26 15" stroke="#d1d5db" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <!-- Stages -->
      <div>
        <div class="col-label">stages</div>
        <template v-for="(stage, i) in stages" :key="stage.id">
          <div
            class="stage-node"
            :class="stageNodeClass(stage.id)"
            @click="selectStage(stage.id)"
          >
            <span class="stage-num">{{ i + 1 }}</span>
            <div class="stage-body">
              <div class="sname">{{ stage.label }}</div>
              <div class="ssub">{{ stage.sub }}</div>
            </div>
          </div>
          <div v-if="i < stages.length - 1" class="stage-connector" />
        </template>
      </div>

    </div><!-- /layout -->

    <!-- Detail -->
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
import { ref, reactive, computed } from 'vue'

// ── data ───────────────────────────────────────────────────────────────────

const DEFAULT_DETAIL = 'Click a file or stage to see what it connects to.'

const STAGE_MAP = {
  load_metrics_config: {
    reads: ['run_py', 'metrics_yaml'],
    writes: [],
    detail: '<strong>Load metrics config</strong> — <code>python run.py --stages measure --metrics_config conf/metrics.yaml</code><br>Reads <code>run.py</code> and <code>conf/metrics.yaml</code> to resolve metric classes, key aliases, and <code>inference_dir</code>.',
  },
  resolve_test_sets: {
    reads: ['metrics_yaml'],
    writes: [],
    detail: '<strong>Resolve test sets</strong> — uses <code>dataset.test</code> from <code>conf/metrics.yaml</code> when present. Otherwise scans <code>exp/{expdir}/inference/</code> and treats each subdirectory (<code>test_1/</code>, <code>test_2/</code>, …) as one evaluation set.',
  },
  compute_metrics: {
    reads: ['metrics_yaml', 'ref_wav_scp', 'hyp_wav_scp'],
    writes: [],
    detail: '<strong>Compute metrics</strong> — builds a dict of SCP paths (<code>ref_wav.scp</code>, <code>hyp_wav.scp</code>) and passes them into each metric class. The metric opens the SCPs, aligns utterance IDs, and returns JSON-serializable scores.',
  },
  write_results: {
    reads: ['metrics_yaml'],
    writes: ['metrics_json'],
    detail: '<strong>Write metrics.json</strong> — collects results for each metric class × each evaluation set, then writes <code>exp/{expdir}/inference/metrics.json</code>.',
  },
}

const FILE_MAP = {
  run_py: {
    stages: { load_metrics_config: 'entry' },
    detail: '<strong>run.py</strong> — entry point for the measure stage.<br><code>python run.py --stages measure --metrics_config conf/metrics.yaml</code>',
  },
  metrics_yaml: {
    stages: { load_metrics_config: 'read', resolve_test_sets: 'read', compute_metrics: 'read', write_results: 'read' },
    detail: `<strong>conf/metrics.yaml</strong> — defines metric classes, input key aliases (<code>ref_wav</code>, <code>hyp_wav</code>), and optionally the evaluation set list. Read at every measure step.<pre class="code-block">exp_tag: my_exp
exp_dir: exp/\${exp_tag}
inference_dir: \${exp_dir}/inference

metrics:
  - metric:
      _target_: espnet3.systems.tts.metrics.mcd.MCD
    inputs:
      ref_wav: ref_wav
      hyp_wav: hyp_wav
  - metric:
      _target_: espnet3.systems.tts.metrics.utmos.UTMOS
    inputs:
      hyp_wav: hyp_wav</pre>`,
  },
  ref_wav_scp: {
    stages: { compute_metrics: 'read' },
    detail: '<strong>ref_wav.scp</strong> — reference waveform SCP. Each line: <code>utt_id  path/to/wav</code>. Passed as a path to the metric class; the metric opens and reads it.<pre class="code-block">utt001  test_1/ref_wav/utt001.wav\nutt002  test_1/ref_wav/utt002.wav</pre>',
  },
  hyp_wav_scp: {
    stages: { compute_metrics: 'read' },
    detail: '<strong>hyp_wav.scp</strong> — generated waveform SCP. Each line: <code>utt_id  path/to/wav</code>. Passed alongside <code>ref_wav.scp</code>; utterance IDs are aligned by the metric.<pre class="code-block">utt001  test_1/hyp_wav/utt001.wav\nutt002  test_1/hyp_wav/utt002.wav</pre>',
  },
  metrics_json: {
    stages: { write_results: 'write' },
    detail: '<strong>metrics.json</strong> — final output of the measure stage. Results nested by metric class name and evaluation set name.<br>Example: <code>{"...MCD": {"test_1": {"MCD": 4.82}, "test_2": {"MCD": 5.01}}}</code>',
  },
}

const stages = [
  { id: 'load_metrics_config', label: 'Load metrics config',  sub: 'read run.py + metrics.yaml'      },
  { id: 'resolve_test_sets',   label: 'Resolve test sets',    sub: 'scan inference/ subdirs'          },
  { id: 'compute_metrics',     label: 'Compute metrics',      sub: 'pass SCP paths to metric class'   },
  { id: 'write_results',       label: 'Write metrics.json',   sub: 'serialize results to JSON'        },
]

// ── state ──────────────────────────────────────────────────────────────────

const activeFile  = ref(null)
const activeStage = ref(null)
const detailHtml  = ref(DEFAULT_DETAIL)

const openDirs = reactive({
  conf:      true,
  inference: true,
  test1:     true,
  test2:     true,
  ref_wav:   true,
  hyp_wav:   true,
})

// ── actions ────────────────────────────────────────────────────────────────

function toggleDir(key) {
  openDirs[key] = !openDirs[key]
}

function selectFile(fid) {
  if (activeFile.value === fid) { reset(); return }
  activeFile.value  = fid
  activeStage.value = null
  detailHtml.value  = FILE_MAP[fid]?.detail ?? DEFAULT_DETAIL
}

function selectStage(sid) {
  if (activeStage.value === sid) { reset(); return }
  activeStage.value = sid
  activeFile.value  = null
  detailHtml.value  = STAGE_MAP[sid]?.detail ?? DEFAULT_DETAIL
}

function reset() {
  activeFile.value  = null
  activeStage.value = null
  detailHtml.value  = DEFAULT_DETAIL
}

// ── class helpers ──────────────────────────────────────────────────────────

function fileNodeClass(fid) {
  if (activeFile.value === fid) return { 'active-sel': true }
  if (activeFile.value) return {}
  if (activeStage.value) {
    const sm = STAGE_MAP[activeStage.value]
    if (!sm) return {}
    if (fid === 'run_py' && sm.reads.includes('run_py')) return { 'hl-entry': true }
    if (sm.reads.includes(fid))  return { 'hl-read':  true }
    if (sm.writes.includes(fid)) return { 'hl-write': true }
  }
  return {}
}

function stageNodeClass(sid) {
  if (activeStage.value === sid) return { 'active-sel': true }
  if (activeStage.value) return {}
  if (activeFile.value) {
    const fm = FILE_MAP[activeFile.value]
    if (fm?.stages[sid]) return { 'hl-stage': true }
  }
  return {}
}
</script>

<style scoped>
.fm {
  padding: 1.5rem 0;
  font-size: 13px;
  font-family: var(--vp-font-family-base);
}

.subtitle { font-size: 12px; color: var(--vp-c-text-2); margin-bottom: 1.5rem; }

.layout {
  display: grid;
  grid-template-columns: 1fr 56px 1fr;
  align-items: start;
}

.col-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--vp-c-text-2);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 8px;
}

/* ── accordion dir ── */
.tree-dir {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  padding: 5px 8px;
  margin-top: 6px;
  border-radius: 6px;
  cursor: pointer;
  user-select: none;
  transition: background 0.15s;
}
.tree-dir:hover { background: var(--vp-c-bg-soft); }

.dir-chevron {
  font-size: 10px;
  color: var(--vp-c-text-3);
  transition: transform 0.2s;
  flex-shrink: 0;
}
.tree-dir.open .dir-chevron { transform: rotate(90deg); }

.dir-children {
  overflow: hidden;
  max-height: 0;
  transition: max-height 0.25s ease;
}
.dir-children.open { max-height: 1000px; }

/* ── tree nodes ── */
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
.tree-node.no-click { cursor: default; }
.tree-node.no-click:hover { background: transparent; }

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
.indent3 { padding-left: 52px; }

/* ── highlights ── */
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

/* ── arrow ── */
.arrow-col {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 28px;
}

/* ── stage nodes ── */
.stage-node {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 4px 0;
  margin: 2px 0;
  cursor: pointer;
}
.stage-num {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 600;
  color: var(--vp-c-text-2);
  flex-shrink: 0;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.stage-connector {
  width: 1px;
  background: var(--vp-c-divider);
  margin: 0 0 0 10px;
  height: 10px;
}
.stage-body { padding-top: 2px; padding-bottom: 8px; }
.sname { font-size: 12px; font-weight: 500; color: var(--vp-c-text-1); line-height: 1.4; }
.ssub  { font-size: 11px; color: var(--vp-c-text-2); margin-top: 2px; line-height: 1.5; }

.hl-stage .stage-num { background: #EEEDFE; border-color: #7F77DD; color: #534AB7; }
.hl-stage .sname { color: #3C3489; }
.hl-stage .ssub  { color: #7F77DD; }

.stage-node.active-sel .stage-num { background: #dbeafe; border-color: #3b82f6; color: #1e40af; }
.stage-node.active-sel .sname { color: #1e40af; }

/* ── detail ── */
.detail {
  margin-top: 1rem;
  padding: 10px 14px;
  border-radius: 8px;
  border: 0.5px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
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
  border: 0.5px solid var(--vp-c-divider);
  color: var(--vp-c-text-1);
}
.detail :deep(.code-block) {
  margin-top: 8px;
  background: var(--vp-c-bg);
  border: 0.5px solid var(--vp-c-divider);
  border-radius: 5px;
  padding: 8px 10px;
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  line-height: 1.8;
  white-space: pre;
  overflow-x: auto;
}

/* ── legend ── */
.legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
.leg { display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--vp-c-text-2); }
.leg-dot { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }

@media (max-width: 680px) {
  .layout { grid-template-columns: 1fr; gap: 12px; }
  .arrow-col { display: none; }
}
</style>
