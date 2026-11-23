<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'

// ── Config (drives the diagram) ───────────────────────────────────────────────
const env         = ref('local')  // 'local' | 'local_gpu' | 'slurm'
const nWorkers    = ref(3)        // 1-6
const batchSlider = ref(0)        // 0 = null, 1-8 = batch_size value
const batchSize   = computed(() => batchSlider.value === 0 ? null : batchSlider.value)

// local env uses build_env_local() + single process; others use build_worker_setup_fn() + Dask
const isLocal = computed(() => env.value === 'local')

// ── Phases — depend on execution env ─────────────────────────────────────────
const PHASES_LOCAL = [
  { id: 'build',   label: '① Provider calls build_env_local() — env dict built on driver', accent: '--provider' },
  { id: 'process', label: '② BaseRunner loops over all indices, calling forward() sequentially', accent: '--forward' },
  { id: 'done',    label: '③ Embeddings written — output_dir/{utt_id}.pt',                   accent: '--done'     },
]
const PHASES_DIST = [
  { id: 'build',    label: '① Provider builds the setup() closure',                accent: '--provider' },
  { id: 'ship',     label: '② setup() closure sent to BaseRunner (driver)',         accent: '--forward'  },
  { id: 'init',     label: '③ BaseRunner distributes setup() — workers initialize', accent: '--worker'   },
  { id: 'dispatch', label: '④ BaseRunner dispatches idx to each worker\'s forward()', accent: '--driver' },
  { id: 'process',  label: '⑤ forward() runs — load → extract → save',             accent: '--forward'  },
  { id: 'done',     label: '⑥ Embeddings written — output_dir/{utt_id}.pt',         accent: '--done'     },
]
const DURATIONS_LOCAL = [2000, 3200, 2000]
const DURATIONS_DIST  = [1800, 2200, 2000, 2400, 3200, 2000]

const PHASES    = computed(() => isLocal.value ? PHASES_LOCAL : PHASES_DIST)
const DURATIONS = computed(() => isLocal.value ? DURATIONS_LOCAL : DURATIONS_DIST)

const TOTAL_UTTS = 30

// Derived workers list — reacts to nWorkers
const workers = computed(() => {
  const n   = nWorkers.value
  const per = Math.floor(TOTAL_UTTS / n)
  return Array.from({ length: n }, (_, i) => {
    const start = i * per
    const end   = i === n - 1 ? TOTAL_UTTS - 1 : (i + 1) * per - 1
    return { id: i, label: `worker ${i + 1}`, start, end, range: `[${start} .. ${end}]` }
  })
})

// How idx appears in dispatch chips — single int or batch list
function idxChips(w) {
  const bs = batchSize.value
  if (!bs) return [w.start, w.start + 1, w.start + 2, '…']
  const b0 = Array.from({ length: Math.min(bs, 4) }, (_, j) => w.start + j)
  return [`[${b0.join(', ')}]`, `[…]`]
}

// forward() idx annotation changes with batch_size
const idxAnnotation = computed(() =>
  batchSize.value ? `list[int]  # batch of ${batchSize.value}` : 'int'
)

const step1Text = computed(() =>
  batchSize.value ? `for i in idx: load utterances[i]` : `load utterances[idx]`
)
const step1Code = computed(() =>
  batchSize.value ? `utterances[i]` : `utterances[idx]`
)
const step1ArgCls = computed(() => 'inject-auto')

// Live YAML preview
const yamlConfig = computed(() => {
  const bs  = batchSize.value ? `\n  batch_size: ${batchSize.value}` : ''
  const nw = isLocal.value
    ? `  n_workers: ${nWorkers.value}  # ignored in local mode (1 shard)`
    : `  n_workers: ${nWorkers.value}`
  const opts = env.value === 'local_gpu'
    ? `  options: {}  # dask_cuda.LocalCUDACluster`
    : env.value === 'slurm'
    ? `  options:\n    queue: gpu\n    # … other SLURM options`
    : `  options: {}`
  return `parallel:\n  env: ${env.value}\n${nw}\n${opts}\n\nrunner:${bs || '\n  batch_size: null  # int idx per call'}`
})

// ── Animation state ───────────────────────────────────────────────────────────
const phaseIdx = ref(0)
const playing  = ref(true)
const procStep = ref(0)

let mainTimer = null
let procTimer = null

const phase = computed(() => PHASES.value[phaseIdx.value]?.id)

function scheduleNext() {
  clearTimeout(mainTimer)
  const isLast = phaseIdx.value === PHASES.value.length - 1
  if (isLast) { playing.value = false; return }
  mainTimer = setTimeout(() => {
    phaseIdx.value++
    if (playing.value) scheduleNext()
  }, DURATIONS.value[phaseIdx.value] ?? 2000)
}

function togglePlay() {
  // At the last phase, replay from the beginning
  if (!playing.value && phaseIdx.value === PHASES.value.length - 1) {
    phaseIdx.value = 0
    playing.value = true
    scheduleNext()
    return
  }
  playing.value = !playing.value
  if (playing.value) scheduleNext()
  else clearTimeout(mainTimer)
}

function goTo(i) {
  clearTimeout(mainTimer)
  phaseIdx.value = Math.max(0, Math.min(i, PHASES.value.length - 1))
  playing.value = phaseIdx.value < PHASES.value.length - 1
  if (playing.value) scheduleNext()
}

watch(phaseIdx, () => {
  clearInterval(procTimer)
  procStep.value = 0
  if (phase.value === 'process') {
    let s = 0
    procTimer = setInterval(() => { s++; procStep.value = s; if (s >= 3) clearInterval(procTimer) }, 800)
  }
})

// restart animation when any config changes
watch([env, nWorkers, batchSlider], () => {
  goTo(0)
})

onMounted(()  => { if (playing.value) scheduleNext() })
onUnmounted(() => { clearTimeout(mainTimer); clearInterval(procTimer) })

const envKeys = [
  { k: '"model"',      v: 'EncoderClassifier(…)' },
  { k: '"toolkit"',    v: '"speechbrain"'          },
  { k: '"utterances"', v: '[(utt_id, path), …]'   },
  { k: '"output_dir"', v: 'Path("data/xvec")'      },
]

const fwdArgs = [
  { name: 'idx',        cls: 'inject-auto'  },
  { name: 'model',      cls: 'inject-match' },
  { name: 'toolkit',    cls: 'inject-match' },
  { name: 'utterances', cls: 'inject-match' },
  { name: 'output_dir', cls: 'inject-match' },
]

const ptFiles = computed(() =>
  [...Array.from({ length: Math.min(nWorkers.value * 2, 5) }, (_, i) => `utt_${String(i).padStart(4,'0')}.pt`), '…']
)
</script>

<template>
  <section class="guide-section xv-anim">
    <div class="container">

      <!-- ══════════════════════════════════════════════════════════════════
           CONFIG PANEL  — controls the animation below
      ══════════════════════════════════════════════════════════════════ -->
      <div class="config-grid cfg-panel">

        <!-- Controls -->
        <div class="config-controls">
          <div class="config-section-label">Parallel config</div>

          <div class="config-field">
            <label class="config-label">
              env
              <span>execution backend</span>
            </label>
            <select v-model="env">
              <option value="local">local — single process on driver</option>
              <option value="local_gpu">local_gpu — one Dask worker per GPU</option>
              <option value="slurm">slurm — one Dask worker per SLURM job</option>
            </select>
          </div>

          <!-- local mode note -->
          <Transition name="cfg-fade">
            <div v-if="isLocal" class="callout callout-info" style="margin-top:.5rem;margin-bottom:.5rem">
              <span class="callout-icon">ℹ</span>
              <div>
                <strong>env: local</strong> — BaseRunner calls
                <code>provider.build_env_local()</code> once on the driver,
                then processes all indices sequentially in a single process.
                <code>n_workers</code> is ignored.
              </div>
            </div>
          </Transition>

          <div class="config-field" :class="{ 'cfg-field-dim': isLocal }">
            <label class="config-label">
              n_workers
              <span>{{
                isLocal          ? 'ignored in local mode' :
                env === 'local_gpu' ? 'number of GPUs to use' :
                                   'number of SLURM jobs'
              }}</span>
            </label>
            <div class="cfg-slider-row">
              <input
                type="range" v-model.number="nWorkers"
                min="1" max="6" step="1"
                class="cfg-slider"
                :disabled="isLocal"
              />
              <span class="cfg-slider-val" :class="isLocal ? 'cfg-val-null' : 'inject-auto'">
                {{ isLocal ? '—' : nWorkers }}
              </span>
            </div>
          </div>

          <div class="config-divider" />
          <div class="config-section-label">Runner config</div>

          <div class="config-field">
            <label class="config-label">
              batch_size
              <span>indices per forward() call</span>
            </label>
            <div class="cfg-slider-row">
              <input
                type="range" v-model.number="batchSlider"
                min="0" max="8" step="1"
                class="cfg-slider"
              />
              <span
                class="cfg-slider-val"
                :class="batchSize ? 'inject-auto' : 'cfg-val-null'"
              >{{ batchSize ?? 'null' }}</span>
            </div>
          </div>

          <!-- batch_size explanation -->
          <Transition name="cfg-fade">
            <div v-if="batchSize" class="callout callout-info" style="margin-top:.75rem;margin-bottom:0">
              <span class="callout-icon">ℹ</span>
              <div>
                <strong>batch_size = {{ batchSize }}</strong> — BaseRunner groups indices into
                lists of {{ batchSize }} before calling <code>forward()</code>.
                The <code>idx</code> argument becomes <code>list[int]</code>
                instead of <code>int</code>. Your <code>forward()</code> must handle both.
              </div>
            </div>
          </Transition>
        </div>

        <!-- Live YAML -->
        <div class="config-output">
          <div class="config-output-header">
            <span>parallel_config.yaml</span>
            <span style="color: var(--worker); font-size: 11px">● live</span>
          </div>
          <pre><code>{{ yamlConfig }}</code></pre>
        </div>
      </div>

      <!-- ══════════════════════════════════════════════════════════════════
           ANIMATION
      ══════════════════════════════════════════════════════════════════ -->

      <!-- Phase progress -->
      <div class="phase-track">
        <button
          v-for="(p, i) in PHASES" :key="i"
          class="phase-pip"
          :class="{ active: phaseIdx === i, past: phaseIdx > i }"
          @click="goTo(i)"
        />
      </div>
      <div class="phase-label-wrap">
        <Transition name="label-fade" mode="out-in">
          <span
            :key="phaseIdx"
            class="phase-label"
            :style="{ color: `var(${PHASES[phaseIdx]?.accent})` }"
          >{{ PHASES[phaseIdx]?.label }}</span>
        </Transition>
      </div>

      <!-- Main grid: Provider | h-channel | Runner+Workers -->
      <div class="xv-grid">

        <!-- ── LEFT: Provider ────────────────────────────────────────────── -->
        <div class="xv-box xv-provider" :class="{ 'box-provider': phase === 'build' || phase === 'ship' }">
          <div class="xv-box-title" style="color: var(--provider)">X-vector Provider</div>

          <!-- local: build_env_local() returns dict directly -->
          <template v-if="isLocal">
            <div class="xv-method-header" style="color:var(--provider);border-color:var(--provider-border);background:var(--provider-bg)">
              build_env_local()
            </div>
            <div class="xv-setup-body" style="border-color:var(--provider-border)">
              <div class="xv-fn-sig" style="color:var(--provider)">returns {</div>
              <div
                v-for="(e, i) in envKeys" :key="e.k"
                class="xv-env-row"
                :class="{ visible: phase !== 'build' }"
                :style="phase === 'build' ? { animationDelay: `${i * 0.18}s` } : {}"
              >
                <span class="inject-match xv-key">{{ e.k }}</span>
                <span class="xv-colon">:</span>
                <span class="xv-val">{{ e.v }}</span>
              </div>
              <div class="xv-fn-close" style="color:var(--provider)">}</div>
            </div>
            <Transition name="pop">
              <div v-if="phase !== 'build'" class="xv-ready-badge box-done">
                <span style="color:var(--done)">✓ env dict ready</span>
              </div>
            </Transition>
          </template>

          <!-- distributed: build_worker_setup_fn() returns a closure -->
          <template v-else>
            <div class="xv-method-header">build_worker_setup_fn()</div>
            <div class="xv-setup-body box-forward">
              <div class="xv-fn-sig" style="color:var(--forward)">def setup(): return {</div>
              <div
                v-for="(e, i) in envKeys" :key="e.k"
                class="xv-env-row"
                :class="{ visible: phase !== 'build' }"
                :style="phase === 'build' ? { animationDelay: `${i * 0.18}s` } : {}"
              >
                <span class="inject-match xv-key">{{ e.k }}</span>
                <span class="xv-colon">:</span>
                <span class="xv-val">{{ e.v }}</span>
              </div>
              <div class="xv-fn-close" style="color:var(--forward)">}</div>
            </div>
            <Transition name="pop">
              <div v-if="phase !== 'build'" class="xv-ready-badge box-done">
                <span style="color:var(--done)">✓ closure ready</span>
              </div>
            </Transition>
          </template>
        </div>

        <!-- ── CENTER: Horizontal channel ────────────────────────────────── -->
        <!-- local:  env dict travels directly (provider color)            -->
        <!-- dist:   setup() closure travels (forward color)               -->
        <div class="xv-h-channel">
          <div class="xv-h-line" :class="{ lit: phase !== 'build' }"></div>
          <div class="xv-h-tip"  :class="{ lit: phase !== 'build' }">›</div>
          <div class="xv-h-label" :class="{ visible: phase !== 'build' }">
            {{ isLocal ? 'env dict' : 'setup()' }}
          </div>
          <!-- local: env dict packet -->
          <div v-if="isLocal && phase === 'build'" :key="`local-build-${phaseIdx}`" class="xv-h-packet pkt-provider">
            env {}
          </div>
          <!-- distributed: setup() closure packet -->
          <div v-if="!isLocal && phase === 'ship'" :key="`ship-${phaseIdx}`" class="xv-h-packet pkt-forward">
            setup()
          </div>
        </div>

        <!-- ── RIGHT: BaseRunner + Workers ───────────────────────────────── -->
        <div class="xv-right-col">

          <!-- BaseRunner (driver) -->
          <div
            class="xv-runner-box"
            :class="{
              'box-driver':  phase === 'ship' || phase === 'dispatch',
              'box-worker':  phase === 'init',
              'box-forward': phase === 'process',
            }"
          >
            <div class="xv-runner-head">
              <span class="xv-box-title" style="color:var(--driver)">BaseRunner (driver)</span>
              <Transition name="pop">
                <span v-if="phase === 'dispatch' || phase === 'process' || phase === 'done'" class="xv-runner-badge">
                  dispatches idx
                </span>
              </Transition>
            </div>
            <div class="xv-runner-body">
              <div v-if="phase === 'build'" class="xv-runner-idle">waiting for setup()…</div>
              <template v-else>
                <div class="xv-runner-row">
                  <span class="step-dot dot-worker" style="width:14px;height:14px;font-size:9px">✓</span>
                  <span style="color:var(--text2);font-size:11px">env from Provider ready</span>
                </div>
                <Transition name="slide-down">
                  <div v-if="phase === 'dispatch' || phase === 'process' || phase === 'done'" class="xv-runner-row" style="margin-top:3px">
                    <span class="inject-auto" style="padding:1px 5px;font-size:11px">idx</span>
                    <span style="color:var(--text3);font-size:11px">
                      = <template v-if="!batchSize">0, 1, 2, … {{ TOTAL_UTTS - 1 }}</template>
                        <template v-else>[0…{{ batchSize - 1 }}], [{{ batchSize }}…{{ batchSize*2 - 1 }}], …</template>
                    </span>
                    <!-- batch size badge -->
                    <span v-if="batchSize" class="xv-batch-badge">batch={{ batchSize }}</span>
                  </div>
                </Transition>
              </template>
            </div>
          </div>

          <!-- ── LOCAL: sequential forward() loop on driver ─────────────── -->
          <template v-if="isLocal">
            <Transition name="slide-down">
              <div v-if="phase === 'process' || phase === 'done'" class="xv-local-loop">
                <div class="xv-local-label">
                  <span class="step-dot dot-worker" style="width:14px;height:14px;font-size:9px">✓</span>
                  <span style="font-size:11px;color:var(--text2)">env ready — processing all indices sequentially</span>
                </div>
                <div class="xv-fwd-detail" style="margin-top:6px">
                  <div class="xv-fwd-sig">
                    <span style="color:var(--text3);font-size:10px">for idx in range({{ TOTAL_UTTS }}):<br>&nbsp;&nbsp;def </span>
                    <span style="color:var(--forward);font-size:10px">forward</span>
                    <span style="color:var(--text2);font-size:10px">(</span>
                    <span v-for="(a, ai) in fwdArgs" :key="a.name">
                      <span :class="a.cls" style="font-size:10px;padding:0 2px">{{ a.name }}</span>
                      <span v-if="a.name === 'idx'" style="color:var(--text3);font-size:9px">:{{ idxAnnotation }}</span>
                      <span v-if="ai < fwdArgs.length-1" style="color:var(--text2);font-size:10px">, </span>
                    </span>
                    <span style="color:var(--text2);font-size:10px">):</span>
                  </div>
                  <div class="xv-steps">
                    <div class="xv-step" :class="{ active: procStep >= 1 }">
                      <span class="xv-step-num">1</span>
                      <span class="xv-step-text">{{ batchSize ? 'for i in' : 'load' }} </span>
                      <code :class="step1ArgCls" class="xv-step-code">{{ step1Code }}</code>
                    </div>
                    <div class="xv-fstep-arrow">↓</div>
                    <div class="xv-step" :class="{ active: procStep >= 2 }">
                      <span class="xv-step-num">2</span>
                      <span class="xv-step-text">extract with </span>
                      <code class="inject-match xv-step-code">model</code>
                    </div>
                    <div class="xv-fstep-arrow">↓</div>
                    <div class="xv-step" :class="{ active: procStep >= 3 }">
                      <span class="xv-step-num">3</span>
                      <span class="xv-step-text">save to </span>
                      <code class="inject-match xv-step-code">output_dir</code>
                    </div>
                  </div>
                </div>
              </div>
            </Transition>
          </template>

          <!-- ── DISTRIBUTED: vertical connector + workers ──────────────── -->
          <template v-else>

          <!-- Vertical connector: idx falls TOP→DOWN from runner to workers -->
          <div
            class="xv-v-connector"
            :class="{ 'connector-lit': phase === 'dispatch' || phase === 'process' || phase === 'done' }"
          >
            <div class="xv-v-line" :class="{ lit: phase === 'dispatch' || phase === 'process' || phase === 'done' }"></div>
            <template v-if="phase === 'dispatch'">
              <div
                v-for="i in Math.min(nWorkers, 3)"
                :key="`fall-${phaseIdx}-${i}`"
                class="xv-fall-packet"
                :style="{ animationDelay: `${(i-1)*0.45}s` }"
              >{{ batchSize ? `[…]` : 'idx' }}</div>
            </template>
            <div class="xv-v-arrow" :class="{ lit: phase === 'dispatch' || phase === 'process' || phase === 'done' }">↓</div>
          </div>

          <!-- Workers — count driven by nWorkers -->
          <div class="xv-workers">
            <TransitionGroup name="worker-list">
              <div
                v-for="(w, i) in workers"
                :key="w.id"
                class="xv-worker"
                :class="{
                  'box-forward': phase === 'init' || phase === 'process',
                  'box-worker':  phase === 'dispatch' || phase === 'done',
                }"
              >
                <!-- Header -->
                <div class="xv-worker-head">
                  <span style="color:var(--worker);font-weight:600;font-size:12px">{{ w.label }}</span>
                  <span class="xv-range">{{ w.range }}</span>
                  <span v-if="phase==='build'"    class="xv-status" style="color:var(--text3)">idle</span>
                  <span v-else-if="phase==='ship'" class="xv-status" style="color:var(--forward)">receiving…</span>
                  <span v-else-if="phase==='init'" class="xv-status fade-stagger" :style="{ animationDelay: `${i*0.35}s` }" style="color:var(--worker)">initializing…</span>
                  <span v-else-if="phase==='dispatch'" class="xv-status" style="color:var(--driver)">waiting idx…</span>
                  <span v-else-if="phase==='process'" class="xv-status" style="color:var(--forward)">processing…</span>
                  <span v-else-if="phase==='done'"    class="xv-status" style="color:var(--done)">✓ done</span>
                </div>

                <!-- env ready -->
                <Transition name="slide-down">
                  <div v-if="['init','dispatch','process','done'].includes(phase)" class="xv-env-check">
                    <span class="step-dot dot-worker" style="width:14px;height:14px;font-size:9px">✓</span>
                    <span style="font-size:10px;color:var(--text2)">env injected by name</span>
                  </div>
                </Transition>

                <!-- idx chips — shape changes with batch_size -->
                <Transition name="slide-down">
                  <div v-if="['dispatch','process','done'].includes(phase)" class="xv-idx-row">
                    <span class="xv-idx-from">↓ from runner</span>
                    <template v-for="(chip, ci) in idxChips(w)" :key="ci">
                      <span
                        class="xv-idx-chip"
                        :class="chip === '…' || String(chip).startsWith('[') ? 'xv-idx-dim' : 'inject-auto'"
                      >{{ chip }}</span>
                    </template>
                    <span v-if="batchSize" class="xv-batch-badge">batch={{ batchSize }}</span>
                  </div>
                </Transition>

                <!-- forward() detail -->
                <Transition name="slide-down">
                  <div v-if="phase === 'process' || phase === 'done'" class="xv-fwd-detail">
                    <!-- signature -->
                    <div class="xv-fwd-sig">
                      <span style="color:var(--text3);font-size:10px">def </span>
                      <span style="color:var(--forward);font-size:10px">forward</span>
                      <span style="color:var(--text2);font-size:10px">(</span>
                      <span v-for="(a, ai) in fwdArgs" :key="a.name">
                        <span :class="a.cls" style="font-size:10px;padding:0 2px">{{ a.name }}</span>
                        <span v-if="a.name === 'idx'" style="color:var(--text3);font-size:9px">:{{ idxAnnotation }}</span>
                        <span v-if="ai < fwdArgs.length-1" style="color:var(--text2);font-size:10px">, </span>
                      </span>
                      <span style="color:var(--text2);font-size:10px">):</span>
                    </div>
                    <!-- steps -->
                    <div class="xv-steps">
                      <div class="xv-step" :class="{ active: procStep >= 1 }">
                        <span class="xv-step-num">1</span>
                        <span class="xv-step-text">{{ batchSize ? 'for i in' : 'load' }} </span>
                        <code :class="step1ArgCls" class="xv-step-code">{{ step1Code }}</code>
                        <span v-if="batchSize" class="xv-step-text"> → load audio</span>
                      </div>
                      <div class="xv-fstep-arrow">↓</div>
                      <div class="xv-step" :class="{ active: procStep >= 2 }">
                        <span class="xv-step-num">2</span>
                        <span class="xv-step-text">extract with </span>
                        <code class="inject-match xv-step-code">model</code>
                      </div>
                      <div class="xv-fstep-arrow">↓</div>
                      <div class="xv-step" :class="{ active: procStep >= 3 }">
                        <span class="xv-step-num">3</span>
                        <span class="xv-step-text">save to </span>
                        <code class="inject-match xv-step-code">output_dir</code>
                      </div>
                    </div>
                  </div>
                </Transition>

              </div><!-- end worker -->
            </TransitionGroup>
          </div>

          </template><!-- end distributed -->

        </div><!-- end right col -->
      </div><!-- end grid -->

      <!-- Output row -->
      <Transition name="slide-down">
        <div v-if="phase === 'done'" class="xv-output-row">
          <div class="xv-output-label" style="color:var(--done)">output_dir/</div>
          <div class="xv-pt-files">
            <span
              v-for="(f, i) in ptFiles" :key="f"
              class="xv-pt-chip box-done"
              :style="{ animationDelay: `${i * 0.1}s` }"
            >{{ f }}</span>
          </div>
        </div>
      </Transition>

      <!-- Arg legend -->
      <div class="xv-legend-row">
        <span class="inject-auto xv-legend-chip">idx</span>
        <span class="xv-legend-desc">dispatched by BaseRunner — not from Provider</span>
        <span class="xv-legend-sep">·</span>
        <span class="inject-match xv-legend-chip">model, utterances…</span>
        <span class="xv-legend-desc">injected from provider env — key name = arg name</span>
      </div>

      <!-- Controls -->
      <div class="xv-controls">
        <button class="btn" @click="goTo((phaseIdx-1+PHASES.length)%PHASES.length)">← Prev</button>
        <button class="btn btn-primary" @click="togglePlay">
          {{ playing ? '⏸ Pause' : phaseIdx === PHASES.length - 1 ? '↺ Replay' : '▶ Play' }}

        </button>
        <button class="btn" @click="goTo((phaseIdx+1)%PHASES.length)">Next →</button>
      </div>

    </div>
  </section>
</template>

<style scoped>
/* ── Config panel ─────────────────────────────────────────────────────────── */
.cfg-panel {
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
}

.cfg-slider-row {
  display: flex; align-items: center; gap: 10px;
}
.cfg-slider {
  flex: 1; accent-color: var(--driver);
  cursor: pointer;
}
.cfg-slider-val {
  min-width: 28px; text-align: center;
  font-family: var(--mono); font-size: 14px; font-weight: 700;
  padding: 2px 6px; border-radius: 4px;
}

.xv-batch-badge {
  font-family: var(--mono); font-size: 9px; font-weight: 700;
  padding: 1px 6px; border-radius: 3px;
  background: var(--driver-bg); border: 1px solid var(--driver-border); color: var(--driver);
  letter-spacing: 0.04em;
}

/* ── Phase pip track ──────────────────────────────────────────────────────── */
.phase-track { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
.phase-pip {
  width: 10px; height: 10px; border-radius: 50%;
  border: 1.5px solid var(--border2); background: var(--bg3);
  cursor: pointer; padding: 0; transition: all 0.2s;
}
.phase-pip.past   { background: var(--done);   border-color: var(--done-border); }
.phase-pip.active { background: var(--driver); border-color: var(--driver-border); transform: scale(1.35); }

.phase-label-wrap { min-height: 26px; margin-bottom: 1.1rem; }
.phase-label { display: block; font-family: var(--mono); font-size: 12.5px; font-weight: 600; }

/* ── Main grid ────────────────────────────────────────────────────────────── */
.xv-grid {
  display: grid;
  grid-template-columns: 1fr 68px 1fr;
  align-items: start;
  margin-bottom: 1rem;
}

/* ── Shared box chrome ────────────────────────────────────────────────────── */
.xv-box {
  border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem 1.1rem; font-family: var(--mono); font-size: 13px;
  transition: border-color 0.35s, background 0.35s;
}
.xv-box-title {
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 0.7rem; display: block;
}

/* ── Provider ────────────────────────────────────────────────────────────── */
.xv-method-header {
  font-size: 11px; font-weight: 600; color: var(--provider);
  background: var(--provider-bg); border: 1px solid var(--provider-border);
  border-radius: 5px 5px 0 0; padding: 4px 8px;
}
.xv-setup-body {
  border: 1px solid var(--forward-border); border-top: none;
  border-radius: 0 0 5px 5px; padding: 8px 10px; margin-bottom: 8px;
}
.xv-fn-sig, .xv-fn-close { font-size: 11.5px; font-weight: 600; line-height: 1.8; }
.xv-env-row {
  display: flex; align-items: center; gap: 5px;
  font-size: 11px; line-height: 1.75; opacity: 0;
  animation: key-appear 0.35s ease forwards;
}
.xv-env-row.visible { opacity: 1; animation: none; }
@keyframes key-appear {
  from { opacity: 0; transform: translateX(-6px); }
  to   { opacity: 1; transform: translateX(0); }
}
.xv-key   { font-weight: 600; font-size: 10.5px; }
.xv-colon { color: var(--text3); }
.xv-val   { color: var(--text2); font-size: 10.5px; }
.xv-ready-badge {
  font-size: 11px; padding: 3px 8px; border-radius: 5px;
  border: 1px solid var(--done-border); display: inline-block; margin-top: 4px;
}

/* ── Horizontal channel (setup() only) ───────────────────────────────────── */
.xv-h-channel {
  display: flex; align-items: center; position: relative;
  padding-top: 52px;
}
.xv-h-line {
  flex: 1; height: 1.5px; background: var(--border2); transition: background 0.4s;
}
.xv-h-line.lit { background: var(--provider); }
.xv-h-tip {
  font-size: 18px; line-height: 1; flex-shrink: 0;
  color: var(--border2); transition: color 0.4s;
}
.xv-h-tip.lit { color: var(--provider); }
.xv-h-label {
  position: absolute; top: 34px; left: 50%; transform: translateX(-50%);
  font-family: var(--mono); font-size: 9px; color: var(--text3);
  white-space: nowrap; opacity: 0; transition: opacity 0.3s;
}
.xv-h-label.visible { opacity: 1; }
.xv-h-packet {
  position: absolute; top: 50%; transform: translateY(-50%);
  font-size: 9px; font-family: var(--mono); font-weight: 700;
  padding: 2px 5px; border-radius: 4px; white-space: nowrap;
  animation: h-fly 1.8s ease-in-out forwards; opacity: 0;
}
.pkt-forward {
  background: var(--forward-bg); border: 1px solid var(--forward-border); color: var(--forward);
}
@keyframes h-fly {
  0%   { left: 0%;  opacity: 0; transform: translateY(-50%) scale(0.7); }
  12%  {            opacity: 1; transform: translateY(-50%) scale(1);   }
  80%  {            opacity: 1; }
  100% { left: 80%; opacity: 0; }
}

/* ── Right column ─────────────────────────────────────────────────────────── */
.xv-right-col {
  border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem 1.1rem; font-family: var(--mono); font-size: 13px;
  display: flex; flex-direction: column;
}

/* BaseRunner box */
.xv-runner-box {
  border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; margin-bottom: 0;
  transition: border-color 0.35s, background 0.35s;
}
.xv-runner-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 5px; flex-wrap: wrap; gap: 4px;
}
.xv-runner-badge {
  font-family: var(--mono); font-size: 9px; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 4px;
  border: 1px solid var(--driver-border); color: var(--driver); background: var(--driver-bg);
}
.xv-runner-body { font-size: 11px; }
.xv-runner-idle { color: var(--text3); font-size: 11px; }
.xv-runner-row  { display: flex; align-items: center; gap: 5px; line-height: 1.7; flex-wrap: wrap; }

/* Vertical connector */
.xv-v-connector {
  position: relative; width: 40px; height: 52px;
  margin: 0 auto; display: flex; flex-direction: column;
  align-items: center; overflow: visible;
}
.xv-v-line {
  width: 1.5px; flex: 1; background: var(--border2); transition: background 0.4s;
}
.xv-v-line.lit { background: var(--driver); }
.xv-v-arrow {
  font-size: 14px; line-height: 1; color: var(--border2); transition: color 0.4s;
}
.xv-v-arrow.lit { color: var(--driver); }

/* Falling idx packets */
.xv-fall-packet {
  position: absolute; top: 0; left: 50%; transform: translateX(-50%);
  font-size: 9px; font-family: var(--mono); font-weight: 700;
  padding: 2px 5px; border-radius: 4px; z-index: 1;
  background: var(--driver-bg); border: 1px solid var(--driver-border); color: var(--driver);
  white-space: nowrap; animation: fall-down 1.3s ease-in-out forwards; opacity: 0;
}
@keyframes fall-down {
  0%   { top:  0px; opacity: 0; transform: translateX(-50%) scale(0.75); }
  18%  { top:  4px; opacity: 1; transform: translateX(-50%) scale(1);    }
  75%  {            opacity: 1;                                            }
  100% { top: 44px; opacity: 0;                                            }
}

/* Workers */
.xv-workers { display: flex; flex-direction: column; gap: 5px; }
.xv-worker {
  border: 1px solid var(--border); border-radius: 7px;
  padding: 7px 9px; font-family: var(--mono); font-size: 12px;
  transition: border-color 0.35s, background 0.35s; overflow: hidden;
}
.xv-worker-head {
  display: flex; align-items: center; gap: 5px; margin-bottom: 4px; flex-wrap: wrap;
}
.xv-range  { font-size: 10px; color: var(--text3); flex: 1; min-width: 60px; }
.xv-status { font-size: 10px; font-family: var(--mono); letter-spacing: 0.03em; }
.fade-stagger { opacity: 0; animation: fade-in 0.4s ease forwards; }
@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }

.xv-env-check {
  display: flex; align-items: center; gap: 5px;
  padding: 3px 0; border-bottom: 1px solid var(--border); margin-bottom: 5px;
}

/* idx row */
.xv-idx-row {
  display: flex; align-items: center; gap: 4px;
  margin-bottom: 5px; flex-wrap: wrap;
}
.xv-idx-from {
  font-size: 9px; color: var(--driver); font-family: var(--mono); font-weight: 700;
}
.xv-idx-chip {
  padding: 1px 5px; border-radius: 3px;
  font-size: 10px; font-family: var(--mono);
}
.xv-idx-dim {
  color: var(--text3); font-size: 10px; font-family: var(--mono);
}

/* forward() detail */
.xv-fwd-detail {
  background: var(--bg3); border: 1px solid var(--border); border-radius: 5px; padding: 6px 8px;
}
.xv-fwd-sig { line-height: 1.65; margin-bottom: 5px; word-break: break-all; }

.xv-steps { display: flex; flex-direction: column; gap: 2px; }
.xv-step {
  display: flex; align-items: center; gap: 5px; flex-wrap: wrap;
  font-size: 11px; color: var(--text3); transition: color 0.25s; padding: 2px 0;
}
.xv-step.active { color: var(--text2); }
.xv-step-num {
  width: 16px; height: 16px; border-radius: 50%;
  border: 1px solid var(--border2); background: var(--bg3); color: var(--text3);
  font-size: 9px; font-weight: 700; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; transition: all 0.25s;
}
.xv-step.active .xv-step-num {
  background: var(--forward-bg); border-color: var(--forward-border); color: var(--forward);
}
.xv-step-text { font-size: 11px; }
.xv-step-code {
  padding: 1px 4px; border-radius: 3px; font-size: 10px;
  background: var(--bg3); color: var(--text3); transition: all 0.25s;
}
.xv-fstep-arrow { color: var(--text3); font-size: 11px; padding-left: 21px; }

/* Output row */
.xv-output-row {
  border: 1px solid var(--done-border); background: var(--done-bg);
  border-radius: 8px; padding: 10px 14px; margin-bottom: 1rem;
}
.xv-output-label {
  font-family: var(--mono); font-size: 11px; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 6px;
}
.xv-pt-files { display: flex; flex-wrap: wrap; gap: 5px; }
.xv-pt-chip {
  font-family: var(--mono); font-size: 11px; padding: 2px 8px;
  border-radius: 4px; border: 1px solid var(--done-border);
  opacity: 0; animation: chip-pop 0.3s ease forwards;
}
@keyframes chip-pop {
  from { opacity: 0; transform: scale(0.8) translateY(4px); }
  to   { opacity: 1; transform: scale(1)   translateY(0); }
}

/* Legend */
.xv-legend-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  font-size: 11px; color: var(--text3); font-family: var(--mono);
  margin-bottom: 1rem; padding: 8px 10px;
  background: var(--bg2); border: 1px solid var(--border); border-radius: 6px;
}
.xv-legend-chip { font-size: 11px; }
.xv-legend-desc { color: var(--text3); }
.xv-legend-sep  { color: var(--border2); }

/* Controls */
.xv-controls { display: flex; gap: 8px; justify-content: center; }

/* Transitions */
.label-fade-enter-active, .label-fade-leave-active { transition: opacity 0.2s, transform 0.2s; }
.label-fade-enter-from,   .label-fade-leave-to     { opacity: 0; transform: translateY(5px); }

.slide-down-enter-active { transition: all 0.35s ease; }
.slide-down-leave-active { transition: all 0.2s ease; }
.slide-down-enter-from   { opacity: 0; transform: translateY(-8px); }
.slide-down-leave-to     { opacity: 0; }

.pop-enter-active { transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1); }
.pop-leave-active { transition: all 0.15s ease; }
.pop-enter-from   { opacity: 0; transform: scale(0.7); }
.pop-leave-to     { opacity: 0; }

.cfg-fade-enter-active, .cfg-fade-leave-active { transition: all 0.25s ease; }
.cfg-fade-enter-from, .cfg-fade-leave-to       { opacity: 0; transform: translateY(-6px); }

/* Worker list add/remove */
.worker-list-enter-active { transition: all 0.3s ease; }
.worker-list-leave-active { transition: all 0.2s ease; }
.worker-list-enter-from   { opacity: 0; transform: translateX(12px); }
.worker-list-leave-to     { opacity: 0; transform: translateX(12px); }


/* null badge style — subdued compared to inject-auto */
/* local sequential loop box */
.xv-local-loop { margin-top: 6px; }
.xv-local-label {
  display: flex; align-items: center; gap: 5px;
  padding: 4px 0; border-bottom: 1px solid var(--border); margin-bottom: 4px;
}

/* greyed-out field when disabled */
.cfg-field-dim { opacity: 0.45; pointer-events: none; }

/* provider-colored h-packet (local mode: env dict travels) */
.pkt-provider {
  background: var(--provider-bg); border: 1px solid var(--provider-border); color: var(--provider);
}

.cfg-val-null {
  font-family: var(--mono);
  font-size: 12px;
  font-style: italic;
  color: var(--text3);
  min-width: 34px;
  text-align: center;
}

/* Responsive */
@media (max-width: 620px) {
  .xv-grid { grid-template-columns: 1fr; }
  .xv-h-channel { display: none; }
  .config-grid { grid-template-columns: 1fr; }
}
</style>
