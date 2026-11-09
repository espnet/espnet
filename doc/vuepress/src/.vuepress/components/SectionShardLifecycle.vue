<script setup>
import { ref, computed } from 'vue'

const TOTAL_STEPS = 6
const currentStep = ref(0)

const canPrev = computed(() => currentStep.value > 0)
const canNext = computed(() => currentStep.value < TOTAL_STEPS - 1)

function goTo(n) {
  currentStep.value = Math.max(0, Math.min(TOTAL_STEPS - 1, n))
}

// indices strip data
const indicesPlain = Array.from({ length: 40 }, (_, i) => ({ i, shard: -1 }))
const indicesSharded = Array.from({ length: 40 }, (_, i) => ({ i, shard: Math.floor(i / 10) }))
</script>

<template>
  <section class="guide-section" id="shard-lifecycle">
    <div class="container">
      <p class="section-desc">
        What happens internally between calling <code>runner(range(200))</code> and getting the
        result back. Step through each stage.
      </p>

      <!-- Stepper nav -->
      <div class="stepper-header">
        <div class="step-indicators">
          <button
            v-for="n in TOTAL_STEPS"
            :key="n"
            class="step-ind"
            :class="{ active: currentStep === n - 1, done: currentStep > n - 1 }"
            @click="goTo(n - 1)"
          >{{ n }}</button>
        </div>
        <div class="stepper-nav">
          <button class="btn" :disabled="!canPrev" @click="goTo(currentStep - 1)">← Prev</button>
          <button class="btn btn-primary" :disabled="!canNext" @click="goTo(currentStep + 1)">Next →</button>
        </div>
      </div>

      <!-- Step 0 -->
      <template v-if="currentStep === 0">
        <div class="stage-title">① Indices received</div>
        <p class="stage-desc">
          Calling <code>runner(range(200))</code> triggers <code>BaseRunner.__call__()</code>.
          All indices are converted to a list. If <code>batch_size</code> is set, they are grouped
          into batches here.
        </p>
        <div class="callout callout-info">
          <span class="callout-icon">ℹ</span>
          <div>
            <p>
              <strong>Where is <code>output_dir</code> set?</strong> It is passed as a constructor
              argument to the Runner. All shard files, <code>manifest.json</code>, and final outputs
              are written under this directory.
            </p>
            <pre style="margin-top: .75rem"><code>runner = MyRunner(
    provider,
    output_dir=<span class="str">"exp/decode"</span>,   <span class="cm"># ← set here</span>
    shard_subdir=<span class="str">"train"</span>,       <span class="cm"># shards go under output_dir/train/</span>
    resume=<span class="num">True</span>,               <span class="cm"># skip already-done shards</span>
)
<span class="cm"># When using collect_stats(), output_dir is passed as a function</span>
<span class="cm"># argument and forwarded to the Runner automatically.</span></code></pre>
          </div>
        </div>
        <div class="shard-visual">
          <div style="font-size: 12px; color: var(--text3); font-family: var(--mono); margin-bottom: .75rem">
            indices = range(200) → 200 indices
          </div>
          <div class="indices-strip">
            <div v-for="{ i } in indicesPlain" :key="i" class="idx-cell">{{ i }}</div>
            <div class="idx-cell">…</div>
          </div>
        </div>
        <pre><code><span class="cm"># First thing BaseRunner.__call__ does</span>
indices = list(indices)
<span class="kw">if</span> self.batch_size <span class="kw">is not</span> None:
    indices = [
        list(indices[i : i + self.batch_size])
        <span class="kw">for</span> i <span class="kw">in</span> range(<span class="num">0</span>, len(indices), self.batch_size)
    ]</code></pre>
      </template>

      <!-- Step 1 -->
      <template v-if="currentStep === 1">
        <div class="stage-title">② Shard planning (_plan_shards)</div>
        <p class="stage-desc">
          Indices are divided evenly according to <code>parallel_config.n_workers</code>.
          In local mode this is always 1 shard. With <code>n_workers=4</code> on SLURM,
          4 shards are created.
        </p>
        <div class="shard-visual">
          <div style="font-size: 12px; color: var(--text3); font-family: var(--mono); margin-bottom: .75rem">
            n_workers=4: 200 indices → 4 shards of 50 each
          </div>
          <div class="indices-strip">
            <div
              v-for="{ i, shard } in indicesSharded"
              :key="i"
              class="idx-cell"
              :class="`shard-${shard}`"
            >{{ i }}</div>
            <div class="idx-cell">…</div>
          </div>
          <div class="shards-row" style="margin-top: 1rem">
            <div class="shard-box pending"><div class="shard-box-title">split.0</div><div class="shard-items">idx 0..49</div></div>
            <div class="shard-box pending"><div class="shard-box-title">split.1</div><div class="shard-items">idx 50..99</div></div>
            <div class="shard-box pending"><div class="shard-box-title">split.2</div><div class="shard-items">idx 100..149</div></div>
            <div class="shard-box pending"><div class="shard-box-title">split.3</div><div class="shard-items">idx 150..199</div></div>
          </div>
        </div>
        <div class="callout callout-info">
          <span class="callout-icon">ℹ</span>
          <div>
            <p><strong><code>batch_size</code> enables batch processing</strong></p>
            <p style="margin-top: .5rem">
              This is a <strong>Runner constructor argument</strong>, not part of parallel config.
              When set, indices within each shard are further grouped into batches and passed to
              <code>forward</code> as a list (see Step ⑤).
            </p>
            <pre style="margin-top: .75rem"><code><span class="cm"># Pass directly to Runner</span>
runner = MyRunner(
    provider,
    output_dir=<span class="str">"exp/decode"</span>,
    batch_size=<span class="num">4</span>,   <span class="cm"># forward receives [idx0, idx1, idx2, idx3]</span>
)

<span class="cm"># For collect_stats(), pass as a function argument (default: 4)</span>
collect_stats(..., batch_size=<span class="num">8</span>)
<span class="cm"># → _chunk_indices() pre-batches before passing to the runner</span></code></pre>
          </div>
        </div>
      </template>

      <!-- Step 2 -->
      <template v-if="currentStep === 2">
        <div class="stage-title">③ manifest.json is written</div>
        <p class="stage-desc">
          Once the shard plan is finalized, it is written to <code>output_dir/manifest.json</code>.
          This allows a resumed run to know exactly what was planned — even after a crash.
        </p>
        <div class="shard-visual">
          <div class="dir-tree">
            <span class="dir-provider">output_dir/</span><br>
            &nbsp;&nbsp;<span class="dir-provider">manifest.json ← written now</span><br>
            &nbsp;&nbsp;<span style="color: var(--text3)">split.0/  (not yet)</span><br>
            &nbsp;&nbsp;<span style="color: var(--text3)">split.1/  (not yet)</span><br>
            &nbsp;&nbsp;<span style="color: var(--text3)">split.2/  (not yet)</span><br>
            &nbsp;&nbsp;<span style="color: var(--text3)">split.3/  (not yet)</span>
          </div>
        </div>
        <pre><code><span class="cm"># manifest.json contents (example)</span>
{
  <span class="str">"version"</span>: <span class="num">1</span>,
  <span class="str">"output_dir"</span>: <span class="str">"exp/decode"</span>,
  <span class="str">"shard_subdir"</span>: <span class="str">"train"</span>,
  <span class="str">"shards"</span>: [
    { <span class="str">"shard_id"</span>: <span class="num">0</span>, <span class="str">"items"</span>: [<span class="num">0</span>, <span class="num">1</span>, ..., <span class="num">49</span>] },
    { <span class="str">"shard_id"</span>: <span class="num">1</span>, <span class="str">"items"</span>: [<span class="num">50</span>, ..., <span class="num">99</span>] },
    ...
  ]
}</code></pre>
      </template>

      <!-- Step 3 -->
      <template v-if="currentStep === 3">
        <div class="stage-title">④ Resume check (_filter_pending_shards)</div>
        <p class="stage-desc">
          With <code>resume=True</code> (default), each shard directory is checked for a
          <code>done</code> file. If it exists, the shard is skipped — no recomputation after a crash.
        </p>
        <div class="shard-visual">
          <div style="font-size: 12px; color: var(--text3); font-family: var(--mono); margin-bottom: .75rem">
            Example: split.0 and split.1 completed in a previous run
          </div>
          <div class="shards-row">
            <div class="shard-box complete"><div class="shard-box-title">split.0</div><div class="shard-items">done ✓</div><div class="shard-status" style="color: var(--done)">→ SKIP</div></div>
            <div class="shard-box complete"><div class="shard-box-title">split.1</div><div class="shard-items">done ✓</div><div class="shard-status" style="color: var(--done)">→ SKIP</div></div>
            <div class="shard-box running"><div class="shard-box-title">split.2</div><div class="shard-items">no done file</div><div class="shard-status" style="color: var(--provider)">→ pending</div></div>
            <div class="shard-box running"><div class="shard-box-title">split.3</div><div class="shard-items">no done file</div><div class="shard-status" style="color: var(--provider)">→ pending</div></div>
          </div>
          <div class="dir-tree" style="margin-top: 1rem">
            output_dir/<br>
            &nbsp;&nbsp;<span class="dir-driver">split.0/</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="dir-done">done  ← presence of this file = shard complete</span><br>
            &nbsp;&nbsp;<span class="dir-driver">split.1/</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="dir-done">done</span><br>
            &nbsp;&nbsp;<span class="dir-provider">split.2/  ← will be processed</span><br>
            &nbsp;&nbsp;<span class="dir-provider">split.3/  ← will be processed</span>
          </div>
        </div>
      </template>

      <!-- Step 4 -->
      <template v-if="currentStep === 4">
        <div class="stage-title">⑤ _run_one_shard (the worker's job)</div>
        <p class="stage-desc">
          Each shard is processed by <code>_run_one_shard()</code>. On SLURM this runs on a remote
          machine; locally it runs in the same process. The internal flow is identical either way.
        </p>
        <div class="shard-visual">
          <div style="display: flex; flex-direction: column; gap: .5rem">
            <div class="exec-box box-provider">
              <div class="exec-box-title">1. init_state()</div>
              <div class="exec-box-sub">Creates split.N/ directory. Opens file handles via open_writers().</div>
            </div>
            <div class="exec-step-arrow">↓</div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: .75rem">
              <div>
                <div style="font-size: 11px; font-family: var(--mono); color: var(--text3); text-transform: uppercase; letter-spacing: .06em; margin-bottom: .4rem">batch_size=None (default)</div>
                <div class="exec-box box-driver" style="height: 100%">
                  <div class="exec-box-title">2. for item in items:</div>
                  <div class="exec-box-sub" style="font-family: var(--mono)">
                    forward(<span style="color: var(--driver)">3</span>, **env)<br>
                    forward(<span style="color: var(--driver)">7</span>, **env)<br>
                    forward(<span style="color: var(--driver)">12</span>, **env)<br>
                    <span style="color: var(--text3)">...</span>
                  </div>
                  <div class="exec-box-sub" style="margin-top: .5rem">idx is a single integer</div>
                </div>
              </div>
              <div>
                <div style="font-size: 11px; font-family: var(--mono); color: var(--text3); text-transform: uppercase; letter-spacing: .06em; margin-bottom: .4rem">batch_size=4</div>
                <div class="exec-box box-forward" style="height: 100%">
                  <div class="exec-box-title">2. for item in items:</div>
                  <div class="exec-box-sub" style="font-family: var(--mono)">
                    forward(<span style="color: var(--forward)">[3,7,12,5]</span>, **env)<br>
                    forward(<span style="color: var(--forward)">[1,9,4,11]</span>, **env)<br>
                    <span style="color: var(--text3)">...</span>
                  </div>
                  <div class="exec-box-sub" style="margin-top: .5rem">idx is a list — handle as <code>Iterable[int]</code> in forward</div>
                </div>
              </div>
            </div>

            <div class="exec-step-arrow">↓</div>
            <div class="exec-box box-provider">
              <div class="exec-box-title">3. finalize_state()</div>
              <div class="exec-box-sub">Flushes and closes file handles via close_writers().</div>
            </div>
            <div class="exec-step-arrow">↓</div>
            <div class="exec-box box-done">
              <div class="exec-box-title">4. Write done file</div>
              <div class="exec-box-sub">split.N/done exists = this shard is complete.</div>
            </div>
          </div>
        </div>
      </template>

      <!-- Step 5 -->
      <template v-if="currentStep === 5">
        <div class="stage-title">⑥ merge() — back on the driver</div>
        <p class="stage-desc">
          Once all shard <code>done</code> files are confirmed, the driver calls
          <code>merge(shard_dirs)</code> to combine per-shard outputs into the final result.
        </p>
        <div class="shard-visual">
          <div class="shards-row">
            <div class="shard-box complete" v-for="n in 4" :key="n">
              <div class="shard-box-title">split.{{ n - 1 }}</div>
              <div class="shard-items">stats.npz ✓<br>shape_keys ✓<br>done ✓</div>
            </div>
          </div>
          <div style="text-align: center; margin: 1rem 0; font-size: 20px; color: var(--text3)">↓ merge()</div>
          <div class="dir-tree">
            output_dir/<br>
            &nbsp;&nbsp;<span class="dir-done">train/</span>  ← final output<br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="dir-done">stats_keys</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="dir-done">feats_stats.npz</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="dir-done">feats_shape</span>
          </div>
          <div class="callout callout-tip" style="margin-top: 1rem; margin-bottom: 0">
            <span class="callout-icon">✓</span>
            <div>
              <p>
                <strong>CollectStats example:</strong> <code>CollectStatsRunner.merge()</code>
                reads each shard's <code>.npz</code> and sums up sum/sq/count to produce the final
                mean and variance statistics used for feature normalization.
              </p>
            </div>
          </div>
        </div>
      </template>
    </div>
  </section>
</template>

<style scoped>
.stage-title { font-size: 18px; font-weight: 500; margin-bottom: .5rem; }
.stage-desc  { color: var(--text2); font-size: 14px; line-height: 1.75; margin-bottom: 1.5rem; max-width: 680px; }
</style>
