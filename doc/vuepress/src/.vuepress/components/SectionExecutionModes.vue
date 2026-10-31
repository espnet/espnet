<script setup>
import { ref } from 'vue'

const modes = ['local', 'local_gpu', 'slurm']
const activeMode = ref('local')
</script>

<template>
  <section class="guide-section" id="execution-modes">
    <div class="container">
      <p class="section-desc">
        A single value in <code>parallel_config.env</code> fundamentally changes where execution
        happens. Switch tabs to see the flow for each mode.
      </p>

      <!-- Tabs -->
      <div class="tab-row">
        <button
          v-for="mode in modes"
          :key="mode"
          class="tab-btn"
          :class="{ active: activeMode === mode }"
          @click="activeMode = mode"
        >
          env: {{ mode }}
        </button>
      </div>

      <!-- local -->
      <template v-if="activeMode === 'local'">
        <div class="exec-diagram">
          <div class="exec-col col-driver">
            <p class="exec-col-title" style="color: var(--driver)">🖥 Your machine (Driver)</p>
            <div class="exec-box box-driver">
              <div class="exec-box-title">__call__(indices)</div>
              <div class="exec-box-sub">Plans shards, writes manifest.json</div>
            </div>
            <div class="exec-step-arrow">↓ _run_local()</div>
            <div class="exec-box box-driver">
              <div class="exec-box-title">_run_one_shard()</div>
              <div class="exec-box-sub">Processes shards sequentially (no parallelism)</div>
            </div>
            <div class="exec-step-arrow">↓</div>
            <div class="exec-box box-forward">
              <div class="exec-box-title">forward(idx, dataset, model)</div>
              <div class="exec-box-sub">Runs in the same process as the driver</div>
            </div>
            <div class="exec-step-arrow">↓ merge()</div>
            <div class="exec-box box-driver">
              <div class="exec-box-title">merge(shard_dirs)</div>
              <div class="exec-box-sub">Combines all shards into final output</div>
            </div>
          </div>
          <div class="exec-col" style="max-width: 260px">
            <p class="exec-col-title" style="color: var(--text3)">📁 Filesystem</p>
            <div class="exec-box">
              <div class="exec-box-title" style="color: var(--text2)">output_dir/</div>
              <div class="exec-box-sub">manifest.json<br>split.0/ → done<br>split.1/ → done<br>...</div>
            </div>
            <div class="env-note" style="margin-top: .75rem">
              <strong>Local mode:</strong> No Dask, no workers. The driver processes all shards in
              sequence. Progress is shown with <code>tqdm</code>.
            </div>
          </div>
        </div>
        <pre><code><span class="cm"># local mode (config.yaml)</span>
<span class="fn">parallel</span>:
  <span class="fn">env</span>: <span class="str">local</span>
  <span class="fn">n_workers</span>: <span class="num">1</span>    <span class="cm"># not used in local mode</span>
  <span class="fn">options</span>: {}</code></pre>
      </template>

      <!-- local_gpu -->
      <template v-if="activeMode === 'local_gpu'">
        <div class="exec-diagram">
          <div class="exec-col col-driver">
            <p class="exec-col-title" style="color: var(--driver)">🖥 Your machine (Driver)</p>
            <div class="exec-box box-driver">
              <div class="exec-box-title">build_local_gpu_cluster()</div>
              <div class="exec-box-sub">Requires dask_cuda. Auto-detects GPU count.</div>
            </div>
          </div>
          <div class="exec-arrow">→</div>
          <div class="exec-col col-worker">
            <p class="exec-col-title" style="color: var(--worker)">🎮 GPU Workers (same machine)</p>
            <div class="exec-box box-worker">
              <div class="exec-box-title">GPU #0, #1, #2, ... each Worker</div>
              <div class="exec-box-sub">1 Worker = 1 GPU<br>n_workers ≤ number of GPUs</div>
            </div>
            <div style="margin-top: .75rem; font-size: 12px; color: var(--text3); font-family: var(--mono)">
              DASK_WORKER_ID=0 → GPU:0<br>
              DASK_WORKER_ID=1 → GPU:1<br>
              ...
            </div>
          </div>
        </div>
        <div class="callout callout-tip">
          <span class="callout-icon">✓</span>
          <div>
            <p>
              <strong>What is local_gpu?</strong> Parallelizes across multiple GPUs on the same
              machine. Requires <code>dask_cuda</code>. Each worker can call
              <code>torch.cuda.set_device(int(os.environ["DASK_WORKER_ID"]))</code> to pin itself
              to a specific GPU.
            </p>
          </div>
        </div>
        <pre><code><span class="cm"># local_gpu mode (config.yaml)</span>
<span class="fn">parallel</span>:
  <span class="fn">env</span>: <span class="str">local_gpu</span>
  <span class="fn">n_workers</span>: <span class="num">4</span>    <span class="cm"># must not exceed number of available GPUs</span>
  <span class="fn">options</span>: {}</code></pre>
      </template>

      <!-- slurm -->
      <template v-if="activeMode === 'slurm'">
        <div class="exec-diagram">
          <div class="exec-col col-driver">
            <p class="exec-col-title" style="color: var(--driver)">🖥 Login Node (Driver)</p>
            <div class="exec-box box-driver">
              <div class="exec-box-title">__call__(indices)</div>
              <div class="exec-box-sub">Splits indices into n_workers shards</div>
            </div>
            <div class="exec-step-arrow">↓ _run_parallel_dask()</div>
            <div class="exec-box box-driver">
              <div class="exec-box-title">get_client() → SLURMCluster</div>
              <div class="exec-box-sub">dask_jobqueue submits sbatch jobs</div>
            </div>
            <div class="exec-step-arrow">↓ client.map()</div>
            <div class="exec-box box-driver">
              <div class="exec-box-title">merge(shard_dirs)</div>
              <div class="exec-box-sub">Runs on the driver after all workers finish</div>
            </div>
          </div>
          <div class="exec-arrow">→</div>
          <div class="exec-col col-worker">
            <p class="exec-col-title" style="color: var(--worker)">⚙ Compute Nodes (Workers)</p>
            <div class="exec-box box-provider">
              <div class="exec-box-title">DictReturnWorkerPlugin.setup()</div>
              <div class="exec-box-sub">Called once per worker on startup.<br>Initializes dataset / model.</div>
            </div>
            <div class="exec-step-arrow">↓ shard job arrives</div>
            <div class="exec-box box-worker">
              <div class="exec-box-title">_run_one_shard()</div>
              <div class="exec-box-sub">Runs on this compute node</div>
            </div>
            <div class="exec-step-arrow">↓</div>
            <div class="exec-box box-forward">
              <div class="exec-box-title">forward(idx, dataset, model)</div>
              <div class="exec-box-sub">env is injected automatically (see §03)</div>
            </div>
          </div>
        </div>
        <div class="callout callout-warn">
          <span class="callout-icon">⚠</span>
          <div>
            <p>
              <strong>Important:</strong> In SLURM mode, <code>forward</code> runs on a
              <strong>different machine</strong>. It must be a <code>@staticmethod</code> —
              methods that capture <code>self</code> cannot be pickled and sent by Dask.
            </p>
          </div>
        </div>
        <pre><code><span class="cm"># slurm mode (config.yaml)</span>
<span class="fn">parallel</span>:
  <span class="fn">env</span>: <span class="str">slurm</span>
  <span class="fn">n_workers</span>: <span class="num">8</span>    <span class="cm"># submits 8 SLURM jobs</span>
  <span class="fn">options</span>:
    <span class="fn">queue</span>: <span class="str">gpu</span>
    <span class="fn">cores</span>: <span class="num">4</span>
    <span class="fn">memory</span>: <span class="str">"32GB"</span>
    <span class="fn">walltime</span>: <span class="str">"02:00:00"</span></code></pre>
        <div class="callout callout-info">
          <span class="callout-icon">ℹ</span>
          <div>
            <p>
              <strong>Other cluster types (PBS, SGE, LSF, HTCondor, ...):</strong>
              Just change <code>env</code> to <code>"pbs"</code>, <code>"sge"</code>,
              <code>"lsf"</code>, etc. The available <code>options</code> vary by scheduler —
              see the
              <a href="https://jobqueue.dask.org/en/latest/" target="_blank">dask-jobqueue documentation</a>.
            </p>
          </div>
        </div>
      </template>
    </div>
  </section>
</template>
