<script setup>
// No reactive state needed — purely presentational
</script>

<template>
  <section class="guide-section" id="overview">
    <div class="container">
      <p class="section-desc">
        ESPnet3 parallel processing is built around four components. Each has a clear responsibility.
        Colors are used consistently throughout this guide.
      </p>

      <div class="legend">
        <span class="legend-item"><span class="legend-dot ld-driver" />Driver-side</span>
        <span class="legend-item"><span class="legend-dot ld-worker" />Worker-side</span>
        <span class="legend-item"><span class="legend-dot ld-provider" />Provider / init</span>
        <span class="legend-item"><span class="legend-dot ld-forward" />forward()</span>
        <span class="legend-item"><span class="legend-dot ld-done" />done / complete</span>
      </div>

      <div class="overview-grid">
        <div class="role-card c-driver">
          <p class="role-label">BaseRunner</p>
          <h3>The orchestrator (Driver)</h3>
          <p>Splits data into shards, dispatches them to workers, and merges results when all workers finish. It never processes data itself.</p>
          <div class="role-methods">
            <span class="method-tag">__call__()</span>
            <span class="method-tag">_plan_shards()</span>
            <span class="method-tag">merge()</span>
          </div>
        </div>

        <div class="role-card c-forward">
          <p class="role-label">forward()</p>
          <h3>The work unit (Worker)</h3>
          <p>
            A pure function that processes one sample or batch. Must be a
            <code>@staticmethod</code> — no <code>self</code> — so Dask can serialize
            and send it to remote workers.
          </p>
          <div class="role-methods">
            <span class="method-tag">@staticmethod</span>
            <span class="method-tag">pickle-safe</span>
          </div>
        </div>

        <div class="role-card c-provider">
          <p class="role-label">EnvironmentProvider</p>
          <h3>The initialization blueprint</h3>
          <p>
            Defines <em>how</em> and <em>where</em> to build the dataset and model —
            locally on the driver, or once per worker process.
            Two methods, two different timings.
          </p>
          <div class="role-methods">
            <span class="method-tag">build_env_local()</span>
            <span class="method-tag">build_worker_setup_fn()</span>
          </div>
        </div>

        <div class="role-card c-worker">
          <p class="role-label">Cluster / Client</p>
          <h3>The execution environment</h3>
          <p>
            Set once with <code>set_parallel(config)</code>. Changing <code>env</code>
            switches from running everything locally to submitting jobs to a SLURM cluster —
            no other code changes needed.
          </p>
          <div class="role-methods">
            <span class="method-tag">env: local</span>
            <span class="method-tag">env: slurm</span>
            <span class="method-tag">env: local_gpu</span>
          </div>
        </div>
      </div>

      <div class="callout callout-info">
        <span class="callout-icon">ℹ</span>
        <div>
          <p>
            <strong>You don't need to know Dask.</strong>
            ESPnet3 uses Dask internally, but you only interact with
            <code>set_parallel(config)</code>. The <code>BaseRunner</code> handles everything else.
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
