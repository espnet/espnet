<script setup>
import { ref } from 'vue'

const activeTab = ref('abstract')
</script>

<template>
  <section class="guide-section" id="provider">
    <div class="container">
      <p class="section-desc">
        The biggest source of confusion with <code>EnvironmentProvider</code> is that the two
        methods are called at completely different times and places.
      </p>

      <div class="provider-split">
        <!-- build_env_local -->
        <div class="provider-half ph-local">
          <div class="provider-half-header">
            <span>build_env_local()</span>
            <span class="timing-badge">called: on the driver, right now</span>
          </div>
          <div class="provider-half-body">
            <div class="step-line">
              <span class="step-dot dot-provider">1</span>
              <span>You call <code>runner(indices)</code>…</span>
            </div>
            <div class="step-line">
              <span class="step-dot dot-provider">2</span>
              <span><code>_run_local()</code> calls <code>provider.build_env_local()</code> directly</span>
            </div>
            <div class="step-line">
              <span class="step-dot dot-provider">3</span>
              <span>The returned <code>dict</code> (dataset, model, ...) is used immediately</span>
            </div>
            <div class="closure-box">
              <div class="closure-title" style="color: var(--provider)">local mode flow</div>
              <code>
                <span style="color: var(--provider)">env = provider.build_env_local()</span><br>
                <span style="color: var(--text3)"># dict is returned immediately</span><br>
                <span style="color: var(--text2)">forward(idx, **env)</span>
              </code>
            </div>
          </div>
        </div>

        <!-- build_worker_setup_fn -->
        <div class="provider-half ph-worker">
          <div class="provider-half-header">
            <span>build_worker_setup_fn()</span>
            <span class="timing-badge">called: on each worker at startup</span>
          </div>
          <div class="provider-half-body">
            <div class="step-line">
              <span class="step-dot dot-worker">1</span>
              <span>Driver calls <code>provider.build_worker_setup_fn()</code> → returns a <strong>function</strong></span>
            </div>
            <div class="step-line">
              <span class="step-dot dot-worker">2</span>
              <span>That function is wrapped in <code>DictReturnWorkerPlugin</code> and sent to workers</span>
            </div>
            <div class="step-line">
              <span class="step-dot dot-worker">3</span>
              <span>When each worker starts, <code>setup()</code> is called — dataset/model are built on the worker</span>
            </div>
            <div class="closure-box">
              <div class="closure-title" style="color: var(--worker)">distributed mode flow</div>
              <code>
                <span style="color: var(--worker)">setup_fn = provider.build_worker_setup_fn()</span><br>
                <span style="color: var(--text3)"># nothing is initialized yet</span><br>
                <span style="color: var(--text3)"># setup_fn is a closure</span><br>
                <span style="color: var(--driver)">plugin = DictReturnWorkerPlugin(setup_fn)</span><br>
                <span style="color: var(--text3)"># when a worker starts:</span><br>
                <span style="color: var(--forward)">env = setup_fn()  # runs on the worker</span>
              </code>
            </div>
          </div>
        </div>
      </div>

      <div class="callout callout-warn">
        <span class="callout-icon">⚠</span>
        <div>
          <p>
            <strong>Why return a function instead of the env directly?</strong><br>
            SLURM workers run on different machines. If the driver initialized the model and tried
            to send it, <strong>the full model weights would need to be serialized and transferred
            over the network</strong>. Instead, only the lightweight initialization recipe
            (closure) is sent, and each worker builds its own copy locally — significantly
            reducing transfer overhead.
          </p>
        </div>
      </div>

      <h3 style="font-size: 17px; margin-bottom: .75rem">
        Automatic env injection: <code>wrap_func_with_worker_env</code>
      </h3>
      <p style="color: var(--text2); font-size: 14px; margin-bottom: 1.25rem; line-height: 1.75">
        When calling <code>forward()</code> on a worker, you don't pass <code>dataset</code> or
        <code>model</code> explicitly. <code>wrap_func_with_worker_env</code> inspects the
        function signature and <strong>matches argument names against the worker's env keys</strong>
        — injecting them automatically.
      </p>
      <div class="inject-demo">
        <div style="margin-bottom: .5rem; color: var(--text3); font-size: 11px; text-transform: uppercase; letter-spacing: .06em">
          Argument names matching env keys are injected automatically
        </div>
        <code>
          @staticmethod<br>
          <span class="kw">def</span> <span class="fn">forward</span>(<span class="inject-auto">idx</span>, <span class="inject-match">dataset</span>, <span class="inject-match">model</span>, <span class="inject-match">device</span>, **env):<br>
          <span style="color: var(--text3)">    # dataset, model, device → injected from worker env</span><br>
          <span style="color: var(--text3)">    # idx → passed by client.map()</span><br>
          <br>
          <span style="color: var(--text3)"># Worker env (returned by setup_fn()):</span><br>
          <span class="inject-match">env = { "dataset": ds, "model": md, "device": dev, ... }</span><br>
          <span style="color: var(--text3)">       ↑ key names match forward's argument names → auto-injected</span>
        </code>
      </div>

      <!-- Tab section -->
      <div style="margin-top: 2rem">
        <h3 style="font-size: 17px; margin-bottom: .75rem">
          Implementation example: subclassing <code>InferenceProvider</code>
        </h3>
        <div class="tab-row">
          <button
            class="tab-btn"
            :class="{ active: activeTab === 'abstract' }"
            @click="activeTab = 'abstract'"
          >EnvironmentProvider (base)</button>
          <button
            class="tab-btn"
            :class="{ active: activeTab === 'inference' }"
            @click="activeTab = 'inference'"
          >InferenceProvider (middle)</button>
          <button
            class="tab-btn"
            :class="{ active: activeTab === 'custom' }"
            @click="activeTab = 'custom'"
          >Custom implementation</button>
        </div>

        <template v-if="activeTab === 'abstract'">
          <pre><code><span class="kw">class</span> <span class="fn">EnvironmentProvider</span>(ABC):
    <span class="str">"""Abstract base — implement both methods."""</span>

    @abstractmethod
    <span class="kw">def</span> <span class="fn">build_env_local</span>(self) -> dict:
        <span class="str">"""Local execution: called on the driver immediately."""</span>
        ...

    @abstractmethod
    <span class="kw">def</span> <span class="fn">build_worker_setup_fn</span>(self) -> Callable:
        <span class="str">"""Distributed execution: the returned function is called on each worker."""</span>
        ...</code></pre>
        </template>

        <template v-if="activeTab === 'inference'">
          <pre><code><span class="kw">class</span> <span class="fn">InferenceProvider</span>(EnvironmentProvider, ABC):
    <span class="str">"""Convenience base class for inference tasks."""</span>

    <span class="kw">def</span> __init__(self, config):
        super().__init__(config)
        <span class="cm"># Build once for local use — avoids redundant I/O</span>
        self._local_env = self.build_worker_setup_fn()()

    @staticmethod @abstractmethod
    <span class="kw">def</span> <span class="fn">build_dataset</span>(config): ...

    @staticmethod @abstractmethod
    <span class="kw">def</span> <span class="fn">build_model</span>(config): ...

    <span class="kw">def</span> <span class="fn">build_env_local</span>(self):
        <span class="cm"># Return the already-built env</span>
        <span class="kw">return</span> dict(self._local_env)

    <span class="kw">def</span> <span class="fn">build_worker_setup_fn</span>(self):
        config = self.config
        <span class="kw">def</span> <span class="fn">setup</span>():
            <span class="cm"># ← this runs on the worker</span>
            <span class="kw">return</span> {
                <span class="str">"dataset"</span>: self.build_dataset(config),
                <span class="str">"model"</span>:   self.build_model(config),
            }
        <span class="kw">return</span> setup  <span class="cm"># ← return the function, don't call it!</span></code></pre>
        </template>

        <template v-if="activeTab === 'custom'">
          <pre><code><span class="kw">class</span> <span class="fn">MyASRProvider</span>(InferenceProvider):
    <span class="str">"""Example Provider for ASR inference."""</span>

    @staticmethod
    <span class="kw">def</span> <span class="fn">build_dataset</span>(config):
        <span class="kw">return</span> ESPnetASRDataset(config.data_path)

    @staticmethod
    <span class="kw">def</span> <span class="fn">build_model</span>(config):
        model = ESPnetModel.from_pretrained(config.model_tag)
        <span class="kw">return</span> model.eval()

<span class="cm"># Usage</span>
provider = MyASRProvider(config)
runner = MyASRRunner(provider, output_dir=<span class="str">"exp/decode"</span>)
runner(range(len(dataset)))  <span class="cm"># that's it!</span></code></pre>
        </template>
      </div>
    </div>
  </section>
</template>
