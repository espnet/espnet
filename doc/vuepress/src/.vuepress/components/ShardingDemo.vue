<script setup>
import { ref, computed } from 'vue'

// ---------- Section 01: shard rotation visualizer ----------
const totalShards = ref(8)
const worldSize = ref(4)
const epochsToShow = ref(6)

const clampedWorldSize = computed(() => Math.max(1, worldSize.value || 1))
const clampedTotalShards = computed(() => Math.max(1, totalShards.value || 1))
const clampedEpochs = computed(() => Math.max(2, Math.min(12, epochsToShow.value || 2)))

const isValid = computed(
  () => clampedTotalShards.value % clampedWorldSize.value === 0
)

const cycleLength = computed(() =>
  isValid.value ? clampedTotalShards.value / clampedWorldSize.value : null
)

const ranks = computed(() =>
  Array.from({ length: clampedWorldSize.value }, (_, i) => i)
)
const epochs = computed(() =>
  Array.from({ length: clampedEpochs.value }, (_, i) => i)
)

function shardIdx(epoch, rank) {
  return (epoch * clampedWorldSize.value + rank) % clampedTotalShards.value
}

const yamlOutput = computed(
  () =>
    `dataset:\n` +
    `  _target_: espnet3.components.data.data_organizer.DataOrganizer\n` +
    `  recipe_dir: \${recipe_dir}\n` +
    `  train:\n` +
    `    - data_src: egs3.my_recipe.asr.dataset.builder\n` +
    `      data_src_args:\n` +
    `        split: train\n` +
    `        total_shards: ${totalShards.value}\n` +
    `        dist_world_size: ${worldSize.value}`
)

// ---------- Section 03: code tabs ----------
const activeTab = ref('basic')
</script>

<template>
  <section class="guide-section" id="sharding-demo">
    <div class="container">
      <p class="section-desc">
        Three parts: an interactive visualizer for the shard rotation formula, a breakdown of what
        you implement versus what ESPnet3 does automatically, and code for the three dataset
        shapes you'll run into.
      </p>

      <!-- ============ 01: visualizer ============ -->
      <h3 style="font-size: 17px; margin-bottom: .5rem">01 · Shard rotation visualizer</h3>
      <p style="color: var(--text2); font-size: 14px; margin-bottom: 1.5rem; line-height: 1.75">
        Set <code>total_shards</code> and <code>dist_world_size</code> to match your training
        setup. The grid below shows exactly which shard each GPU rank receives, epoch by epoch,
        using <code>shard_idx = (epoch × world_size + rank) % total_shards</code>.
      </p>

      <div class="config-controls" style="margin-bottom: 1.5rem">
        <div class="slurm-grid">
          <div class="config-field">
            <label class="config-label">total_shards</label>
            <input type="number" v-model.number="totalShards" min="1" max="32" />
          </div>
          <div class="config-field">
            <label class="config-label">dist_world_size <span>GPUs</span></label>
            <input type="number" v-model.number="worldSize" min="1" max="8" />
          </div>
          <div class="config-field">
            <label class="config-label">epochs to preview</label>
            <input type="number" v-model.number="epochsToShow" min="2" max="12" />
          </div>
        </div>
      </div>

      <div v-if="!isValid" class="callout callout-warn">
        <span class="callout-icon">⚠</span>
        <div>
          <p>
            <strong>total_shards must be divisible by dist_world_size.</strong>
            {{ clampedTotalShards }} is not divisible by {{ clampedWorldSize }} — ESPnet3 raises a
            <code>RuntimeError</code> at startup for this combination. Adjust one of the values
            above to see the rotation.
          </p>
        </div>
      </div>

      <template v-else>
        <div class="shard-table-wrap">
          <table class="shard-table">
            <thead>
              <tr>
                <th>epoch \ rank</th>
                <th v-for="r in ranks" :key="r">GPU {{ r }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="e in epochs" :key="e">
                <td class="shard-table-row-label">epoch {{ e }}</td>
                <td v-for="r in ranks" :key="r">
                  <span class="idx-cell" :class="`shard-${shardIdx(e, r) % 4}`">
                    {{ shardIdx(e, r) }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="env-note" style="margin-bottom: 1.5rem">
          Cell values are shard indices, not colors — colors repeat every 4 shard ids purely for
          visual separation. No two ranks ever share a shard index within the same epoch.
        </div>

        <div class="callout callout-tip">
          <span class="callout-icon">✓</span>
          <div>
            <p>
              <strong>Full coverage every {{ cycleLength }} epoch{{ cycleLength === 1 ? '' : 's' }}.</strong>
              With <code>total_shards={{ totalShards }}</code> and
              <code>dist_world_size={{ worldSize }}</code>, each rank rotates through all
              {{ cycleLength }} of its reachable shards before repeating, and the union of all
              ranks covers the full dataset every epoch.
            </p>
          </div>
        </div>
      </template>

      <div class="config-output" style="margin-top: 1.5rem; margin-bottom: 2.5rem">
        <div class="config-output-header">
          <span>training.yaml</span>
          <span style="color: var(--worker); font-size: 11px">● live</span>
        </div>
        <pre><code>{{ yamlOutput }}</code></pre>
      </div>

      <div class="config-divider" />

      <!-- ============ 02: responsibility split ============ -->
      <h3 style="font-size: 17px; margin-bottom: .5rem">02 · Responsibility split</h3>
      <p style="color: var(--text2); font-size: 14px; margin-bottom: 1.5rem; line-height: 1.75">
        You write a small, mechanical <code>ShardedDataset</code> subclass. ESPnet3 decides
        <em>which</em> shard to ask for and <em>when</em> — you never call <code>shard()</code>
        yourself.
      </p>

      <div class="provider-split">
        <div class="provider-half ph-local">
          <div class="provider-half-header">
            <span>You implement</span>
            <span class="timing-badge">written once, per dataset class</span>
          </div>
          <div class="provider-half-body">
            <div class="step-line">
              <span class="step-dot dot-provider">1</span>
              <span>Set <code>total_shards</code> and <code>dist_world_size</code> as instance attributes</span>
            </div>
            <div class="step-line">
              <span class="step-dot dot-provider">2</span>
              <span>Implement <code>__len__</code> and <code>__getitem__</code> as for any PyTorch dataset</span>
            </div>
            <div class="step-line">
              <span class="step-dot dot-provider">3</span>
              <span>Implement <code>shard(shard_idx)</code> → return a <code>Dataset</code> covering only that shard</span>
            </div>
            <div class="closure-box">
              <div class="closure-title" style="color: var(--provider)">ShardedDataset subclass</div>
              <code>
                <span class="kw">class</span> <span class="fn">MyDataset</span>(ShardedDataset):<br>
                &nbsp;&nbsp;<span class="kw">def</span> <span class="fn">__init__</span>(self, ...):<br>
                &nbsp;&nbsp;&nbsp;&nbsp;self.total_shards = total_shards<br>
                &nbsp;&nbsp;&nbsp;&nbsp;self.dist_world_size = dist_world_size<br>
                <br>
                &nbsp;&nbsp;<span class="kw">def</span> <span class="fn">shard</span>(self, shard_idx):<br>
                &nbsp;&nbsp;&nbsp;&nbsp;<span class="kw">return</span> Subset(self, ...)
              </code>
            </div>
          </div>
        </div>

        <div class="provider-half ph-worker">
          <div class="provider-half-header">
            <span>ESPnet3 handles</span>
            <span class="timing-badge">runs automatically, every epoch</span>
          </div>
          <div class="provider-half-body">
            <div class="step-line">
              <span class="step-dot dot-worker">1</span>
              <span>At startup, validates <code>total_shards % world_size == 0</code> and that <code>dist_world_size</code> matches the runtime world size</span>
            </div>
            <div class="step-line">
              <span class="step-dot dot-worker">2</span>
              <span>Each epoch, computes <code>shard_idx = (epoch × world_size + rank) % total_shards</code> for this rank</span>
            </div>
            <div class="step-line">
              <span class="step-dot dot-worker">3</span>
              <span>Calls <code>dataset.shard(shard_idx)</code> once and builds the <code>DataLoader</code> from the result</span>
            </div>
            <div class="closure-box">
              <div class="closure-title" style="color: var(--worker)">DataLoaderBuilder._maybe_shard_dataset()</div>
              <code>
                <span style="color: var(--text3)"># once per epoch, per rank</span><br>
                shard_idx = (epoch * world_size + rank) % total_shards<br>
                shard = dataset.shard(shard_idx)<br>
                loader = DataLoader(shard, ...)
              </code>
            </div>
          </div>
        </div>
      </div>

      <div class="callout callout-info">
        <span class="callout-icon">ℹ</span>
        <div>
          <p>
            <strong>Single-GPU runs need none of this.</strong>
            Skip <code>ShardedDataset</code> entirely — a plain <code>Dataset</code> has no
            <code>total_shards</code> attribute, so it is returned unsharded. If your dataset does
            subclass <code>ShardedDataset</code>, set both <code>total_shards</code> and
            <code>dist_world_size</code> to <code>1</code>.
          </p>
        </div>
      </div>

      <div class="config-divider" />

      <!-- ============ 03: code tabs ============ -->
      <h3 style="font-size: 17px; margin-bottom: .5rem">03 · Code examples</h3>
      <p style="color: var(--text2); font-size: 14px; margin-bottom: 1.5rem; line-height: 1.75">
        The same dataset, at three levels of sharding: none, single-dataset, and multiple datasets
        combined in one split.
      </p>

      <div class="tab-row">
        <button class="tab-btn" :class="{ active: activeTab === 'basic' }" @click="activeTab = 'basic'">
          Basic dataset
        </button>
        <button class="tab-btn" :class="{ active: activeTab === 'sharded' }" @click="activeTab = 'sharded'">
          Sharded dataset
        </button>
        <button class="tab-btn" :class="{ active: activeTab === 'multi' }" @click="activeTab = 'multi'">
          Multiple datasets
        </button>
      </div>

      <template v-if="activeTab === 'basic'">
        <pre><code><span class="kw">from</span> torch.utils.data <span class="kw">import</span> Dataset


<span class="kw">class</span> <span class="fn">MyASRDataset</span>(Dataset):
    <span class="str">"""No total_shards attribute -> DataLoaderBuilder returns
    this dataset unsharded, unchanged, every epoch."""</span>

    <span class="kw">def</span> <span class="fn">__init__</span>(self, data_dir: str, split: str):
        self.samples = load_manifest(data_dir, split)

    <span class="kw">def</span> <span class="fn">__len__</span>(self) -> int:
        <span class="kw">return</span> len(self.samples)

    <span class="kw">def</span> <span class="fn">__getitem__</span>(self, idx: int) -> dict:
        item = self.samples[idx]
        <span class="kw">return</span> {
            <span class="str">"speech"</span>: load_audio(item[<span class="str">"path"</span>]),
            <span class="str">"text"</span>: item[<span class="str">"transcript"</span>],
        }</code></pre>
      </template>

      <template v-if="activeTab === 'sharded'">
        <pre><code><span class="kw">from</span> torch.utils.data <span class="kw">import</span> Dataset, Subset

<span class="kw">from</span> espnet3.components.data.dataset <span class="kw">import</span> ShardedDataset


<span class="kw">class</span> <span class="fn">MyASRDataset</span>(ShardedDataset):

    <span class="kw">def</span> <span class="fn">__init__</span>(
        self,
        data_dir: str,
        split: str,
        total_shards: int = <span class="num">8</span>,
        dist_world_size: int = <span class="num">4</span>,
    ):
        self.samples = load_manifest(data_dir, split)
        self.total_shards = total_shards
        self.dist_world_size = dist_world_size

    <span class="kw">def</span> <span class="fn">__len__</span>(self) -> int:
        <span class="cm"># total across ALL shards — DataLoaderBuilder never calls this directly</span>
        <span class="kw">return</span> len(self.samples)

    <span class="kw">def</span> <span class="fn">__getitem__</span>(self, idx: int) -> dict:
        item = self.samples[idx]
        <span class="kw">return</span> {
            <span class="str">"speech"</span>: load_audio(item[<span class="str">"path"</span>]),
            <span class="str">"text"</span>: item[<span class="str">"transcript"</span>],
        }

    <span class="kw">def</span> <span class="fn">shard</span>(self, shard_idx: int) -> Dataset:
        n = len(self.samples)
        shard_size = n // self.total_shards
        start = shard_idx * shard_size
        <span class="kw">return</span> Subset(self, list(range(start, start + shard_size)))</code></pre>
      </template>

      <template v-if="activeTab === 'multi'">
        <pre><code><span style="color: var(--text3)"># CombinedDataset requires every dataset in a split to:
#  1. subclass ShardedDataset (no mixing with plain Dataset)
#  2. agree on total_shards and dist_world_size</span>

train:
  - data_src: egs3.my_recipe.asr.dataset.builder   <span style="color: var(--text3)"># total_shards=8</span>
    data_src_args:
      split: train
      total_shards: <span class="num">8</span>
      dist_world_size: <span class="num">4</span>
  - data_src: egs3.my_recipe.asr.dataset.extra      <span style="color: var(--text3)"># must match ← 8</span>
    data_src_args:
      split: train
      total_shards: <span class="num">8</span>
      dist_world_size: <span class="num">4</span>

<span style="color: var(--text3)"># CombinedDataset.shard(shard_idx) calls dataset.shard(shard_idx) on
# every component dataset and wraps the results in a new CombinedDataset
# of the same shape. It also asserts all datasets return the same
# __getitem__ keys, with or without sharding.</span></code></pre>
      </template>
    </div>
  </section>
</template>

<style scoped>
.shard-table-wrap {
  overflow-x: auto;
  margin-bottom: 1rem;
}

.shard-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.shard-table th,
.shard-table td {
  padding: 6px 8px;
  border: 1px solid var(--border);
  text-align: center;
  white-space: nowrap;
}

.shard-table th {
  background: var(--bg2);
  color: var(--text3);
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 400;
}

.shard-table-row-label {
  color: var(--text2);
  font-family: var(--mono);
  text-align: left !important;
}
</style>
