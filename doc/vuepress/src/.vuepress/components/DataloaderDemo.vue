<template>
  <div class="dl-demo">
    <!-- Tabs -->
    <div class="tabs">
      <div
        v-for="tab in tabs"
        :key="tab.id"
        class="tab"
        :class="{ active: activeTab === tab.id }"
        @click="switchTab(tab.id)"
      >{{ tab.label }}</div>
    </div>

    <!-- ① Overview -->
    <div v-show="activeTab === 'flow'" class="tab-content">
      <div class="section-title">Pipeline overview</div>
      <div class="card">
        <div class="card-title">Data flow in the ESPnet3 training pipeline</div>
        <div class="card-desc" style="margin-bottom:18px">
          The <code>collect_stats</code> stage generates all files consumed by the subsequent
          <code>train</code> stage. Both stages share the same <code>dataloader:</code> block
          in <code>training.yaml</code>.
        </div>
        <div class="flow">
          <div class="flow-node highlight">collect_stats<br><span class="node-sub">model.collect_feats()</span></div>
          <div class="flow-arrow">→</div>
          <div class="flow-node out">feats_shape<br><span class="node-sub">stats_dir/train/</span></div>
          <div class="flow-arrow">+</div>
          <div class="flow-node out">feats_stats.npz<br><span class="node-sub">for GlobalMVN</span></div>
          <div class="flow-arrow">→</div>
          <div class="flow-node">train<br><span class="node-sub">ESPnetLightningModule</span></div>
        </div>
      </div>

      <div class="grid2">
        <div class="card">
          <div class="card-title">① feats_shape — for batching</div>
          <div class="card-desc">
            A text file recording the number of frames per sample.<br>
            Passed to <code>batches.shape_files</code> in <code>SequenceIterFactory</code>,
            enabling length-aware batching (reduced padding, OOM prevention).
          </div>
          <pre style="margin-top:12px"><span class="cmt"># stats_dir/train/feats_shape</span>
utt_001 312
utt_002 489
utt_003 156
utt_004 701
...</pre>
        </div>
        <div class="card">
          <div class="card-title">② feats_stats.npz — for normalization</div>
          <div class="card-desc">
            A compressed numpy archive storing dataset-wide mean / variance.<br>
            Referenced by <code>model.normalize: global_mvn</code>.
            Written by <code>collect_stats</code>, read during <code>train</code>.
          </div>
          <pre style="margin-top:12px"><span class="kw">model</span>:
  <span class="key">normalize</span>: <span class="val">global_mvn</span>
  <span class="key">normalize_conf</span>:
    <span class="key">stats_file</span>: <span class="str">${stats_dir}/train/feats_stats.npz</span></pre>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Shared structure of training.yaml</div>
        <div class="card-desc" style="margin-bottom:12px">
          The <code>dataloader:</code> block is read by both <code>collect_stats</code> and
          <code>train</code>. Variable interpolation (<code>${stats_dir}/...</code>) ensures
          that collect_stats output paths and train input paths always match automatically.
        </div>
        <pre><span class="key">stats_dir</span>: <span class="str">${exp_dir}/stats</span>

<span class="kw">dataset</span>:
  <span class="key">_target_</span>: <span class="val">espnet3.components.data.data_organizer.DataOrganizer</span>
  <span class="key">train</span>: [...]
  <span class="key">valid</span>: [...]

<span class="kw">dataloader</span>:
  <span class="key">collate_fn</span>:
    <span class="key">_target_</span>: <span class="val">espnet2.train.collate_fn.CommonCollateFn</span>
    <span class="key">int_pad_value</span>: <span class="val">-1</span>
  <span class="key">train</span>:
    <span class="key">iter_factory</span>:
      <span class="key">_target_</span>: <span class="val">espnet2.iterators.sequence_iter_factory.SequenceIterFactory</span>
      <span class="key">batches</span>:
        <span class="key">type</span>: <span class="val">numel</span>
        <span class="key">shape_files</span>:
          - <span class="str">${stats_dir}/train/feats_shape</span>  <span class="cmt"># written by collect_stats</span>
        <span class="key">batch_bins</span>: <span class="val">1200000</span>

<span class="kw">model</span>:
  <span class="key">normalize_conf</span>:
    <span class="key">stats_file</span>: <span class="str">${stats_dir}/train/feats_stats.npz</span>  <span class="cmt"># written by collect_stats</span></pre>
      </div>
    </div>

    <!-- ② iter_factory on/off -->
    <div v-show="activeTab === 'iter'" class="tab-content">
      <div class="section-title">with vs without iter_factory</div>
      <div>
        <div class="card">
          <div class="card-title">
            <span class="badge badge-coral">iter_factory: null</span>
            &nbsp;Standard PyTorch DataLoader
          </div>
          <div class="card-desc" style="margin-bottom:12px">
            Setting <code>iter_factory</code> to <code>null</code> falls back to a plain
            PyTorch DataLoader. Simple to configure, but padding grows quickly when sample
            lengths vary. Use when <code>collect_stats</code> is unnecessary.
          </div>
          <pre><span class="kw">dataloader</span>:
  <span class="key">collate_fn</span>:
    <span class="key">_target_</span>: <span class="val">espnet2.train.collate_fn.CommonCollateFn</span>
    <span class="key">int_pad_value</span>: <span class="val">-1</span>
  <span class="key">train</span>:
    <span class="key">iter_factory</span>: <span class="val">null</span>   <span class="cmt"># ESPnet iterator disabled</span>
    <span class="key">batch_size</span>: <span class="val">8</span>
    <span class="key">num_workers</span>: <span class="val">4</span>
    <span class="key">shuffle</span>: <span class="val">true</span>
  <span class="key">valid</span>:
    <span class="key">iter_factory</span>: <span class="val">null</span>
    <span class="key">batch_size</span>: <span class="str">${dataloader.train.batch_size}</span>
    <span class="key">num_workers</span>: <span class="str">${dataloader.train.num_workers}</span>
    <span class="key">shuffle</span>: <span class="val">false</span></pre>
          <div class="warn-box">
            ⚠ Fixed batch size of 8. When long and short samples are mixed, padding on
            short samples grows, hurting both GPU memory and throughput.
          </div>
        </div>

        <div class="card" style="border-color:var(--accent)">
          <div class="card-title">
            <span class="badge badge-blue">iter_factory enabled</span>
            &nbsp;ESPnet IteratorFactory
          </div>
          <div class="card-desc" style="margin-bottom:12px">
            <code>SequenceIterFactory</code> + a batch sampler automatically builds
            length-aware batches from <code>feats_shape</code>.
            The seed is fixed per epoch (<code>seed + epoch</code>), so training is
            fully reproducible across restarts.
          </div>
          <pre><span class="kw">dataloader</span>:
  <span class="key">collate_fn</span>:
    <span class="key">_target_</span>: <span class="val">espnet2.train.collate_fn.CommonCollateFn</span>
    <span class="key">int_pad_value</span>: <span class="val">-1</span>
  <span class="key">train</span>:
    <span class="key">iter_factory</span>:
      <span class="key">_target_</span>: <span class="val">espnet2.iterators.sequence_iter_factory.SequenceIterFactory</span>
      <span class="key">shuffle</span>: <span class="val">true</span>
      <span class="key">collate_fn</span>: <span class="str">${dataloader.collate_fn}</span>
      <span class="key">batches</span>:
        <span class="key">type</span>: <span class="val">numel</span>        <span class="cmt"># ← choose strategy here</span>
        <span class="key">batch_bins</span>: <span class="val">1200000</span>
        <span class="key">shape_files</span>:
          - <span class="str">${stats_dir}/train/feats_shape</span>
  <span class="key">valid</span>:
    <span class="key">iter_factory</span>:
      <span class="key">_target_</span>: <span class="val">espnet2.iterators.sequence_iter_factory.SequenceIterFactory</span>
      <span class="key">shuffle</span>: <span class="val">false</span>
      <span class="key">collate_fn</span>: <span class="str">${dataloader.collate_fn}</span>
      <span class="key">batches</span>:
        <span class="key">type</span>: <span class="str">${dataloader.train.iter_factory.batches.type}</span>
        <span class="key">batch_bins</span>: <span class="str">${dataloader.train.iter_factory.batches.batch_bins}</span>
        <span class="key">shape_files</span>:
          - <span class="str">${stats_dir}/valid/feats_shape</span></pre>
          <div class="info-box">
            ✓ Requires <code>collect_stats</code>. Once <code>feats_shape</code> is produced,
            batches are formed with variable sizes based on sequence length.
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">build_iter() call flow</div>
        <div class="card-desc" style="margin-bottom:12px">
          <code>ESPnetLightningModule</code> calls <code>iter_factory.build_iter(epoch)</code>
          at the start of each epoch. Because the seed is fixed as <code>seed + epoch</code>,
          the same batch order is reproduced when training is resumed.
        </div>
        <pre><span class="cmt"># How it's used internally (espnet3/components/modeling/lightning_module.py)</span>
<span class="kw">for</span> epoch <span class="kw">in</span> range(max_epoch):
    iterator = iter_factory.build_iter(epoch)   <span class="cmt"># seed = base_seed + epoch</span>
    <span class="kw">for</span> uids, batch <span class="kw">in</span> iterator:
        model(**batch)                           <span class="cmt"># speech, speech_lengths, text, ...</span></pre>
      </div>
    </div>

    <!-- ③ Batch strategy simulator -->
    <div v-show="activeTab === 'sim'" class="tab-content">
      <div class="section-title">Batch strategy simulator</div>
      <div class="card-desc" style="margin-bottom:18px;color:var(--text2)">
        50 audio samples (1–15 s) visualised in real time. Same colour = same batch.
      </div>

      <div class="sim-controls">
        <div class="ctrl-group">
          <span class="ctrl-label">BATCH TYPE</span>
          <select v-model="simType" @change="runSim">
            <option value="torch_dl">torch DataLoader (shuffle)</option>
            <option value="unsorted">unsorted</option>
            <option value="sorted">sorted</option>
            <option value="folded">folded</option>
            <option value="length">length (batch_bins)</option>
            <option value="numel">numel (batch_bins)</option>
          </select>
        </div>
        <div v-if="showBsize" class="ctrl-group">
          <span class="ctrl-label">BATCH_SIZE</span>
          <input type="number" v-model.number="simBatchSize" min="1" max="50" style="width:60px" @input="runSim">
        </div>
        <div v-if="showBins" class="ctrl-group">
          <span class="ctrl-label">BATCH_BINS</span>
          <input type="range" v-model.number="simBins" min="500" max="6000" @input="runSim">
          <span class="mono-label">{{ simBins }}</span>
        </div>
        <div v-if="showFold" class="ctrl-group">
          <span class="ctrl-label">FOLD_LENGTH</span>
          <input type="range" v-model.number="simFoldLen" min="100" max="900" @input="runSim">
          <span class="mono-label">{{ simFoldLen }}</span>
          <span class="ctrl-label" style="margin-left:8px">BASE_BATCH</span>
          <input type="number" v-model.number="simFoldBase" min="1" max="50" style="width:52px" @input="runSim">
        </div>
        <button @click="reseed" class="reseed-btn">↺ reseed</button>
      </div>

      <div v-if="simType === 'torch_dl'" class="warn-box" style="margin-bottom:12px">
        ⚠ This is pure PyTorch DataLoader behaviour when <code>iter_factory: null</code> is set.
        Samples are shuffled randomly then cut into fixed-size batches — length variance within
        a batch is high and padding increases.
      </div>

      <div class="sim-stats" style="margin-bottom:12px">
        <div class="stat-item"><span class="stat-label">Batches</span><span class="stat-value">{{ simStats.nBatch }}</span></div>
        <div class="stat-item"><span class="stat-label">Avg batch size</span><span class="stat-value">{{ simStats.avgBs }}</span></div>
        <div class="stat-item"><span class="stat-label">Avg padding rate</span><span class="stat-value">{{ simStats.padRate }}%</span></div>
        <div class="stat-item"><span class="stat-label">Max length diff in batch</span><span class="stat-value">{{ simStats.maxDiff }}f</span></div>
      </div>

      <pre style="margin-bottom:16px" v-html="simYaml"></pre>

      <div class="sim-canvas">
        <div class="sim-samples">
          <template v-for="(item, i) in simDisplay" :key="i">
            <div v-if="item.type === 'divider'" class="batch-divider">
              <span class="batch-label">batch {{ item.batchIdx + 1 }} ({{ item.count }} utts)</span>
            </div>
            <div v-else class="sample-row">
              <span class="sample-id">{{ item.id }}</span>
              <div class="sample-bar-wrap">
                <div class="sample-bar" :style="{ width: (item.len / simMaxLen * 100) + '%', background: item.color, opacity: 0.75 }"></div>
              </div>
              <span class="sample-len">{{ item.len }}f</span>
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- ④ ChunkIterFactory -->
    <div v-show="activeTab === 'chunk'" class="tab-content">
      <div class="section-title">ChunkIterFactory — splitting long sequences into fixed-length chunks</div>

      <div class="card">
        <div class="card-title">Concept</div>
        <div class="card-desc">
          Long audio is split into fixed-length windows (<code>chunk_length</code>) and batches
          are formed from those windows. Used in SpeechLM and long-form audio models.
          Because the model always sees fixed-length input, <code>feats_shape</code> is not required.
        </div>
      </div>

      <div class="sim-controls">
        <div class="ctrl-group">
          <span class="ctrl-label">CHUNK_LENGTH</span>
          <input type="range" v-model.number="chunkLen" min="100" max="800" @change="drawChunks">
          <span class="mono-label">{{ chunkLen }}</span>
        </div>
        <div class="ctrl-group">
          <span class="ctrl-label">SHIFT_RATIO</span>
          <input type="range" v-model.number="chunkShiftPct" min="10" max="100" step="10" @change="drawChunks">
          <span class="mono-label">{{ (chunkShiftPct/100).toFixed(1) }}</span>
        </div>
        <div class="ctrl-group">
          <span class="ctrl-label">BATCH_SIZE</span>
          <input type="number" v-model.number="chunkBatchSize" min="1" max="12" style="width:52px" @change="drawChunks">
        </div>
        <div class="ctrl-group" style="margin-left:auto">
          <span class="ctrl-label">POOL SHUFFLE</span>
          <label style="display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none;font-size:12px;color:var(--text2)">
            <input type="checkbox" v-model="chunkShuffle" @change="drawChunks" style="accent-color:var(--accent);width:14px;height:14px">
            <span>{{ chunkShuffle ? 'on' : 'off' }}</span>
          </label>
        </div>
        <button @click="reseedChunks" class="reseed-btn">↺ reseed</button>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start;margin-top:4px">
        <div>
          <div class="section-title" style="margin-bottom:8px">Utterance → chunk splitting</div>
          <div class="card-desc" style="font-size:11px;margin-bottom:10px;color:var(--text2)">
            Each utterance is sliced at <code>chunk_length</code>. Same colour = same utterance.
          </div>
          <div ref="chunkVisRef"></div>
        </div>
        <div>
          <div class="section-title" style="margin-bottom:8px">Chunk pool → batch assembly</div>
          <div class="card-desc" style="font-size:11px;margin-bottom:10px;color:var(--text2)">
            All chunks are pooled then grouped <code>batch_size</code> at a time.
            Every batch is the same length — <strong style="color:var(--accent)">padding = 0</strong>.
          </div>
          <div ref="chunkBatchVisRef"></div>
        </div>
      </div>

      <div class="sim-stats" style="margin-top:14px">
        <div class="stat-item"><span class="stat-label">Total chunks</span><span class="stat-value" style="font-size:13px">{{ chunkStats.totalChunks }}</span></div>
        <div class="stat-item"><span class="stat-label">Total batches</span><span class="stat-value" style="font-size:13px">{{ chunkStats.totalBatches }}</span></div>
        <div class="stat-item"><span class="stat-label">chunk_length</span><span class="stat-value" style="font-size:13px;font-family:var(--font-family-code)">{{ chunkLen }}f</span></div>
        <div class="stat-item"><span class="stat-label">shift</span><span class="stat-value" style="font-size:13px;font-family:var(--font-family-code)">{{ Math.round(chunkLen * chunkShiftPct/100) }}f (×{{ (chunkShiftPct/100).toFixed(1) }})</span></div>
        <div class="stat-item"><span class="stat-label">padding</span><span class="stat-value" style="font-size:13px;color:var(--green)">0</span></div>
        <div class="stat-item"><span class="stat-label">feats_shape</span><span class="stat-value" style="font-size:13px;color:var(--green)">not needed</span></div>
      </div>

      <hr>

      <div class="section-title" style="margin-bottom:10px">YAML config</div>
      <div id="chunk-yaml-display" v-html="chunkYaml"></div>

      <div class="section-title" style="margin-bottom:10px;margin-top:16px">CategoryChunkIterFactory (category-aware)</div>
      <pre><span class="kw">dataloader</span>:
  <span class="key">train</span>:
    <span class="key">iter_factory</span>:
      <span class="key">_target_</span>: <span class="val">espnet2.iterators.category_chunk_iter_factory.CategoryChunkIterFactory</span>
      <span class="key">batch_size</span>: <span class="val">8</span>
      <span class="key">chunk_length</span>: <span class="val">800</span>
      <span class="key">batch_type</span>: <span class="val">catbel</span>
      <span class="key">sampler_args</span>:
        <span class="key">category2utt_file</span>: <span class="str">${stats_dir}/train/utt2category</span>
        <span class="key">batch_size</span>: <span class="val">8</span></pre>
      <div class="info-box" style="margin-top:8px">
        Use when you need category-balanced batches on long sequences
        (e.g. speaker-balanced SpeechLM training).
      </div>
    </div>

    <!-- ⑤ Category iterators -->
    <div v-show="activeTab === 'cat'" class="tab-content">
      <div class="section-title">Category iterators — balancing by speaker, language, dataset, etc.</div>

      <div class="grid3" style="margin-bottom:20px">
        <div class="card">
          <div class="card-title"><span class="badge badge-blue">catbel</span>&nbsp;CategoryBalanced</div>
          <div class="card-desc">
            Round-robin sampling so each batch contains an even mix of categories
            (speakers / languages / etc.). Best when you have a <strong>single dataset</strong>
            with class imbalance.
          </div>
        </div>
        <div class="card">
          <div class="card-title"><span class="badge badge-purple">catpow</span>&nbsp;CategoryPower</div>
          <div class="card-desc">
            Power-law upsampling with <code>upsampling_factor</code> to boost under-resourced
            categories. Use within a <strong>single dataset</strong> when per-category data
            volumes differ (e.g. uneven speaker recording hours).
            Does not account for cross-dataset imbalance.
          </div>
        </div>
        <div class="card">
          <div class="card-title"><span class="badge badge-teal">catpow_bal_ds</span>&nbsp;DatasetPower</div>
          <div class="card-desc">
            Two-stage balancing for <strong>multi-dataset training</strong>:
            ① <code>dataset_upsampling_factor</code> corrects the volume gap between datasets,
            ② <code>category_upsampling_factor</code> further corrects within-category skew.
            Example: LibriSpeech (1800 h) + CommonVoice (30 h) in a multilingual ASR recipe.
            Requires <code>utt2dataset</code> / <code>dataset2utt</code>.
          </div>
        </div>
      </div>

      <div class="sim-controls">
        <div class="ctrl-group">
          <span class="ctrl-label">MODE</span>
          <select v-model="catMode" @change="drawCat">
            <option value="none">no balancing (unsorted)</option>
            <option value="catbel">catbel — balanced</option>
            <option value="catpow">catpow — power-law</option>
            <option value="catpow_bal_ds">catpow_balance_dataset — dataset balancing</option>
          </select>
        </div>
        <div v-if="catMode === 'catpow' || catMode === 'catpow_bal_ds'" class="ctrl-group">
          <span class="ctrl-label">UPSAMPLING_FACTOR</span>
          <input type="range" v-model.number="catAlphaTimes10" min="1" max="10" @input="drawCat">
          <span class="mono-label">{{ (catAlphaTimes10/10).toFixed(1) }}</span>
        </div>
        <div class="ctrl-group">
          <span class="ctrl-label">BATCH_SIZE</span>
          <input type="number" v-model.number="catBatchSize" min="5" max="30" style="width:55px" @input="drawCat">
        </div>
      </div>

      <div class="legend">
        <div v-for="(color, spk) in SPK_COLORS" :key="spk" class="legend-item">
          <div class="legend-dot" :style="{ background: color }"></div>
          {{ spk }} ({{ catUttCounts[spk] }})
        </div>
      </div>

      <div class="cat-canvas">
        <div v-for="(batch, bi) in catBatches.slice(0,5)" :key="bi" class="cat-row">
          <span class="cat-id">batch {{ bi+1 }}</span>
          <div class="cat-bar-bg">
            <div
              v-for="seg in catSegments(batch)"
              :key="seg.spk"
              class="cat-bar-fill"
              :style="{ left: seg.left + '%', width: seg.width + '%', background: SPK_COLORS[seg.spk], opacity: 0.8 }"
            ></div>
          </div>
          <span class="cat-spk-label" style="font-size:9px;color:var(--text3)">{{ catBatchLabel(batch) }}</span>
        </div>
      </div>

      <div v-if="catMode === 'catpow_bal_ds'" class="info-box" style="margin-top:10px">
        <strong>catpow_balance_dataset</strong> additionally requires:<br>
        <code>stats_dir/train/dataset2utt</code> (dataset → utt list) and
        <code>stats_dir/train/utt2dataset</code> (utt → dataset name)
      </div>

      <hr>
      <div class="section-title" style="margin-top:0">YAML config example</div>
      <pre v-html="catYaml"></pre>

      <hr>
      <div class="section-title">Required file: utt2category</div>
      <pre><span class="cmt"># stats_dir/train/utt2category  (category → utt id list)</span>
spk_A utt_001 utt_006 utt_011 utt_016
spk_B utt_002 utt_007 utt_012
spk_C utt_003 utt_008
spk_D utt_004 utt_009 utt_014 utt_019 utt_024
spk_E utt_005 utt_010 utt_015</pre>

      <div class="grid2" style="margin-top:16px">
        <pre><span class="cmt"># utt2dataset</span>
utt_001 librispeech
utt_002 commonvoice
utt_003 librispeech
...</pre>
        <pre><span class="cmt"># dataset2utt</span>
librispeech utt_001 utt_003 utt_005 ...
commonvoice utt_002 utt_004 ...</pre>
      </div>
    </div>

    <!-- ⑥ Required files -->
    <div v-show="activeTab === 'files'" class="tab-content">
      <div class="section-title">Required files — quick reference</div>
      <div class="card-desc" style="margin-bottom:16px">
        Summary of files required by each batching strategy. ✓ = required, — = not needed.
      </div>
      <div style="overflow-x:auto">
        <table class="files-table">
          <thead>
            <tr>
              <th>sampler / mode</th>
              <th>iterator</th>
              <th><span class="filename">feats_shape</span></th>
              <th><span class="filename">utt2category</span></th>
              <th><span class="filename">dataset2utt<br>utt2dataset</span></th>
              <th>primary use case</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in fileRows" :key="row.mode">
              <td><span :class="['badge', row.badgeClass]">{{ row.mode }}</span></td>
              <td>{{ row.iterator }}</td>
              <td><span :class="row.shape ? 'check' : 'dash'">{{ row.shape ? '✓' : '—' }}</span></td>
              <td><span :class="row.utt2cat ? 'check' : 'dash'">{{ row.utt2cat ? '✓' : '—' }}</span></td>
              <td><span :class="row.ds ? 'check' : 'dash'">{{ row.ds ? '✓' : '—' }}</span></td>
              <td>{{ row.desc }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'

// ── constants ──────────────────────────────────────────────────────────────
const COLORS = [
  '#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444',
  '#06b6d4','#a855f7','#6ee7b7','#fde68a','#fca5a5',
  '#93c5fd','#c4b5fd','#86efac','#fef08a','#fecaca',
  '#60a5fa','#818cf8','#4ade80','#facc15','#fb923c',
]
const CHUNK_COLORS = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#a855f7','#84cc16']
const SPK_COLORS = { spk_A:'#2563eb', spk_B:'#7c3aed', spk_C:'#059669', spk_D:'#d97706', spk_E:'#dc2626' }

// ── tabs ───────────────────────────────────────────────────────────────────
const tabs = [
  { id:'flow',  label:'① Overview' },
  { id:'iter',  label:'② iter_factory on/off' },
  { id:'sim',   label:'③ Batch strategy simulator' },
  { id:'chunk', label:'④ ChunkIterFactory' },
  { id:'cat',   label:'⑤ Category iterators' },
  { id:'files', label:'⑥ Required files' },
]
const activeTab = ref('flow')
function switchTab(id) {
  activeTab.value = id
  if (id === 'sim')   nextTick(runSim)
  if (id === 'chunk') nextTick(drawChunks)
  if (id === 'cat')   nextTick(drawCat)
}

// ── seeded RNG helper ──────────────────────────────────────────────────────
function makeRng(seed) {
  let s = seed
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff }
}

// ══════════════════════════════════════════════════════════════════════════
// TAB 3 — Batch strategy simulator
// ══════════════════════════════════════════════════════════════════════════
const simType      = ref('numel')
const simBatchSize = ref(8)
const simBins      = ref(2500)
const simFoldLen   = ref(400)
const simFoldBase  = ref(16)
let   _seed        = 42
let   samples      = []

const simDisplay = ref([])
const simMaxLen  = ref(1)
const simStats   = ref({ nBatch:0, avgBs:'0', padRate:'0', maxDiff:0 })
const simYaml    = ref('')

const showBsize = computed(() => ['torch_dl','unsorted','sorted'].includes(simType.value))
const showBins  = computed(() => ['length','numel'].includes(simType.value))
const showFold  = computed(() => simType.value === 'folded')

function generateSamples(n=50, seed=42) {
  const rng = makeRng(seed)
  samples = Array.from({length:n}, (_,i) => ({
    id:  `utt_${String(i+1).padStart(3,'0')}`,
    len: Math.round(50 + rng()*rng()*1200 + rng()*200),
    dim: 80,
  }))
}
function reseed() { _seed = Math.floor(Math.random()*9999); generateSamples(50,_seed); runSim() }

function runSim() {
  const type      = simType.value
  const bsize     = simBatchSize.value || 8
  const bins      = simBins.value      || 2500
  const foldLen   = simFoldLen.value   || 400
  const foldBase  = simFoldBase.value  || 16

  let ordered = [...samples]
  let batches

  if (type === 'torch_dl') {
    const rng = makeRng(_seed)
    ordered = [...samples].sort(() => rng() - 0.5)
    batches = []
    for (let i=0; i<ordered.length; i+=bsize) batches.push(ordered.slice(i,i+bsize))
  } else if (type === 'unsorted') {
    batches = []
    for (let i=0; i<ordered.length; i+=bsize) batches.push(ordered.slice(i,i+bsize))
  } else if (type === 'sorted') {
    ordered.sort((a,b)=>a.len-b.len)
    batches = []
    for (let i=0; i<ordered.length; i+=bsize) batches.push(ordered.slice(i,i+bsize))
  } else if (type === 'folded') {
    ordered.sort((a,b)=>a.len-b.len)
    batches = []; let i=0
    while (i<ordered.length) {
      const slice  = ordered.slice(i, i+foldBase)
      const maxLen = Math.max(...slice.map(s=>s.len))
      const factor = maxLen > foldLen ? Math.max(1, Math.round(foldLen/maxLen*foldBase)) : foldBase
      batches.push(ordered.slice(i, i+Math.max(1,factor)))
      i += Math.max(1,factor)
    }
  } else if (type === 'length') {
    ordered.sort((a,b)=>b.len-a.len)
    batches = []; let cur=[], curBins=0
    for (const s of ordered) {
      if (curBins+s.len > bins && cur.length>0) { batches.push(cur); cur=[]; curBins=0 }
      cur.push(s); curBins+=s.len
    }
    if (cur.length) batches.push(cur)
  } else {
    ordered.sort((a,b)=>b.len-a.len)
    batches = []; let cur=[], maxLen=0
    for (const s of ordered) {
      const newMax    = Math.max(maxLen, s.len)
      const projected = newMax * (cur.length+1) * s.dim
      if (projected > bins*s.dim && cur.length>0) { batches.push(cur); cur=[]; maxLen=0 }
      cur.push(s); maxLen=Math.max(maxLen,s.len)
    }
    if (cur.length) batches.push(cur)
  }

  const colorMap = {}
  batches.forEach((b,bi) => b.forEach(s => { colorMap[s.id] = COLORS[bi % COLORS.length] }))
  simMaxLen.value = Math.max(...samples.map(s=>s.len))

  const groupByBatch = type === 'torch_dl'
  const display = []
  if (groupByBatch) {
    batches.forEach((batch,bi) => {
      batch.forEach(s => display.push({ type:'sample', ...s, color:colorMap[s.id] }))
      display.push({ type:'divider', batchIdx:bi, count:batch.length })
    })
  } else {
    const batchOf = {}
    batches.forEach((b,bi) => b.forEach(s => { batchOf[s.id]=bi }))
    const dispOrder = ['sorted','folded','length','numel'].includes(type) ? ordered : samples
    let last=-1
    for (const s of dispOrder) {
      const bi = batchOf[s.id]
      if (bi!==last && last>=0) display.push({ type:'divider', batchIdx:last, count:batches[last].length })
      last=bi
      display.push({ type:'sample', ...s, color:colorMap[s.id] })
    }
    if (last>=0) display.push({ type:'divider', batchIdx:last, count:batches[last].length })
  }
  simDisplay.value = display

  // stats
  const nbatch   = batches.length
  let totalPad=0, totalData=0, maxDiff=0
  for (const b of batches) {
    const ml = Math.max(...b.map(s=>s.len))
    for (const s of b) totalPad += ml - s.len
    totalData += b.reduce((a,s)=>a+s.len,0)
    maxDiff = Math.max(maxDiff, ml - Math.min(...b.map(s=>s.len)))
  }
  simStats.value = {
    nBatch:  nbatch,
    avgBs:   (batches.reduce((a,b)=>a+b.length,0)/nbatch).toFixed(1),
    padRate: (totalPad/(totalData+totalPad)*100).toFixed(1),
    maxDiff,
  }

  // yaml
  const kw=(t)=>`<span class="kw">${t}</span>`
  const ky=(t)=>`<span class="key">${t}</span>`
  const vl=(t)=>`<span class="val">${t}</span>`
  const st=(t)=>`<span class="str">${t}</span>`
  const cm=(t)=>`<span class="cmt">${t}</span>`
  const SF = st('${stats_dir}/train/feats_shape')

  if (type==='torch_dl') simYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}: ${vl('null')}   ${cm('# ESPnet iterator disabled')}\n    ${ky('batch_size')}: ${vl(bsize)}\n    ${ky('shuffle')}: ${vl('true')}\n    ${ky('num_workers')}: ${vl(4)}\n  ${ky('valid')}:\n    ${ky('iter_factory')}: ${vl('null')}\n    ${ky('batch_size')}: ${vl(bsize)}\n    ${ky('shuffle')}: ${vl('false')}`
  else if (type==='unsorted') simYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('batches')}:\n        ${ky('type')}: ${vl('unsorted')}\n        ${ky('batch_size')}: ${vl(bsize)}\n        ${cm('# feats_shape not needed')}`
  else if (type==='sorted')   simYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('batches')}:\n        ${ky('type')}: ${vl('sorted')}\n        ${ky('batch_size')}: ${vl(bsize)}\n        ${ky('shape_files')}:\n          - ${SF}`
  else if (type==='folded')   simYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('batches')}:\n        ${ky('type')}: ${vl('folded')}\n        ${ky('batch_size')}: ${vl(foldBase)}\n        ${ky('min_batch_size')}: ${vl(1)}\n        ${ky('fold_lengths')}:\n          - ${vl(foldLen)}\n        ${ky('shape_files')}:\n          - ${SF}`
  else if (type==='length')   simYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('batches')}:\n        ${ky('type')}: ${vl('length')}\n        ${ky('batch_bins')}: ${vl(bins)}\n        ${ky('shape_files')}:\n          - ${SF}`
  else                        simYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('batches')}:\n        ${ky('type')}: ${vl('numel')}\n        ${ky('batch_bins')}: ${vl(bins)}\n        ${ky('shape_files')}:\n          - ${SF}`
}

// ══════════════════════════════════════════════════════════════════════════
// TAB 4 — Chunk visualiser
// ══════════════════════════════════════════════════════════════════════════
const chunkLen       = ref(300)
const chunkShiftPct  = ref(50)
const chunkBatchSize = ref(4)
const chunkShuffle   = ref(false)
const chunkStats     = ref({ totalChunks:0, totalBatches:0 })
const chunkYaml      = ref('')
const chunkVisRef    = ref(null)
const chunkBatchVisRef = ref(null)
let   _chunkSeed     = 1
let   chunkUtts      = [
  {id:'utt_001',len:950},{id:'utt_002',len:480},{id:'utt_003',len:1200},
  {id:'utt_004',len:310},{id:'utt_005',len:730},
]

function reseedChunks() {
  const s0 = Math.floor(Math.random()*99999)
  _chunkSeed = s0
  const rng  = makeRng(s0)
  chunkUtts  = Array.from({length:5},(_,i)=>({
    id:'utt_'+String(i+1).padStart(3,'0'),
    len: Math.round(200 + rng()*rng()*1400 + rng()*300),
  }))
  drawChunks()
}

function drawChunks() {
  const cLen    = chunkLen.value
  const shift   = Math.round(cLen * chunkShiftPct.value / 100)
  const bSize   = chunkBatchSize.value || 4
  const doShuf  = chunkShuffle.value
  const maxULen = Math.max(...chunkUtts.map(u=>u.len))
  const MAX_BAR = 300

  // STEP1: compute all chunks
  const allChunks   = []
  const uttChunkMap = {}
  const uttRemainder= {}
  let   globalIdx   = 0

  chunkUtts.forEach((utt, uttIdx) => {
    const color  = CHUNK_COLORS[uttIdx % CHUNK_COLORS.length]
    const chunks = []
    let pos=0; const startIdx=globalIdx
    while (pos+cLen <= utt.len) {
      const e = { uttId:utt.id, globalIdx, color }
      allChunks.push(e); chunks.push(e)
      pos+=shift; globalIdx++
    }
    uttChunkMap[utt.id]  = chunks
    const lastEnd        = globalIdx>startIdx ? (pos-shift)+cLen : 0
    uttRemainder[utt.id] = utt.len - lastEnd
  })

  // STEP2: left panel (DOM)
  const vis = chunkVisRef.value
  if (!vis) return
  vis.innerHTML = ''
  chunkUtts.forEach(utt => {
    const barW   = Math.round(utt.len/maxULen*MAX_BAR)
    const scale  = barW/utt.len
    const chunks = uttChunkMap[utt.id]
    const LANE_H = 18, LANE_GAP=4
    const nLanes = shift<cLen ? 2 : 1
    const barH   = nLanes*LANE_H+(nLanes-1)*LANE_GAP+2

    const row    = document.createElement('div')
    row.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:10px'
    const lbl    = document.createElement('div')
    lbl.style.cssText = 'font-family:var(--font-mono);font-size:10px;color:var(--text2);width:48px;flex-shrink:0'
    lbl.textContent   = utt.id
    row.appendChild(lbl)

    const bar = document.createElement('div')
    bar.style.cssText = `position:relative;width:${barW}px;min-width:${barW}px;height:${barH}px;background:var(--bg3);border:1px solid var(--border2);border-radius:4px;flex-shrink:0`

    chunks.forEach((c,ci) => {
      const left  = Math.round(ci*shift*scale)
      const width = Math.max(16, Math.round(cLen*scale))
      const lane  = nLanes===1 ? 0 : ci%nLanes
      const top   = lane*(LANE_H+LANE_GAP)
      const p     = document.createElement('div')
      p.style.cssText = `position:absolute;left:${left}px;top:${top}px;width:${width}px;height:${LANE_H}px;background:${c.color};opacity:0.88;border-radius:3px;display:flex;align-items:center;justify-content:center;font-family:var(--font-mono);font-size:9px;font-weight:600;color:#fff;overflow:hidden;box-sizing:border-box;white-space:nowrap`
      p.textContent = 'c'+(c.globalIdx+1)
      bar.appendChild(p)
    })

    const rem = uttRemainder[utt.id]
    if (rem>0 && rem<cLen) {
      const dropLeft = Math.round((utt.len-rem)*scale)
      const dropW    = Math.round(rem*scale)
      const r = document.createElement('div')
      r.style.cssText = `position:absolute;left:${dropLeft}px;top:0;width:${Math.max(dropW,14)}px;height:100%;border-left:2px dashed var(--border2);display:flex;align-items:center;justify-content:center;font-size:8px;color:var(--text3);white-space:nowrap;overflow:hidden;box-sizing:border-box`
      r.textContent = rem+'f dropped'
      bar.appendChild(r)
    }
    row.appendChild(bar)
    vis.appendChild(row)
  })

  // STEP3: right panel
  let pooled = [...allChunks]
  if (doShuf) {
    const rng = makeRng(_chunkSeed ^ 0xdeadbeef)
    pooled = pooled.sort(() => rng()-0.5)
  }
  const batches = []
  for (let i=0; i<pooled.length; i+=bSize) batches.push(pooled.slice(i,i+bSize))

  const bvis = chunkBatchVisRef.value
  if (!bvis) return
  bvis.innerHTML = ''
  const showN = Math.min(batches.length, 6)
  for (let bi=0; bi<showN; bi++) {
    const batch = batches[bi]
    const wrap  = document.createElement('div')
    wrap.style.cssText = 'margin-bottom:8px'
    const lbl   = document.createElement('div')
    lbl.style.cssText = 'font-family:var(--font-mono);font-size:9px;color:var(--text3);margin-bottom:3px'
    lbl.textContent   = `batch ${bi+1}  (${batch.length} chunks × ${cLen}f)${doShuf?' shuffled':''}`
    wrap.appendChild(lbl)
    const brow  = document.createElement('div')
    brow.style.cssText = 'display:flex;gap:3px;align-items:center;flex-wrap:wrap'
    for (const c of batch) {
      const chip = document.createElement('div')
      chip.style.cssText = `width:52px;height:26px;background:${c.color};opacity:0.82;border-radius:4px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:var(--font-mono);color:#fff;font-size:9px;font-weight:600;flex-shrink:0`
      chip.innerHTML = `c${c.globalIdx+1}<span style="font-weight:400;font-size:8px;opacity:0.85">${c.uttId.replace('utt_','u')}</span>`
      brow.appendChild(chip)
    }
    const note = document.createElement('span')
    note.style.cssText = 'font-size:10px;color:var(--text3);margin-left:6px'
    note.textContent   = `← all ${cLen}f, padding=0`
    brow.appendChild(note)
    wrap.appendChild(brow)
    bvis.appendChild(wrap)
  }
  if (batches.length>showN) {
    const more = document.createElement('div')
    more.style.cssText = 'font-size:11px;color:var(--text3);margin-top:4px'
    more.textContent   = `… ${batches.length-showN} more batches`
    bvis.appendChild(more)
  }

  chunkStats.value = { totalChunks: allChunks.length, totalBatches: batches.length }

  // YAML
  const kw=(t)=>`<span class="kw">${t}</span>`
  const ky=(t)=>`<span class="key">${t}</span>`
  const vl=(t)=>`<span class="val">${t}</span>`
  const cm=(t)=>`<span class="cmt">${t}</span>`
  const shLine = doShuf
    ? `      ${ky('num_cache_chunks')}: ${vl(allChunks.length)}  ${cm('# shuffle pool → chunks cross utterance boundaries')}`
    : `      ${cm('# num_cache_chunks not set → no shuffle')}`
  chunkYaml.value = `<pre>${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('_target_')}: ${vl('espnet2.iterators.chunk_iter_factory.ChunkIterFactory')}\n      ${ky('batch_size')}: ${vl(bSize)}\n      ${ky('chunk_length')}: ${vl(cLen)}\n      ${ky('chunk_shift_ratio')}: ${vl((chunkShiftPct.value/100).toFixed(1))}\n${shLine}\n      ${ky('batches')}:\n        - [${`<span class="str">utt_001</span>`}]\n        - [${`<span class="str">utt_002</span>`}]\n        ${cm('# ← generated by UnsortedBatchSampler')}</pre>`
}

// ══════════════════════════════════════════════════════════════════════════
// TAB 5 — Category visualiser
// ══════════════════════════════════════════════════════════════════════════
const catMode        = ref('none')
const catAlphaTimes10= ref(4)
const catBatchSize   = ref(10)
const catBatches     = ref([])
const catYaml        = ref('')

const CAT_DIST = { spk_A:20, spk_B:3, spk_C:5, spk_D:15, spk_E:7 }
const CAT_UTTS = (() => {
  const utts = []
  Object.entries(CAT_DIST).forEach(([spk,count]) => {
    for (let i=0;i<count;i++) utts.push({ id:`utt_${String(utts.length+1).padStart(3,'0')}`, spk, len:100+Math.round(Math.random()*800) })
  })
  return utts
})()
const catUttCounts = computed(() => {
  const c={}; Object.keys(SPK_COLORS).forEach(s=>{ c[s]=CAT_UTTS.filter(u=>u.spk===s).length }); return c
})

function catSegments(batch) {
  const spks=Object.keys(SPK_COLORS), cnt={}
  spks.forEach(s=>cnt[s]=0); batch.forEach(u=>cnt[u.spk]++)
  const segs=[]; let offset=0
  for (const s of spks) {
    if (!cnt[s]) continue
    const w=cnt[s]/batch.length*100
    segs.push({spk:s,left:offset,width:w}); offset+=w
  }
  return segs
}
function catBatchLabel(batch) {
  const spks=Object.keys(SPK_COLORS), cnt={}
  spks.forEach(s=>cnt[s]=0); batch.forEach(u=>cnt[u.spk]++)
  return spks.filter(s=>cnt[s]>0).map(s=>`${s.replace('spk_','')}:${cnt[s]}`).join(' ')
}

function drawCat() {
  const mode  = catMode.value
  const bs    = catBatchSize.value || 10
  const alpha = catAlphaTimes10.value / 10
  const spks  = Object.keys(SPK_COLORS)
  let batches = []

  if (mode==='none') {
    const shuffled = [...CAT_UTTS].sort(()=>Math.random()-0.5)
    for (let i=0;i<shuffled.length;i+=bs) batches.push(shuffled.slice(i,i+bs))
  } else if (mode==='catbel') {
    const queues={}; spks.forEach(s=>{ queues[s]=CAT_UTTS.filter(u=>u.spk===s).slice() })
    while (Object.values(queues).some(q=>q.length>0)) {
      const batch=[], avail=spks.filter(s=>queues[s].length>0)
      if (!avail.length) break
      const perSpk=Math.max(1,Math.floor(bs/avail.length))
      for (const s of avail) for (let j=0;j<perSpk&&queues[s].length>0&&batch.length<bs;j++) batch.push(queues[s].shift())
      for (const s of spks) while (batch.length<bs&&queues[s].length>0) batch.push(queues[s].shift())
      if (batch.length>0) batches.push(batch)
    }
  } else {
    const DS_MAP = { spk_A:'librispeech',spk_B:'commonvoice',spk_C:'commonvoice',spk_D:'librispeech',spk_E:'commonvoice' }
    const counts={}; spks.forEach(s=>{ counts[s]=CAT_UTTS.filter(u=>u.spk===s).length })
    let effCounts={...counts}
    if (mode==='catpow_bal_ds') {
      const dsCounts={}
      spks.forEach(s=>{ const ds=DS_MAP[s]; dsCounts[ds]=(dsCounts[ds]||0)+counts[s] })
      const maxDs=Math.max(...Object.values(dsCounts))
      spks.forEach(s=>{ const ds=DS_MAP[s]; effCounts[s]=counts[s]*Math.pow(maxDs/dsCounts[ds],alpha*0.5) })
    }
    const powered={},probs={}
    spks.forEach(s=>{ powered[s]=Math.pow(effCounts[s],alpha) })
    const total=Object.values(powered).reduce((a,b)=>a+b,0)
    spks.forEach(s=>{ probs[s]=powered[s]/total })
    const queues={}; spks.forEach(s=>{ queues[s]=[...CAT_UTTS.filter(u=>u.spk===s)] })
    for (let bi=0;bi<Math.ceil(50/bs);bi++) {
      const batch=[]
      for (let j=0;j<bs;j++) {
        let r=Math.random(),acc=0,chosen=spks[spks.length-1]
        for (const s of spks) { acc+=probs[s]; if(r<acc){chosen=s;break} }
        const q=queues[chosen]; if (q.length===0) continue
        batch.push(q[Math.floor(Math.random()*q.length)])
      }
      if (batch.length>0) batches.push(batch)
    }
  }
  catBatches.value = batches

  const kw=(t)=>`<span class="kw">${t}</span>`
  const ky=(t)=>`<span class="key">${t}</span>`
  const vl=(t)=>`<span class="val">${t}</span>`
  const st=(t)=>`<span class="str">${t}</span>`
  const cm=(t)=>`<span class="cmt">${t}</span>`
  const CAT = st('${stats_dir}/train/utt2category')
  const SHP = st('${stats_dir}/train/feats_shape')

  if (mode==='none') catYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('batches')}:\n        ${ky('type')}: ${vl('unsorted')}\n        ${ky('batch_size')}: ${vl(bs)}`
  else if (mode==='catbel') catYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('_target_')}: ${vl('espnet2.iterators.category_iter_factory.CategoryIterFactory')}\n      ${ky('batch_type')}: ${vl('catbel')}\n      ${ky('sampler_args')}:\n        ${ky('category2utt_file')}: ${CAT}\n        ${ky('batch_size')}: ${vl(bs)}\n        ${ky('min_batch_size')}: ${vl(1)}`
  else if (mode==='catpow') catYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('_target_')}: ${vl('espnet2.iterators.category_iter_factory.CategoryIterFactory')}\n      ${ky('batch_type')}: ${vl('catpow')}\n      ${ky('sampler_args')}:\n        ${ky('category2utt_file')}: ${CAT}\n        ${ky('shape_files')}:\n          - ${SHP}\n        ${ky('batch_bins')}: ${vl(1200000)}\n        ${ky('upsampling_factor')}: ${vl(alpha.toFixed(1))}\n        ${ky('min_batch_size')}: ${vl(1)}\n        ${ky('max_batch_size')}: ${vl(bs)}`
  else catYaml.value=`${kw('dataloader')}:\n  ${ky('train')}:\n    ${ky('iter_factory')}:\n      ${ky('_target_')}: ${vl('espnet2.iterators.category_iter_factory.CategoryIterFactory')}\n      ${ky('batch_type')}: ${vl('catpow_balance_dataset')}\n      ${ky('sampler_args')}:\n        ${ky('category2utt_file')}: ${CAT}\n        ${ky('dataset2utt_file')}:  ${st('${stats_dir}/train/dataset2utt')}\n        ${ky('utt2dataset_file')}:  ${st('${stats_dir}/train/utt2dataset')}\n        ${ky('shape_files')}:\n          - ${SHP}\n        ${ky('batch_bins')}: ${vl(1200000)}\n        ${ky('category_upsampling_factor')}: ${vl(alpha.toFixed(1))}\n        ${ky('dataset_upsampling_factor')}:  ${vl('1.0')}\n        ${ky('dataset_scaling_factor')}:     ${vl('1.2')}\n        ${ky('min_batch_size')}: ${vl(1)}\n        ${ky('max_batch_size')}: ${vl(bs)}`
}

// ── required files table ───────────────────────────────────────────────────
const fileRows = [
  { mode:'null (PyTorch DL)', badgeClass:'badge-coral',   iterator:'—',               shape:false, utt2cat:false, ds:false, desc:'—' },
  { mode:'unsorted',          badgeClass:'badge-blue',    iterator:'SequenceIter',    shape:false, utt2cat:false, ds:false, desc:'Fixed batch size, random order' },
  { mode:'sorted',            badgeClass:'badge-blue',    iterator:'SequenceIter',    shape:true,  utt2cat:false, ds:false, desc:'Reduced padding, fixed batch size' },
  { mode:'folded',            badgeClass:'badge-blue',    iterator:'SequenceIter',    shape:true,  utt2cat:false, ds:false, desc:'Shrinks batch size for longer sequences' },
  { mode:'length',            badgeClass:'badge-blue',    iterator:'SequenceIter',    shape:true,  utt2cat:false, ds:false, desc:'Bin-packing by total frames (1-D)' },
  { mode:'numel',             badgeClass:'badge-blue',    iterator:'SequenceIter',    shape:true,  utt2cat:false, ds:false, desc:'Bin-packing by total elements (frames×dim)' },
  { mode:'chunk',             badgeClass:'badge-green',   iterator:'ChunkIter',       shape:false, utt2cat:false, ds:false, desc:'Fixed-length chunks, long-sequence models' },
  { mode:'catbel',            badgeClass:'badge-purple',  iterator:'CategoryIter',    shape:false, utt2cat:true,  ds:false, desc:'Balanced category sampling' },
  { mode:'catpow',            badgeClass:'badge-purple',  iterator:'CategoryIter',    shape:true,  utt2cat:true,  ds:false, desc:'Power-law upsampling per category' },
  { mode:'catpow_balance_dataset', badgeClass:'badge-teal', iterator:'CategoryIter', shape:true,  utt2cat:true,  ds:true,  desc:'Category + cross-dataset balancing' },
  { mode:'chunk + catbel',    badgeClass:'badge-green',   iterator:'CategoryChunkIter', shape:false, utt2cat:true, ds:false, desc:'Long sequences + category balance' },
]

// ── init ───────────────────────────────────────────────────────────────────
onMounted(() => {
  generateSamples(50, 42)
  runSim()
  drawCat()
  drawChunks()
})
</script>

<style scoped>
.dl-demo {
  --bg: #ffffff; --bg2: transparent; --bg3: #f0f2f5;
  --border: #dde1ea; --border2: #c8cdd8;
  --text: #1a1e2e; --text2: #5a6180; --text3: #9098b0;
  --accent: #2563eb; --green: #059669;
  --font-mono: 'JetBrains Mono','Fira Code','Cascadia Code',monospace;
  font-family: -apple-system,'Inter',sans-serif;
  font-size: 14px; line-height: 1.6; color: var(--text);
  max-width: 100%; box-sizing: border-box; overflow-x: hidden;
}

/* tabs */
.tabs { display:flex; border-bottom:1px solid var(--border); overflow-x:auto; }
.tab  { padding:14px 20px 12px; font-size:13px; font-weight:500; color:var(--text2); cursor:pointer; border-bottom:2px solid transparent; white-space:nowrap; transition:color .15s,border-color .15s; user-select:none; }
.tab:hover { color:var(--text); }
.tab.active { color:var(--accent); border-bottom-color:var(--accent); }

/* content */
.tab-content { padding:24px 16px; max-width:100%; box-sizing:border-box; }
.section-title { font-size:11px; font-weight:600; letter-spacing:.1em; color:var(--text3); text-transform:uppercase; margin-bottom:12px; }

/* cards */
.card { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:20px 24px; margin-bottom:16px; }
.card-title { font-size:13px; font-weight:600; color:var(--text); margin-bottom:6px; display:flex; align-items:center; gap:8px; }
.card-desc { font-size:12px; color:var(--text2); line-height:1.7; }

/* code */
pre { background:var(--code-bg-color); border-radius:8px; padding:16px 18px; font-family:var(--font-family-code); font-size:12px; line-height:1.8; overflow-x:auto; color:var(--code-color, #e2e8f0); margin:10px 0; }
:deep(.kw)  { color:#6d4ff7; }
:deep(.str) { color:#059669; }
:deep(.key) { color:#2563eb; }
:deep(.cmt) { color:#9098b0; font-style:italic; }
:deep(.val) { color:#b45309; }

/* grids */
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }
@media(max-width:900px){.grid3{grid-template-columns:1fr 1fr}}
@media(max-width:640px){.grid2,.grid3{grid-template-columns:1fr}}

/* badges */
.badge { display:inline-block; font-size:10px; font-weight:600; font-family:var(--font-mono); padding:2px 7px; border-radius:4px; }
.badge-blue   { background:#dbeafe; color:#1d4ed8; }
.badge-purple { background:#ede9fe; color:#5b21b6; }
.badge-green  { background:#d1fae5; color:#065f46; }
.badge-coral  { background:#fee2e2; color:#991b1b; }
.badge-teal   { background:#cffafe; color:#155e75; }

/* flow */
.flow { display:flex; align-items:center; flex-wrap:wrap; row-gap:12px; }
.flow-node { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:10px 16px; font-size:12px; font-weight:500; }
.flow-node.highlight { border-color:var(--accent); color:var(--accent); }
.flow-node.out { border-color:var(--green); color:var(--green); }
.flow-arrow { color:var(--text3); padding:0 10px; font-size:16px; }
.node-sub { font-size:10px; color:var(--text3); display:block; }

/* controls */
.sim-controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:20px; padding:14px 18px; background:var(--bg2); border:1px solid var(--border); border-radius:10px; }
.ctrl-group   { display:flex; align-items:center; gap:8px; }
.ctrl-label   { font-size:11px; font-weight:600; color:var(--text2); white-space:nowrap; }
.mono-label   { font-family:var(--font-mono); font-size:11px; color:var(--text2); min-width:36px; }
select,input[type=range],input[type=number] { background:#fff; border:1px solid var(--border2); color:var(--text); border-radius:6px; padding:5px 10px; font-size:12px; outline:none; }
input[type=range] { padding:0; width:110px; accent-color:var(--accent); }
.reseed-btn { margin-left:auto; background:#fff; border:1px solid var(--border2); color:var(--text2); padding:5px 12px; border-radius:6px; cursor:pointer; font-size:11px; }

/* simulator */
.sim-canvas  { width:100%; overflow-x:auto; }
.sim-samples { display:flex; flex-direction:column; gap:3px; min-width:280px; }
.sample-row  { display:flex; align-items:center; gap:8px; height:22px; }
.sample-id   { font-family:var(--font-mono); font-size:10px; color:var(--text3); width:44px; text-align:right; flex-shrink:0; }
.sample-bar-wrap { flex:1; height:16px; position:relative; }
.sample-bar  { height:100%; border-radius:3px; position:absolute; left:0; }
.sample-len  { font-family:var(--font-mono); font-size:9px; color:var(--text3); width:36px; flex-shrink:0; }
.batch-divider { height:1px; background:var(--border2); margin:4px 0; display:flex; align-items:center; }
.batch-label { font-family:var(--font-mono); font-size:9px; color:var(--text3); padding-left:4px; white-space:nowrap; }
.sim-stats   { display:flex; gap:20px; flex-wrap:wrap; margin-top:16px; padding:14px 18px; background:var(--bg2); border:1px solid var(--border); border-radius:8px; }
.stat-item   { display:flex; flex-direction:column; gap:2px; }
.stat-label  { font-size:10px; color:var(--text3); }
.stat-value  { font-family:var(--font-mono); font-size:14px; font-weight:600; color:var(--text); }

/* category */
.legend      { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }
.legend-item { display:flex; align-items:center; gap:5px; font-size:11px; color:var(--text2); }
.legend-dot  { width:10px; height:10px; border-radius:2px; flex-shrink:0; }
.cat-canvas  { margin-top:12px; }
.cat-row     { display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.cat-id      { font-family:var(--font-mono); font-size:10px; color:var(--text3); width:44px; text-align:right; }
.cat-bar-bg  { flex:1; height:16px; background:var(--bg3); border-radius:3px; position:relative; overflow:hidden; min-width:200px; }
.cat-bar-fill{ height:100%; border-radius:3px; position:absolute; }
.cat-spk-label { font-size:10px; color:var(--text2); width:40px; }

/* files table */
.files-table { width:100%; border-collapse:collapse; font-size:12px; }
.files-table th { text-align:left; padding:10px 14px; color:var(--text3); font-weight:600; font-size:10px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid var(--border); background:var(--bg2); }
.files-table td { padding:10px 14px; border-bottom:1px solid var(--border); vertical-align:top; line-height:1.6; }
.files-table tr:last-child td { border-bottom:none; }
.files-table tr:hover td { background:var(--bg2); }
.filename { font-family:var(--font-mono); font-size:11px; color:var(--green); }
.check { color:var(--green); font-size:13px; }
.dash  { color:var(--text3); }

/* alerts */
.info-box { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:12px 16px; font-size:12px; color:#1e40af; line-height:1.7; margin:12px 0; }
.warn-box { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:12px 16px; font-size:12px; color:#92400e; line-height:1.7; margin:12px 0; }

hr { border:none; border-top:1px solid var(--border); margin:24px 0; }
</style>