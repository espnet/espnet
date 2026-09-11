<script setup>
import { ref, computed } from 'vue'

// ── Toolkit selector ──────────────────────────────────────────────────────────
const toolkit = ref('speechbrain')
const toolkits = ['speechbrain', 'espnet', 'rawnet']

// ── Provider / Runner tab ─────────────────────────────────────────────────────
const activeSection = ref('provider')

// ── Copy ──────────────────────────────────────────────────────────────────────
const copiedKey = ref(null)
function copy(key, text) {
  navigator.clipboard.writeText(text).then(() => {
    copiedKey.value = key
    setTimeout(() => { copiedKey.value = null }, 2000)
  })
}

const yamlCopied = ref(false)
function copyYaml() {
  navigator.clipboard.writeText(yamlRaw.value).then(() => {
    yamlCopied.value = true
    setTimeout(() => { yamlCopied.value = false }, 2000)
  })
}

// ── Toolkit metadata ──────────────────────────────────────────────────────────
const toolkitMeta = {
  speechbrain: {
    badge: 'default', badgeColor: '#0f7a5a', badgeBg: '#edf8f3',
    desc:  'ECAPA-TDNN encoder via SpeechBrain. No resampling needed — AudioNormalizer handles it.',
    dim:   '192-D',
    model: 'speechbrain/spkrec-ecapa-voxceleb',
  },
  espnet: {
    badge: 'espnet2', badgeColor: '#185fa5', badgeBg: '#e6f1fb',
    desc:  'Speech2Embedding from espnet2. Resamples to 16 kHz; accepts a HuggingFace tag or local .pth path.',
    dim:   'model-specific',
    model: 'espnet/voxceleb_xvector_wavlm',
  },
  rawnet: {
    badge: 'RawNet3', badgeColor: '#b06a00', badgeBg: '#fdf5e6',
    desc:  'RawNet3 with Bottle2neck blocks. Averages embeddings from 10 overlapping 3-second segments.',
    dim:   '256-D',
    model: '/path/to/rawnet3.pth',
  },
}
const meta = computed(() => toolkitMeta[toolkit.value])

// ── Syntax highlighting ───────────────────────────────────────────────────────
function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function hlYaml(raw) {
  return esc(raw).split('\n').map(line => {
    if (!line.trim()) return line
    if (/^\s*#/.test(line))
      return `<span class="hl-cm">${line}</span>`
    if (/^\s*- /.test(line))
      return line.replace(/^(\s*- )(.+)$/, (_, d, v) =>
        `${d}<span class="hl-str">${v}</span>`)
    return line.replace(/^(\s*)([\w_]+)(\s*:)(\s*)(.*)$/, (_, i, k, col, sp, v) => {
      const hk = `<span class="hl-key">${k}</span>`
      if (!v) return `${i}${hk}${col}`
      const hv = v
        .replace(/(\$\{[^}]+\})/g, m => `<span class="hl-ref">${m}</span>`)
        .replace(/^(\d+)(\s*)$/, (_, n, s2) => `<span class="hl-num">${n}</span>${s2}`)
        .replace(/(#.*)$/, m => `<span class="hl-cm"> ${m}</span>`)
      return `${i}${hk}${col}${sp}${hv}`
    })
  }).join('\n')
}

function hlPython(raw) {
  // Single-pass tokenizer — matches all token types on the original escaped text
  // so spans inserted for one token type are never re-processed by another rule.
  const TOKEN = /(f?"[^"]*"|f?'[^']*')|(#[^\n]*)|\b(def|class|return|from|import|if|elif|else|for|in|not|and|or|with|as|raise|try|except|None|True|False|self)\b|(@[\w.]+)|\b(\d+)\b/g

  return esc(raw).split('\n').map(line => {
    line = line.replace(TOKEN, (match, str, cm, kw, dec, num) => {
      if (str) return `<span class="hl-str">${match}</span>`
      if (cm)  return `<span class="hl-cm">${match}</span>`
      if (kw)  return `<span class="hl-kw">${match}</span>`
      if (dec) return `<span class="hl-dec">${match}</span>`
      if (num) return `<span class="hl-num">${match}</span>`
      return match
    })
    // Safe post-pass: only matches our own known tag patterns
    line = line.replace(/(<span class="hl-kw">def<\/span>) (\w+)/g,
      (_, kw, name) => `${kw} <span class="hl-fn">${name}</span>`)
    line = line.replace(/(<span class="hl-kw">class<\/span>) (\w+)/g,
      (_, kw, name) => `${kw} <span class="hl-fn">${name}</span>`)
    return line
  }).join('\n')
}

// ── Live YAML ─────────────────────────────────────────────────────────────────
const yamlRaw = computed(() => `xvector:
  toolkit: ${toolkit.value}
  pretrained_model: ${meta.value.model}
  device: cuda:0
  spk_embed_tag: spkrec-ecapa-voxceleb
  save_path: \${data_dir}/x_vectors
  batch_size: 20
  manifest_paths:
    train: \${data_dir}/manifest/train.tsv
    valid: \${data_dir}/manifest/valid.tsv
    test:  \${data_dir}/manifest/test.tsv
  splits:
    - train
    - valid
    - test`)

const yamlHl = computed(() => hlYaml(yamlRaw.value))

// ── Provider code ─────────────────────────────────────────────────────────────
const providerCode = computed(() => {
  if (toolkit.value === 'speechbrain') return `\
from espnet3.parallel.env_provider import EnvironmentProvider
from speechbrain.inference.classifiers import EncoderClassifier

class XVectorProvider(EnvironmentProvider):
    def build_env_local(self):
        xvec_cfg = self.config.xvector
        model = EncoderClassifier.from_hparams(
            source=xvec_cfg.get("pretrained_model",
                                "speechbrain/spkrec-ecapa-voxceleb"),
            run_opts={"device": xvec_cfg.get("device", "cpu")},
        )
        utterances, _ = self._load_manifest(self.params["manifest_path"])
        return {
            "model":      model,
            "toolkit":    "speechbrain",
            "device":     xvec_cfg.get("device", "cpu"),
            "utterances": utterances,
            "output_dir": Path(self.params["output_dir"]),
        }

    def build_worker_setup_fn(self):
        config, params = self.config, self.params
        def setup():
            xvec_cfg = config.xvector
            model = EncoderClassifier.from_hparams(
                source=xvec_cfg.get("pretrained_model",
                                    "speechbrain/spkrec-ecapa-voxceleb"),
                run_opts={"device": xvec_cfg.get("device", "cpu")},
            )
            utterances, _ = XVectorProvider._load_manifest(
                params["manifest_path"])
            return {
                "model":      model,
                "toolkit":    "speechbrain",
                "device":     xvec_cfg.get("device", "cpu"),
                "utterances": utterances,
                "output_dir": Path(params["output_dir"]),
            }
        return setup   # ← return the function, not the dict!`

  if (toolkit.value === 'espnet') return `\
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet2.bin.spk_inference import Speech2Embedding

class XVectorProvider(EnvironmentProvider):
    def build_env_local(self):
        xvec_cfg = self.config.xvector
        tag = xvec_cfg.get("pretrained_model",
                           "espnet/voxceleb_xvector_wavlm")
        model = Speech2Embedding.from_pretrained(
            model_tag=tag if not tag.endswith(".pth") else None,
            model_file=tag if tag.endswith(".pth") else None,
            batch_size=1, dtype="float32", train_config=None,
        )
        utterances, _ = self._load_manifest(self.params["manifest_path"])
        return {
            "model":      model,
            "toolkit":    "espnet",
            "device":     xvec_cfg.get("device", "cpu"),
            "utterances": utterances,
            "output_dir": Path(self.params["output_dir"]),
        }

    def build_worker_setup_fn(self):
        config, params = self.config, self.params
        def setup():
            xvec_cfg = config.xvector
            tag = xvec_cfg.get("pretrained_model",
                               "espnet/voxceleb_xvector_wavlm")
            model = Speech2Embedding.from_pretrained(
                model_tag=tag if not tag.endswith(".pth") else None,
                model_file=tag if tag.endswith(".pth") else None,
                batch_size=1, dtype="float32", train_config=None,
            )
            utterances, _ = XVectorProvider._load_manifest(
                params["manifest_path"])
            return {
                "model":      model,
                "toolkit":    "espnet",
                "device":     xvec_cfg.get("device", "cpu"),
                "utterances": utterances,
                "output_dir": Path(params["output_dir"]),
            }
        return setup   # ← return the function, not the dict!`

  return `\
from espnet3.parallel.env_provider import EnvironmentProvider
import torch
from RawNet3 import RawNet3
from RawNetBasicBlock import Bottle2neck

class XVectorProvider(EnvironmentProvider):
    def build_env_local(self):
        xvec_cfg = self.config.xvector
        model = self._load_rawnet(
            xvec_cfg["pretrained_model"],
            xvec_cfg.get("device", "cpu"),
        )
        utterances, _ = self._load_manifest(self.params["manifest_path"])
        return {
            "model":      model,
            "toolkit":    "rawnet",
            "device":     xvec_cfg.get("device", "cpu"),
            "utterances": utterances,
            "output_dir": Path(self.params["output_dir"]),
        }

    @staticmethod
    def _load_rawnet(checkpoint_path, device):
        model = RawNet3(Bottle2neck, model_scale=8, context=True,
                        summed=True, encoder_type="ECA", nOut=256,
                        out_bn=False, sinc_stride=10, log_sinc=True,
                        norm_sinc="mean", grad_mult=1)
        ckpt = torch.load(checkpoint_path, map_location=lambda s, l: s)
        model.load_state_dict(ckpt["model"])
        return model.to(device).eval()

    def build_worker_setup_fn(self):
        config, params = self.config, self.params
        def setup():
            xvec_cfg = config.xvector
            model = XVectorProvider._load_rawnet(
                xvec_cfg["pretrained_model"],
                xvec_cfg.get("device", "cpu"),
            )
            utterances, _ = XVectorProvider._load_manifest(
                params["manifest_path"])
            return {
                "model":      model,
                "toolkit":    "rawnet",
                "device":     xvec_cfg.get("device", "cpu"),
                "utterances": utterances,
                "output_dir": Path(params["output_dir"]),
            }
        return setup   # ← return the function, not the dict!`
})

// ── Runner code ───────────────────────────────────────────────────────────────
const runnerCode = computed(() => {
  if (toolkit.value === 'speechbrain') return `\
from espnet3.parallel.base_runner import BaseRunner
from speechbrain.dataio.preprocess import AudioNormalizer
import librosa, torch, numpy as np
from pathlib import Path

class XVectorRunner(BaseRunner):
    @staticmethod
    def forward(idx, model, toolkit, device,
                utterances, output_dir):
        if isinstance(idx, int):
            return XVectorRunner._process_one(
                idx, model, toolkit, device, utterances, output_dir)
        return [XVectorRunner._process_one(
                    i, model, toolkit, device, utterances, output_dir)
                for i in idx]

    @staticmethod
    def _process_one(idx, model, toolkit, device, utterances, output_dir):
        utt_id, wav_path = utterances[idx]
        out_path = Path(output_dir) / f"{utt_id}.pt"
        if out_path.exists():                    # resume-safe skip
            return {"utt_id": utt_id, "status": "skipped"}

        wav, in_sr = librosa.load(str(wav_path), sr=None)

        # SpeechBrain: AudioNormalizer resamples automatically
        audio_norm = AudioNormalizer()
        wav_t = audio_norm(torch.from_numpy(wav), in_sr).to(device)
        with torch.no_grad():
            emb = model.encode_batch(wav_t)
        tensor = emb.detach().cpu().float()[0]   # shape: (192,)

        torch.save(tensor, str(out_path))
        return {"utt_id": utt_id, "status": "ok"}`

  if (toolkit.value === 'espnet') return `\
from espnet3.parallel.base_runner import BaseRunner
import librosa, torch, numpy as np
from pathlib import Path

class XVectorRunner(BaseRunner):
    @staticmethod
    def forward(idx, model, toolkit, device,
                utterances, output_dir):
        if isinstance(idx, int):
            return XVectorRunner._process_one(
                idx, model, toolkit, device, utterances, output_dir)
        return [XVectorRunner._process_one(
                    i, model, toolkit, device, utterances, output_dir)
                for i in idx]

    @staticmethod
    def _process_one(idx, model, toolkit, device, utterances, output_dir):
        utt_id, wav_path = utterances[idx]
        out_path = Path(output_dir) / f"{utt_id}.pt"
        if out_path.exists():                    # resume-safe skip
            return {"utt_id": utt_id, "status": "skipped"}

        wav, in_sr = librosa.load(str(wav_path), sr=None)

        # espnet2: resample to 16 kHz, convert to mono
        tgt_sr = 16000
        if in_sr != tgt_sr:
            wav = librosa.resample(wav, orig_sr=in_sr, target_sr=tgt_sr)
        if wav.ndim == 2:
            wav = wav.mean(axis=0)
        wav_t = torch.from_numpy(wav.astype(np.float32)).to(device)
        with torch.no_grad():
            emb = model(wav_t)
        tensor = emb.cpu().float()               # shape: model-specific

        torch.save(tensor, str(out_path))
        return {"utt_id": utt_id, "status": "ok"}`

  return `\
from espnet3.parallel.base_runner import BaseRunner
import librosa, torch, numpy as np
from pathlib import Path

class XVectorRunner(BaseRunner):
    @staticmethod
    def forward(idx, model, toolkit, device,
                utterances, output_dir):
        if isinstance(idx, int):
            return XVectorRunner._process_one(
                idx, model, toolkit, device, utterances, output_dir)
        return [XVectorRunner._process_one(
                    i, model, toolkit, device, utterances, output_dir)
                for i in idx]

    @staticmethod
    def _process_one(idx, model, toolkit, device, utterances, output_dir):
        utt_id, wav_path = utterances[idx]
        out_path = Path(output_dir) / f"{utt_id}.pt"
        if out_path.exists():                    # resume-safe skip
            return {"utt_id": utt_id, "status": "skipped"}

        wav, in_sr = librosa.load(str(wav_path), sr=None)

        # RawNet3: 16 kHz, 10 overlapping 3-second segments
        tgt_sr, n_samples, n_seg = 16000, 48000, 10
        if in_sr != tgt_sr:
            wav = librosa.resample(wav, orig_sr=in_sr, target_sr=tgt_sr)
        if len(wav) < n_samples:
            wav = np.pad(wav, (0, n_samples - len(wav) + 1), "wrap")
        starts = np.linspace(0, len(wav) - n_samples, num=n_seg)
        segs = np.stack([wav[int(s):int(s)+n_samples] for s in starts])
        segs_t = torch.from_numpy(segs.astype(np.float32)).to(device)
        with torch.no_grad():
            emb = model(segs_t).mean(0)          # average over segments
        tensor = emb.detach().cpu().float()      # shape: (256,)

        torch.save(tensor, str(out_path))
        return {"utt_id": utt_id, "status": "ok"}`
})

const providerCodeHl = computed(() => hlPython(providerCode.value))
const runnerCodeHl   = computed(() => hlPython(runnerCode.value))

const usageCode = `from omegaconf import OmegaConf
from src.xvector_provider import XVectorProvider
from src.xvector_runner import XVectorRunner

config = OmegaConf.load("conf/training.yaml")
params = {
    "manifest_path": "data/train/manifest.tsv",
    "output_dir":    "data/x_vectors/train",
}

provider = XVectorProvider(config, params=params)
runner   = XVectorRunner(provider)

# Local: uses build_env_local()
runner(range(n_utterances))

# Distributed: uses build_worker_setup_fn() on each Dask worker
# with set_parallel(env="slurm", n_workers=8):
#     runner(range(n_utterances))`

const usageCodeHl = computed(() => hlPython(usageCode))

// ── Field reference ───────────────────────────────────────────────────────────
const fields = [
  { key: 'toolkit',         type: 'speechbrain | espnet | rawnet',
    desc:   'Which speaker embedding library to use.',
    effect: 'Controls which model class is loaded inside the Provider.' },
  { key: 'pretrained_model', type: 'str',
    desc:   'HuggingFace model ID (speechbrain / espnet) or local .pth path (rawnet).',
    effect: 'Passed directly to the model constructor inside build_env_local() / setup().' },
  { key: 'device',          type: 'str',
    desc:   'PyTorch device string. The model is moved here inside setup().',
    effect: 'Use cuda:0 for GPU, cpu for CPU-only environments.' },
  { key: 'spk_embed_tag',   type: 'str',
    desc:   'Short identifier for this embedding run, used as a subdirectory name.',
    effect: 'Output files land at save_path/{spk_embed_tag}/{split}/{utt_id}.pt — lets you store embeddings from multiple models side by side.' },
  { key: 'save_path',       type: 'str (Hydra resolver OK)',
    desc:   'Root output directory. Hydra resolvers like ${data_dir} are expanded before the stage runs.',
    effect: 'Passed to XVectorRunner as output_dir. Full per-split path: save_path/spk_embed_tag/{split}/.' },
  { key: 'batch_size',      type: 'int',
    desc:   'Number of utterances per forward() call.',
    effect: 'Sets BaseRunner(batch_size=...). When > 1, forward() receives list[int] instead of int.' },
  { key: 'manifest_paths',  type: 'dict[str, str]',
    desc:   'Maps each split name to its TSV manifest (utt_id \\t wav_path \\t text \\t speaker_id).',
    effect: 'The Provider reads the manifest for each split to build the utterances list.' },
  { key: 'splits',          type: 'list[str]',
    desc:   'Which splits to extract embeddings for. Must be a subset of manifest_paths keys.',
    effect: 'TTSSystem.compute_xvectors() iterates over this list, creating one XVectorRunner per split.' },
]

const showFields = ref(false)
</script>

<template>
  <section class="guide-section xvcfg">
    <div class="container">

      <!-- ── Toolkit chips ──────────────────────────────────────────────── -->
      <div class="xvcfg-shelf">
        <div class="xvcfg-shelf-label">TOOLKIT</div>
        <div class="xvcfg-chips">
          <button
            v-for="tk in toolkits" :key="tk"
            class="xvcfg-chip"
            :class="{ active: toolkit === tk }"
            @click="toolkit = tk"
          >{{ tk }}</button>
        </div>
      </div>

      <!-- ── Toolkit info card ──────────────────────────────────────────── -->
      <div class="xvcfg-tk-card">
        <div class="xvcfg-tk-head">
          <span class="xvcfg-tk-name">{{ toolkit }}</span>
          <span class="xvcfg-tk-badge" :style="{ background: meta.badgeBg, color: meta.badgeColor }">
            {{ meta.badge }}
          </span>
          <span class="xvcfg-tk-dim">embedding dim: {{ meta.dim }}</span>
        </div>
        <p class="xvcfg-tk-desc">{{ meta.desc }}</p>
      </div>

      <!-- ── YAML config ────────────────────────────────────────────────── -->
      <div class="xvcfg-yaml-box">
        <div class="xvcfg-yaml-header">
          <span>training_config.yaml</span>
          <span style="color:var(--worker);font-size:11px">● live</span>
          <button class="xvcfg-copy-btn" :class="{ copied: yamlCopied }" @click="copyYaml">
            <Transition name="fade" mode="out-in">
              <span :key="yamlCopied ? 'c' : 'u'">{{ yamlCopied ? '✓ Copied' : 'Copy' }}</span>
            </Transition>
          </button>
        </div>
        <pre class="xvcfg-pre"><code v-html="yamlHl"></code></pre>
      </div>

      <!-- ── Field reference (collapsible) ────────────────────────────── -->
      <button class="xvcfg-ref-toggle" @click="showFields = !showFields">
        Field reference
        <span class="xvcfg-ref-toggle-arrow">{{ showFields ? '▲' : '▼' }}</span>
      </button>
      <Transition name="fields-slide">
        <div v-show="showFields" class="xvcfg-ref-table">
          <div
            v-for="f in fields" :key="f.key"
            class="xvcfg-ref-row"
          >
            <div class="xvcfg-ref-left">
              <code class="hl-key xvcfg-ref-key">{{ f.key }}</code>
              <span class="xvcfg-ref-type">{{ f.type }}</span>
            </div>
            <div class="xvcfg-ref-right">
              <div class="xvcfg-ref-desc">{{ f.desc }}</div>
              <div class="xvcfg-ref-effect">
                <span class="xvcfg-effect-label">effect</span>{{ f.effect }}
              </div>
            </div>
          </div>
        </div>
      </Transition>

      <!-- ══════════════════════════════════════════════════════════════════
           IMPLEMENTATION  — Provider / Runner code
      ══════════════════════════════════════════════════════════════════ -->
      <div class="xvcfg-impl-divider">Implementation</div>

      <!-- Provider / Runner tabs -->
      <div class="section-tabs">
        <button class="section-tab" :class="{ active: activeSection === 'provider' }" @click="activeSection = 'provider'">
          XVectorProvider
        </button>
        <button class="section-tab" :class="{ active: activeSection === 'runner' }" @click="activeSection = 'runner'">
          XVectorRunner
        </button>
        <button class="section-tab" :class="{ active: activeSection === 'usage' }" @click="activeSection = 'usage'">
          Usage
        </button>
      </div>

      <!-- Provider panel -->
      <template v-if="activeSection === 'provider'">
        <div class="panel-desc">
          <p>
            <code>XVectorProvider</code> extends <code>EnvironmentProvider</code> and builds
            the runtime environment — model, manifest, and output directory — once per process.
          </p>
          <div class="two-col">
            <div class="method-card">
              <div class="method-badge local">local</div>
              <strong>build_env_local()</strong>
              <p>Called once on the driver. Builds and returns the env dict immediately.</p>
            </div>
            <div class="method-card">
              <div class="method-badge worker">distributed</div>
              <strong>build_worker_setup_fn()</strong>
              <p>Returns a zero-arg closure. Each Dask worker calls it once at startup.</p>
            </div>
          </div>
        </div>
        <div class="code-box">
          <div class="code-box-header">
            <span class="code-box-label">src/xvector_provider.py — {{ toolkit }}</span>
            <button class="copy-btn" :class="{ copied: copiedKey === 'provider' }" @click="copy('provider', providerCode)">
              <Transition name="fade" mode="out-in">
                <span :key="copiedKey === 'provider' ? 'c' : 'u'">{{ copiedKey === 'provider' ? '✓ Copied' : 'Copy' }}</span>
              </Transition>
            </button>
          </div>
          <pre class="code-pre"><code v-html="providerCodeHl"></code></pre>
        </div>
        <div class="callout callout-warn">
          <span class="callout-icon">⚠</span>
          <div>
            <strong>build_worker_setup_fn() must return a function, not a dict.</strong>
            The closure is pickled and shipped to each Dask worker. The model is built
            <em>inside</em> the closure so GPU memory is allocated on the right device.
          </div>
        </div>
      </template>

      <!-- Runner panel -->
      <template v-if="activeSection === 'runner'">
        <div class="panel-desc">
          <p>
            <code>XVectorRunner</code> extends <code>BaseRunner</code>. The only required
            method is <code>forward()</code> — a <code>@staticmethod</code> so it is
            pickle-safe for Dask.
          </p>
          <div class="flow-row">
            <div class="flow-step"><span class="flow-num">1</span><span>Resolve <code>utt_id, wav_path</code> from <code>utterances[idx]</code></span></div>
            <div class="flow-arrow">→</div>
            <div class="flow-step"><span class="flow-num">2</span><span>Skip if <code>output_dir/{utt_id}.pt</code> exists</span></div>
            <div class="flow-arrow">→</div>
            <div class="flow-step"><span class="flow-num">3</span><span>Load audio → extract embedding → save as <code>.pt</code></span></div>
          </div>
        </div>
        <div class="code-box">
          <div class="code-box-header">
            <span class="code-box-label">src/xvector_runner.py — {{ toolkit }}</span>
            <button class="copy-btn" :class="{ copied: copiedKey === 'runner' }" @click="copy('runner', runnerCode)">
              <Transition name="fade" mode="out-in">
                <span :key="copiedKey === 'runner' ? 'c' : 'u'">{{ copiedKey === 'runner' ? '✓ Copied' : 'Copy' }}</span>
              </Transition>
            </button>
          </div>
          <pre class="code-pre"><code v-html="runnerCodeHl"></code></pre>
        </div>
        <div class="callout callout-info">
          <span class="callout-icon">ℹ</span>
          <div>
            <strong>Resume safety:</strong> if <code>output_dir/{utt_id}.pt</code> already
            exists it is skipped — no audio is loaded. Restart after any interruption and
            only missing files are regenerated.
          </div>
        </div>
      </template>

      <!-- Usage panel -->
      <template v-if="activeSection === 'usage'">
        <div class="code-box">
          <div class="code-box-header">
            <span class="code-box-label">run.py</span>
            <button class="copy-btn" :class="{ copied: copiedKey === 'usage' }" @click="copy('usage', usageCode)">
              <Transition name="fade" mode="out-in">
                <span :key="copiedKey === 'usage' ? 'c' : 'u'">{{ copiedKey === 'usage' ? '✓ Copied' : 'Copy' }}</span>
              </Transition>
            </button>
          </div>
          <pre class="code-pre"><code v-html="usageCodeHl"></code></pre>
        </div>
      </template>

    </div>
  </section>
</template>

<style scoped>
/* ── Toolkit shelf ───────────────────────────────────────────────────────── */
.xvcfg-shelf {
  border: 1px solid var(--border2);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 10px;
  background: var(--bg2);
}
.xvcfg-shelf-label {
  font-family: var(--mono); font-size: 10px; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text3); margin-bottom: 8px;
}
.xvcfg-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.xvcfg-chip {
  font-family: var(--mono); font-size: 12px; padding: 4px 14px;
  border-radius: 6px; background: var(--bg3); color: var(--text2);
  border: 1px solid var(--border2); cursor: pointer; transition: all 0.15s;
}
.xvcfg-chip:hover { border-color: var(--worker-border); color: var(--worker); }
.xvcfg-chip.active {
  background: var(--worker-bg); color: var(--worker);
  border-color: var(--worker-border); font-weight: 600;
}

/* ── Toolkit info card ───────────────────────────────────────────────────── */
.xvcfg-tk-card {
  border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; margin-bottom: 12px; background: var(--bg2);
}
.xvcfg-tk-head {
  display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
}
.xvcfg-tk-name {
  font-family: var(--mono); font-size: 13px; font-weight: 700; color: var(--text);
}
.xvcfg-tk-badge {
  font-family: var(--mono); font-size: 10px; padding: 2px 7px; border-radius: 4px;
}
.xvcfg-tk-dim {
  margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--text3);
}
.xvcfg-tk-desc { font-size: 13px; color: var(--text2); line-height: 1.6; margin: 0; }

/* ── YAML box ────────────────────────────────────────────────────────────── */
.xvcfg-yaml-box {
  border: 1px solid var(--border); border-radius: 8px;
  overflow: hidden; margin-bottom: 1.25rem;
}
.xvcfg-yaml-header {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 13px; border-bottom: 1px solid var(--border);
  font-family: var(--mono); font-size: 11px; color: var(--text3); background: var(--bg2);
}
.xvcfg-yaml-header span:first-child { flex: 1; }
.xvcfg-copy-btn {
  font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px;
  border: 1px solid var(--border2); background: transparent; color: var(--text3);
  cursor: pointer; transition: all 0.15s; min-width: 52px; text-align: center;
}
.xvcfg-copy-btn:hover,
.xvcfg-copy-btn.copied { border-color: var(--worker-border); color: var(--worker); }
.xvcfg-pre {
  margin: 0; padding: 0.85rem 1rem; background: var(--bg3);
  font-family: var(--mono); font-size: 12.5px; line-height: 1.7;
  overflow-x: auto; border: none; border-radius: 0;
}
.xvcfg-pre code {
  display: block; white-space: pre; background: transparent;
  color: var(--text2); border: none; padding: 0;
}

/* ── YAML highlight tokens ───────────────────────────────────────────────── */
.xvcfg-pre :deep(.hl-key) { color: var(--driver); }
.xvcfg-pre :deep(.hl-str) { color: var(--worker); }
.xvcfg-pre :deep(.hl-num) { color: var(--provider); }
.xvcfg-pre :deep(.hl-ref) { color: var(--forward); }
.xvcfg-pre :deep(.hl-cm)  { color: var(--text3); font-style: italic; }

/* ── Field reference toggle ──────────────────────────────────────────────── */
.xvcfg-ref-toggle {
  display: flex; align-items: center; gap: 6px; width: 100%;
  font-family: var(--mono); font-size: 11px; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--text3);
  background: var(--bg2); border: 1px solid var(--border); border-radius: 6px;
  padding: 7px 12px; cursor: pointer; margin-bottom: 4px;
  transition: border-color 0.15s, color 0.15s;
}
.xvcfg-ref-toggle:hover { border-color: var(--driver-border); color: var(--driver); }
.xvcfg-ref-toggle-arrow { margin-left: auto; font-size: 10px; }

/* ── Field reference table ───────────────────────────────────────────────── */
.xvcfg-ref-table { display: flex; flex-direction: column; gap: 3px; margin-bottom: 2rem; }
.xvcfg-ref-row {
  display: grid; grid-template-columns: 200px 1fr; gap: 6px 14px;
  padding: 9px 12px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--bg2); cursor: default; transition: border-color 0.15s, background 0.15s;
}
.xvcfg-ref-left { display: flex; flex-direction: column; gap: 3px; }
.xvcfg-ref-key {
  font-family: var(--mono); font-size: 12.5px; font-weight: 700;
  background: transparent; padding: 0; border: none;
}
.xvcfg-ref-type { font-family: var(--mono); font-size: 10px; color: var(--text3); line-height: 1.4; }
.xvcfg-ref-right { display: flex; flex-direction: column; gap: 4px; }
.xvcfg-ref-desc { font-size: 13px; color: var(--text2); line-height: 1.55; }
.xvcfg-ref-effect { font-size: 12px; color: var(--text3); line-height: 1.5; }
.xvcfg-effect-label {
  font-family: var(--mono); font-size: 9px; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--provider); background: var(--provider-bg);
  border: 1px solid var(--provider-border); border-radius: 3px;
  padding: 1px 5px; margin-right: 5px;
}

/* ── Implementation divider ──────────────────────────────────────────────── */
.xvcfg-impl-divider {
  font-family: var(--mono); font-size: 10px; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--text3);
  border-top: 1px solid var(--border); padding-top: 1.25rem; margin-bottom: 1rem;
}

/* ── Section tabs ────────────────────────────────────────────────────────── */
.section-tabs {
  display: flex; gap: 4px; margin-bottom: 16px;
  border-bottom: 1px solid var(--border2);
}
.section-tab {
  font-family: var(--mono); font-size: 12.5px; padding: 7px 18px;
  border: none; border-bottom: 2px solid transparent;
  background: transparent; color: var(--text2); cursor: pointer;
  transition: all 0.15s; margin-bottom: -1px;
}
.section-tab:hover { color: var(--text); }
.section-tab.active { color: var(--worker); border-bottom-color: var(--worker); font-weight: 600; }

/* ── Panel desc ──────────────────────────────────────────────────────────── */
.panel-desc { margin-bottom: 16px; }
.panel-desc p { font-size: 14px; color: var(--text2); line-height: 1.65; margin-bottom: 12px; }

/* ── Two-col method cards ────────────────────────────────────────────────── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.method-card {
  border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; background: var(--bg2);
}
.method-card strong {
  font-family: var(--mono); font-size: 13px; display: block;
  margin-bottom: 4px; color: var(--text);
}
.method-card p { font-size: 12px; color: var(--text2); margin: 0; line-height: 1.55; }
.method-badge {
  font-family: var(--mono); font-size: 9px; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 4px; display: inline-block; margin-bottom: 6px;
}
.method-badge.local  { background: var(--provider-bg); color: var(--provider); }
.method-badge.worker { background: var(--worker-bg);   color: var(--worker);   }

/* ── Flow row ────────────────────────────────────────────────────────────── */
.flow-row {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px; margin-bottom: 4px;
}
.flow-step { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text2); }
.flow-num {
  width: 20px; height: 20px; border-radius: 50%;
  background: var(--worker-bg); color: var(--worker);
  font-size: 11px; font-weight: 700;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.flow-arrow { color: var(--text3); font-size: 16px; }

/* ── Code box (dark) ─────────────────────────────────────────────────────── */
.code-box { background: #0d1520; border-radius: 8px; overflow: hidden; margin-bottom: 14px; }
.code-box-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 14px; border-bottom: 1px solid #1e2d40;
}
.code-box-label { font-family: var(--mono); font-size: 11px; color: #6b7a8d; }
.copy-btn {
  font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 5px;
  border: 1px solid #2a3a50; background: transparent; color: #7a8898;
  cursor: pointer; transition: all 0.15s; min-width: 60px; text-align: center;
}
.copy-btn:hover, .copy-btn.copied { border-color: #0f7a5a; color: #0f7a5a; }
.code-pre {
  margin: 0; padding: 16px; overflow-x: auto;
  font-family: var(--mono); font-size: 12.5px; line-height: 1.65;
  color: #c9d8e8; background: transparent;
}
.code-pre code { display: block; white-space: pre; background: transparent; color: inherit; border: none; padding: 0; }

/* ── Python highlight tokens ─────────────────────────────────────────────── */
.code-pre :deep(.hl-kw)  { color: #c792ea; }
.code-pre :deep(.hl-fn)  { color: #82aaff; }
.code-pre :deep(.hl-dec) { color: #ffcb6b; }
.code-pre :deep(.hl-str) { color: #c3e88d; }
.code-pre :deep(.hl-num) { color: #f78c6c; }
.code-pre :deep(.hl-cm)  { color: #546e7a; font-style: italic; }
/* YAML tokens (shared) */
.code-pre :deep(.hl-key) { color: #82aaff; }
.code-pre :deep(.hl-ref) { color: #c792ea; }

/* ── Callouts ────────────────────────────────────────────────────────────── */
.callout {
  display: flex; gap: 10px; border-radius: 8px;
  padding: 12px 14px; font-size: 13px; line-height: 1.6; margin-bottom: 16px;
}
.callout-warn { background: var(--provider-bg); border: 1px solid var(--provider-border); color: var(--text2); }
.callout-info { background: var(--driver-bg);   border: 1px solid var(--driver-border);   color: var(--text2); }
.callout-icon { font-size: 16px; flex-shrink: 0; line-height: 1.4; color: var(--text3); }

/* ── Transitions ─────────────────────────────────────────────────────────── */
.fade-enter-active, .fade-leave-active { transition: opacity 0.15s; }
.fade-enter-from,   .fade-leave-to     { opacity: 0; }

.fields-slide-enter-active { transition: opacity 0.2s ease; }
.fields-slide-leave-active { transition: opacity 0.15s ease; }
.fields-slide-enter-from, .fields-slide-leave-to { opacity: 0; }

/* ── Responsive ──────────────────────────────────────────────────────────── */
@media (max-width: 560px) {
  .xvcfg-ref-row, .two-col { grid-template-columns: 1fr; }
  .flow-row { flex-direction: column; align-items: flex-start; }
  .flow-arrow { transform: rotate(90deg); }
}
</style>
