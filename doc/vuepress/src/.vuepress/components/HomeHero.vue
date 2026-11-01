<template>
  <div class="home-hero">

    <!-- Logo + tagline -->
    <div class="hero-top">
      <img
        src="/assets/image/espnet3_logo.png"
        alt="ESPnet3"
        class="hero-logo"
      />
      <div class="hero-text">
        <div class="hero-tagline">Speech Foundation Model</div>
        <div class="hero-tagline">Research Platform</div>
        <div class="hero-sub">Train · Evaluate · Scale</div>
      </div>
    </div>

    <!-- CTA buttons -->
    <div class="hero-cta">
      <a class="btn btn-primary" href="./get-started/index.html">Get Started</a>
      <a class="btn btn-secondary" href="https://github.com/espnet/espnet" target="_blank">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" style="flex-shrink:0">
          <path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.868-.013-1.703-2.782.604-3.369-1.342-3.369-1.342-.454-1.155-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0 1 12 6.836a9.59 9.59 0 0 1 2.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.741 0 .267.18.578.688.48C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
        </svg>
        GitHub
      </a>
    </div>

    <!-- Install selector -->
    <div class="install-box">
      <div class="install-row">
        <div class="install-label">Package Manager</div>
        <div class="install-options">
          <button
            v-for="opt in options"
            :key="opt.key"
            class="iopt"
            :class="{ active: selected === opt.key }"
            @click="selected = opt.key"
          >{{ opt.label }}</button>
        </div>
      </div>

      <div class="install-cmd">
        <code class="install-code">
          <span class="cmd-prefix">$</span>
          <Transition name="cmd-fade" mode="out-in">
            <span :key="selected" class="cmd-text">{{ command }}</span>
          </Transition>
        </code>
        <button class="copy-btn" :class="{ copied }" @click="copy">
          <Transition name="cmd-fade" mode="out-in">
            <span :key="copied ? 'copied' : 'copy'">{{ copied ? '✓ Copied' : 'Copy' }}</span>
          </Transition>
        </button>
      </div>
    </div>

    <!-- Stats row -->
    <div class="hero-stats">
      <div class="stat">
        <span class="stat-num">9.4k</span>
        <span class="stat-label">GitHub stars</span>
      </div>
      <div class="stat-divider" />
      <div class="stat">
        <span class="stat-num">140+</span>
        <span class="stat-label">languages</span>
      </div>
      <div class="stat-divider" />
      <div class="stat">
        <span class="stat-num">Apache 2.0</span>
        <span class="stat-label">license</span>
      </div>
    </div>

  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const selected = ref('pip')
const copied = ref(false)

const options = [
  { key: 'pip',  label: 'pip'  },
  { key: 'uv',   label: 'uv'   },
  { key: 'pixi', label: 'pixi' },
  { key: 'git',  label: 'git'  },
]

const commands = {
  pip:  'pip install espnet',
  uv:   'uv pip install espnet',
  pixi: 'pixi add --pypi espnet',
  git:  'pip install git+https://github.com/espnet/espnet.git',
}

const command = computed(() => commands[selected.value])

function copy() {
  navigator.clipboard.writeText(command.value).then(() => {
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  })
}
</script>

<style scoped>
/* ── Root ──────────────────────────────────────────── */
.home-hero {
  background: #ffffff;
  border-bottom: 0.5px solid #e2e6ec;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 56px 24px 52px;
  text-align: center;
  gap: 0;
}

/* ── Logo + tagline ────────────────────────────────── */
.hero-top {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 28px;
  margin-bottom: 32px;
  flex-wrap: wrap;
}

.hero-logo {
  height: 72px;
  width: auto;
  object-fit: contain;
  flex-shrink: 0;
}

.hero-text {
  text-align: left;
}

.hero-tagline {
  font-size: clamp(22px, 3.5vw, 30px);
  font-weight: 700;
  color: #0d1520;
  line-height: 1.2;
  letter-spacing: -0.02em;
}

.hero-sub {
  margin-top: 8px;
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 12px;
  letter-spacing: 0.18em;
  color: #0f7a5a;
  text-transform: uppercase;
}

/* ── CTA ───────────────────────────────────────────── */
.hero-cta {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
  margin-bottom: 32px;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 13px;
  font-weight: 600;
  padding: 9px 22px;
  border-radius: 8px;
  cursor: pointer;
  border: none;
  transition: all 0.15s;
  text-decoration: none;
  line-height: 1;
}

.btn-primary {
  background: #0f7a5a;
  color: #fff;
}
.btn-primary:hover {
  background: #0d6b4f;
  transform: translateY(-1px);
}

.btn-secondary {
  background: #fff;
  color: #1a2030;
  border: 0.5px solid #d0d7e2;
}
.btn-secondary:hover {
  border-color: #0f7a5a;
  color: #0f7a5a;
}

/* ── Install box ───────────────────────────────────── */
.install-box {
  width: 100%;
  max-width: 540px;
  background: #ffffff;
  border: 0.5px solid #e2e6ec;
  border-radius: 12px;
  padding: 18px 22px;
  text-align: left;
  margin-bottom: 28px;
}

.install-row {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.install-label {
  font-size: 11px;
  font-weight: 600;
  color: #9aaabb;
  white-space: nowrap;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-family: 'SFMono-Regular', Consolas, monospace;
}

.install-options {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
}

.iopt {
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 12px;
  padding: 4px 13px;
  border-radius: 6px;
  border: 0.5px solid #d0d7e2;
  background: #fff;
  color: #3a4a5c;
  cursor: pointer;
  transition: all 0.15s;
}
.iopt:hover {
  border-color: #0f7a5a;
  color: #0f7a5a;
}
.iopt.active {
  background: #0f7a5a;
  border-color: #0f7a5a;
  color: #fff;
}

.install-cmd {
  display: flex;
  align-items: center;
  gap: 10px;
  background: #0d1520;
  border-radius: 8px;
  padding: 11px 16px;
}

.install-code {
  flex: 1;
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 13px;
  color: #e2e8f0;
  background: transparent;
  white-space: nowrap;
  overflow-x: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.cmd-prefix {
  color: #0f7a5a;
  user-select: none;
}

.cmd-text {
  display: inline-block;
}

.copy-btn {
  font-size: 11px;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: 5px;
  border: 0.5px solid #2a3a50;
  background: transparent;
  color: #7a8898;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
  flex-shrink: 0;
  min-width: 64px;
  text-align: center;
}
.copy-btn:hover {
  border-color: #0f7a5a;
  color: #0f7a5a;
}
.copy-btn.copied {
  border-color: #0f7a5a;
  color: #0f7a5a;
}

/* ── Stats ─────────────────────────────────────────── */
.hero-stats {
  display: flex;
  align-items: center;
  gap: 24px;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.stat-num {
  font-size: 17px;
  font-weight: 700;
  color: #0d1520;
  letter-spacing: -0.02em;
}

.stat-label {
  font-size: 11px;
  color: #9aaabb;
  font-family: 'SFMono-Regular', Consolas, monospace;
  letter-spacing: 0.04em;
}

.stat-divider {
  width: 0.5px;
  height: 28px;
  background: #e2e6ec;
}

/* ── Transitions ───────────────────────────────────── */
.cmd-fade-enter-active,
.cmd-fade-leave-active {
  transition: opacity 0.15s, transform 0.15s;
}
.cmd-fade-enter-from,
.cmd-fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}

/* ── Responsive ────────────────────────────────────── */
@media (max-width: 520px) {
  .hero-top {
    flex-direction: column;
    gap: 16px;
  }
  .hero-text {
    text-align: center;
  }
  .hero-logo {
    height: 56px;
  }
}
</style>
