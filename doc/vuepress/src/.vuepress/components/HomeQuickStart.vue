<template>
  <section class="home-quickstart">
    <div class="home-section-inner">

      <div class="qs-layout">

        <!-- Left: system list -->
        <div class="qs-list">
          <div class="qs-list-head">Systems</div>
          <div
            v-for="s in systems"
            :key="s.key"
            class="qs-list-item"
            :class="{ active: activeKey === s.key }"
            @click="activeKey = s.key"
          >
            {{ s.key }}
            <span class="qs-list-dot" />
          </div>
        </div>

        <!-- Right: detail + install -->
        <div class="qs-panel">

          <!-- System detail -->
          <div class="qs-detail">
            <div class="qs-detail-name">{{ active.name }}</div>
            <div class="qs-detail-group">{{ active.group }}</div>
            <p class="qs-detail-desc">{{ active.desc }}</p>
            <div class="qs-pkg-label">Packages</div>
            <div class="qs-badges">
              <span
                v-for="pkg in active.pkgs"
                :key="pkg.name"
                class="qs-badge"
                :class="pkg.src === 'github' ? 'qs-badge-github' : 'qs-badge-pypi'"
                :title="pkg.note ? pkg.note : pkg.src"
              >
                {{ pkg.name }}
                <span v-if="pkg.note" class="qs-badge-note">{{ pkg.note }}</span>
              </span>
            </div>
          </div>

          <!-- Install box -->
          <div class="qs-install">
            <div class="qs-pkg-opts">
              <button
                v-for="p in pkgManagers"
                :key="p"
                class="qs-pkg-opt"
                :class="{ active: activePkg === p }"
                @click="activePkg = p"
              >{{ p }}</button>
            </div>
            <div class="qs-cmd">
              <div class="qs-cmd-top">
                <span class="qs-cmd-badge">{{ active.group }}</span>
                <button class="qs-copy-btn" :class="{ copied }" @click="copy">
                  <Transition name="cmd-fade" mode="out-in">
                    <span :key="copied ? 'c' : 'n'">{{ copied ? '✓ Copied' : 'Copy' }}</span>
                  </Transition>
                </button>
              </div>
              <div class="qs-cmd-line">
                <span class="qs-cmd-prefix">$</span>
                <Transition name="cmd-fade" mode="out-in">
                  <span :key="installCmd" class="qs-cmd-text">{{ installCmd }}</span>
                </Transition>
              </div>
            </div>
            <p class="qs-install-note">
              conda / source install →
              <a href="./get-started/installation.html">Installation docs</a>
            </p>
          </div>

        </div>
      </div>

    </div>
  </section>
</template>

<script setup>
import { ref, computed } from 'vue'
import systems from '../data/systems.json'

const pkgManagers = ['pip', 'uv']

const activeKey = ref(systems[0].key)
const activePkg = ref('pip')
const copied    = ref(false)

const active = computed(() => systems.find(s => s.key === activeKey.value))

const installCmd = computed(() => {
  const g = active.value.group   // e.g. "espnet[asr]"
  return activePkg.value === 'pip'
    ? `pip install "${g}"`
    : `uv pip install "${g}"`
})

function copy() {
  navigator.clipboard.writeText(installCmd.value).then(() => {
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  })
}
</script>

<style scoped>
/* ── Section wrapper ───────────────────────────────── */
.home-quickstart :deep(.home-section-inner) {
  padding-top: 12px;
}

/* ── Layout ────────────────────────────────────────── */
.qs-layout {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 12px;
  align-items: start;
}

/* ── System list ───────────────────────────────────── */
.qs-list {
  background: var(--bg-color, #ffffff);
  border: 0.5px solid #e2e6ec;
  border-radius: 10px;
  overflow: hidden;
}

.qs-list-head {
  font-family: var(--font-family-code);
  font-size: 0.75em;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #9aaabb;
  padding: 10px 14px 8px;
  border-bottom: 0.5px solid #f0f3f6;
}

.qs-list-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 9px 14px;
  cursor: pointer;
  transition: background 0.12s;
  border-bottom: 0.5px solid #f7f8fa;
  font-family: var(--font-family-code);
  font-size: 0.85em;
  font-weight: 600;
  color: #3a4a5c;
}
.qs-list-item:last-child { border-bottom: none; }
.qs-list-item:hover { background: #f7f8fa; color: #0d1520; }
.qs-list-item.active { background: #edf8f3; color: #0f7a5a; }

.qs-list-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: transparent;
  flex-shrink: 0;
  transition: background 0.12s;
}
.qs-list-item.active .qs-list-dot { background: #0f7a5a; }

/* ── Right panel ───────────────────────────────────── */
.qs-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* ── System detail ─────────────────────────────────── */
.qs-detail {
  background: var(--bg-color, #ffffff);
  border: 0.5px solid #e2e6ec;
  border-radius: 10px;
  padding: 14px 16px;
}

.qs-detail-name {
  font-size: 1em;
  font-weight: 700;
  color: #0d1520;
  margin-bottom: 2px;
}

.qs-detail-group {
  font-family: var(--font-family-code);
  font-size: 0.8em;
  color: #0f7a5a;
  margin-bottom: 10px;
}

.qs-detail-desc {
  font-size: 0.9em;
  color: #3a4a5c;
  line-height: 1.55;
  margin-bottom: 12px;
}

.qs-pkg-label {
  font-family: var(--font-family-code);
  font-size: 0.75em;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #9aaabb;
  margin-bottom: 8px;
}

.qs-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.qs-badge {
  font-family: var(--font-family-code);
  font-size: 0.8em;
  padding: 2px 8px;
  border-radius: 4px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}

.qs-badge-pypi {
  background: #edf8f3;
  color: #0f7a5a;
  border: 0.5px solid rgba(15, 122, 90, 0.2);
}

.qs-badge-github {
  background: #eef1ff;
  color: #4254c8;
  border: 0.5px solid rgba(66, 84, 200, 0.2);
}

.qs-badge-note {
  font-size: 0.75em;
  opacity: 0.7;
  font-weight: 600;
  letter-spacing: 0.04em;
}

/* ── Install box ───────────────────────────────────── */
.qs-install {
  background: var(--bg-color, #ffffff);
  border: 0.5px solid #e2e6ec;
  border-radius: 10px;
  padding: 14px 16px;
}

.qs-install-label {
  font-family: var(--font-family-code);
  font-size: 0.75em;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #9aaabb;
  margin-bottom: 10px;
}

.qs-pkg-opts {
  display: flex;
  gap: 4px;
  margin-bottom: 10px;
}

.qs-pkg-opt {
  font-family: var(--font-family-code);
  font-size: 0.8em;
  padding: 3px 11px;
  border-radius: 5px;
  border: 0.5px solid #d0d7e2;
  background: #f7f8fa;
  color: #3a4a5c;
  cursor: pointer;
  transition: all 0.15s;
}
.qs-pkg-opt:hover {
  border-color: rgba(15, 122, 90, 0.3);
  color: #0f7a5a;
}
.qs-pkg-opt.active {
  background: #0f7a5a;
  border-color: #0f7a5a;
  color: #ffffff;
}

.qs-cmd {
  background: var(--code-bg-color);
  border-radius: 8px;
  padding: 11px 14px;
}

.qs-cmd-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.qs-cmd-badge {
  font-family: var(--font-family-code);
  font-size: 0.75em;
  color: var(--code-color, #9aaabb);
  opacity: 0.6;
  letter-spacing: 0.06em;
}

.qs-copy-btn {
  font-size: 0.75em;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 5px;
  border: 0.5px solid rgba(255, 255, 255, 0.12);
  background: transparent;
  color: var(--code-color, #9aaabb);
  opacity: 0.7;
  cursor: pointer;
  transition: all 0.15s;
  min-width: 68px;
  text-align: center;
}
.qs-copy-btn:hover,
.qs-copy-btn.copied {
  border-color: #0f7a5a;
  color: #0f7a5a;
  opacity: 1;
}

.qs-cmd-line {
  font-family: var(--font-family-code);
  font-size: 0.85em;
  color: var(--code-color, #e2e8f0);
  display: flex;
  align-items: center;
  gap: 8px;
}

.qs-cmd-prefix { color: #0f7a5a; user-select: none; }
.qs-cmd-text   { display: inline-block; }

.qs-install-note {
  font-family: var(--font-family-code);
  font-size: 0.8em;
  color: #9aaabb;
  margin-top: 10px;
}
.qs-install-note a {
  color: #0f7a5a;
  text-decoration: none;
}
.qs-install-note a:hover { text-decoration: underline; }

/* ── Responsive ────────────────────────────────────── */
@media (max-width: 520px) {
  .qs-layout {
    grid-template-columns: 1fr;
  }
  .qs-list {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
  }
  .qs-list-head { grid-column: 1 / -1; }
}

/* ── Transition ────────────────────────────────────── */
.cmd-fade-enter-active,
.cmd-fade-leave-active {
  transition: opacity 0.15s, transform 0.15s;
}
.cmd-fade-enter-from,
.cmd-fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>