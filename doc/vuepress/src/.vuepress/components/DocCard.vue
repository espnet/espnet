<template>
  <a :href="href" class="doc-card" :class="{ 'doc-card--external': external }">
    <div v-if="icon" class="doc-card__icon" aria-hidden="true">
      <HopeIcon :icon="icon" />
    </div>
    <div class="doc-card__body">
      <p class="doc-card__title">{{ title }}</p>
      <p v-if="desc" class="doc-card__desc">{{ desc }}</p>
    </div>
    <span class="doc-card__arrow" aria-hidden="true">
      <HopeIcon :icon="external ? 'tabler:external-link' : 'tabler:arrow-right'" />
    </span>
  </a>
</template>

<script setup>
import HopeIcon from "vuepress-theme-hope/components/HopeIcon.js";

defineProps({
  title:    { type: String, required: true },
  desc:     { type: String, default: '' },
  href:     { type: String, required: true },
  icon:     { type: String, default: '' },   // Iconify icon name e.g. "tabler:file-code"
  external: { type: Boolean, default: false },
})
</script>

<style scoped>
.doc-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 20px;
  background: var(--vp-c-bg);
  border: 1px solid #e0e0e0;
  border-radius: 10px;
  text-decoration: none;
  color: inherit;
  transition: border-color 0.15s ease, background 0.15s ease;
  cursor: pointer;
}

.doc-card:hover {
  border-color: var(--vp-c-brand-2, #3ecf8e);
  background: var(--vp-c-bg-soft);
}

.doc-card:hover .doc-card__title {
  color: var(--vp-c-text-1);
}

.doc-card:hover .doc-card__arrow {
  transform: translateX(3px);
  color: var(--vp-c-brand-2, #3ecf8e);
}

.doc-card__icon {
  width: 34px;
  height: 34px;
  border-radius: 8px;
  background: color-mix(in srgb, var(--vp-c-brand-2, #3ecf8e) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--vp-c-brand-2, #3ecf8e) 20%, transparent);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--vp-c-brand-2, #3ecf8e);
  font-size: 17px;
  flex-shrink: 0;
}

.doc-card__body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.doc-card__title {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
  color: var(--vp-c-text-2);
  line-height: 1.3;
  transition: color 0.15s;
}

.doc-card__desc {
  margin: 0;
  font-size: 12px;
  color: var(--vp-c-text-3);
  line-height: 1.6;
}

.doc-card__arrow {
  font-size: 14px;
  color: var(--vp-c-text-3);
  align-self: flex-end;
  transition: transform 0.15s ease, color 0.15s ease;
}
</style>