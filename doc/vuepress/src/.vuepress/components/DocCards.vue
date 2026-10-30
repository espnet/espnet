<template>
  <div class="doc-cards" :style="colsStyle">
    <slot />
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  cols: { type: [Number, String], default: 2 },
})

const colsStyle = computed(() => ({
  '--doc-cards-cols': String(props.cols),
}))
</script>

<style scoped>
.doc-cards {
  display: grid;
  grid-template-columns: repeat(var(--doc-cards-cols, 2), minmax(0, 1fr));
  gap: 12px;
  margin: 1.5rem 0;
}

@media (max-width: 640px) {
  .doc-cards {
    grid-template-columns: 1fr;
  }
}
</style>