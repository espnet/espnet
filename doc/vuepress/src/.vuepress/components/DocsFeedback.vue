<script setup>
import { computed, ref } from "vue";

const feedbackEndpoint = "https://script.google.com/macros/s/AKfycbxB8Zums5cY5v-W_l6_rgC3MB6awTW_P76ezE49DNow8t8b4rU5YeXeEYFXPEFoeyj1/exec";

const selection = ref(null);
const comment = ref("");
const submitted = ref(false);

const canSubmitComment = computed(() => comment.value.trim().length > 0);

const fireGaEvent = (helpful) => {
  if (typeof window.gtag === "function") {
    window.gtag("event", "feedback_click", {
      page_path: window.location.pathname,
      helpful,
    });
  }
};

const handleYes = () => {
  fireGaEvent(true);
  selection.value = "yes";
  submitted.value = true;
};

const handleNo = () => {
  fireGaEvent(false);
  selection.value = "no";
  submitted.value = false;
};

const submitComment = () => {
  if (!canSubmitComment.value) return;

  void fetch(feedbackEndpoint, {
    method: "POST",
    mode: "no-cors",
    headers: {
      "Content-Type": "text/plain",
    },
    body: JSON.stringify({
      page: window.location.pathname,
      helpful: false,
      comment: comment.value.trim(),
      userAgent: navigator.userAgent,
    }),
  });

  submitted.value = true;
};
</script>

<template>
  <section class="docs-feedback" aria-label="Page feedback">
    <div class="docs-feedback__title">Is this helpful?</div>

    <template v-if="submitted">
      <p class="docs-feedback__message">Thank you for the feedback.</p>
    </template>

    <template v-else-if="selection === 'no'">
      <label class="docs-feedback__prompt" for="docs-feedback-comment">
        Tell us what was missing
      </label>

      <textarea
        id="docs-feedback-comment"
        v-model="comment"
        class="docs-feedback__textarea"
        rows="3"
        placeholder="Short comment"
      />

      <div class="docs-feedback__actions">
        <button class="docs-feedback__button" type="button" @click="submitComment">
          Send
        </button>
      </div>
    </template>

    <div v-else class="docs-feedback__actions">
      <button class="docs-feedback__button" type="button" @click="handleYes">
        Yes
      </button>
      <button class="docs-feedback__button" type="button" @click="handleNo">
        No
      </button>
    </div>
  </section>
</template>
