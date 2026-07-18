<script lang="ts">
  import type { ModelInfo } from '$lib/api';

  interface Props {
    models: ModelInfo[];
    selected: string;
    disabled?: boolean;
    onchange: (modelType: string) => void;
  }

  let { models, selected, disabled = false, onchange }: Props = $props();

  function handleChange(e: Event) {
    const val = (e.target as HTMLSelectElement).value;
    onchange(val);
  }

  let currentModel = $derived(models.find((m) => m.type === selected));
</script>

<div class="model-selector">
  <label for="model-select">Visual Captioning Model</label>
  <select id="model-select" value={selected} {disabled} onchange={handleChange}>
    {#each models as m}
      <option value={m.type}>{m.type}</option>
    {/each}
  </select>

  {#if currentModel}
    <div class="model-id mono">{currentModel.default_id}</div>
  {/if}
</div>

<style>
  .model-selector {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .model-id {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
  }
</style>
