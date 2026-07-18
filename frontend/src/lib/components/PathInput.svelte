<script lang="ts">
  interface Props {
    value: string;
    disabled?: boolean;
  }

  let { value = $bindable(), disabled = false }: Props = $props();
</script>

<div class="path-input">
  <label for="video-path">Video File Path</label>
  <input
    id="video-path"
    type="text"
    bind:value
    {disabled}
    placeholder="e.g. testData/wildlife-documentry.mp4"
  />
  {#if value}
    <div class="path-hint mono">{value}</div>
  {/if}
</div>

<style>
  .path-input {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .path-hint {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
