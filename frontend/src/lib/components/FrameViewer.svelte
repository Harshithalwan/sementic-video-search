<script lang="ts">
  interface Props {
    source: string;
    timestampMs: number;
    show: boolean;
    onclose: () => void;
  }

  let { source, timestampMs, show, onclose }: Props = $props();

  let loading = $state(false);
  let loadError = $state('');

  let frameSrc = $derived(
    `/api/frame?path=${encodeURIComponent(source)}&timestamp_ms=${encodeURIComponent(timestampMs)}`
  );

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && show) {
      onclose();
    }
  }

  $effect(() => {
    if (show) {
      document.addEventListener('keydown', handleKeydown);
      loading = true;
      loadError = '';
    }
    return () => {
      document.removeEventListener('keydown', handleKeydown);
    };
  });
</script>

{#if show}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="backdrop" role="presentation" onclick={onclose}>
    <div class="modal" role="dialog" tabindex="-1" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <span class="modal-title">{source.split(/[/\\]/).pop()}</span>
        <button class="close-btn" onclick={onclose}>&times;</button>
      </div>
      <div class="frame-wrap">
        {#if loading}
          <div class="status-text text-muted">Loading frame…</div>
        {:else if loadError}
          <div class="status-text text-error">{loadError}</div>
        {:else}
          <img
            class="frame-img"
            src={frameSrc}
            alt="Video frame at timestamp"
            onload={() => (loading = false)}
            onerror={() => {
              loading = false;
              loadError = 'Could not load frame for this result.';
            }}
          />
        {/if}
      </div>
      <div class="modal-footer">
        <span>Frame at {(timestampMs / 1000).toFixed(1)}s</span>
      </div>
    </div>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 1000;
    background: rgba(0, 0, 0, 0.75);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .modal {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    width: 90vw;
    max-width: 960px;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
  }
  .modal-title {
    font-weight: 500;
    font-size: 0.95rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .close-btn {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 1.5rem;
    line-height: 1;
    padding: 0.2rem 0.5rem;
    cursor: pointer;
    border-radius: var(--radius);
  }
  .close-btn:hover {
    color: var(--text);
    background: var(--bg-elevated);
  }
  .frame-wrap {
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    overflow: hidden;
  }
  .frame-img {
    width: 100%;
    max-height: 70vh;
    object-fit: contain;
  }
  .status-text {
    font-size: 0.95rem;
    padding: 1.5rem;
  }
  .text-error {
    color: var(--error);
  }
  .modal-footer {
    padding: 0.5rem 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--text-muted);
  }
</style>
