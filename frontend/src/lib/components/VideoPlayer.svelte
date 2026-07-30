<script lang="ts">
  import { tick } from 'svelte';

  interface Props {
    videoSource: string;
    timestampMs: number;
    show: boolean;
    onclose: () => void;
  }

  let { videoSource, timestampMs, show, onclose }: Props = $props();

  let videoEl = $state<HTMLVideoElement | null>(null);

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && show) {
      onclose();
    }
  }

  $effect(() => {
    if (show) {
      document.addEventListener('keydown', handleKeydown);
      tick().then(() => {
        if (videoEl) {
          videoEl.src = `/api/video?path=${encodeURIComponent(videoSource)}`;
          videoEl.load();
          videoEl.onloadedmetadata = () => {
            videoEl!.currentTime = timestampMs / 1000;
          };
        }
      });
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
        <span class="modal-title">{videoSource.split(/[/\\]/).pop()}</span>
        <button class="close-btn" onclick={onclose}>&times;</button>
      </div>
      <div class="player-wrap">
        <video
          bind:this={videoEl}
          controls
          autoplay
          class="video-el"
        >
          <track kind="captions" />
        </video>
      </div>
      <div class="modal-footer">
        <span class="timestamp-label">Seeking to {Math.floor(timestampMs / 1000)}s ({(timestampMs / 1000).toFixed(1)}s)</span>
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
  .player-wrap {
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 300px;
  }
  .video-el {
    width: 100%;
    max-height: 70vh;
  }
  .modal-footer {
    padding: 0.5rem 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--text-muted);
  }
</style>
