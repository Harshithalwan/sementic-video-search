<script lang="ts">
  import type { Caption } from '$lib/stores/processing';

  interface Props {
    captions: Caption[];
    isProcessing: boolean;
  }

  let { captions, isProcessing }: Props = $props();

  let feedEl: HTMLDivElement | undefined = $state();

  $effect(() => {
    if (feedEl && captions.length) {
      feedEl.scrollTop = 0;
    }
  });
</script>

<div class="caption-feed">
  {#if isProcessing}
    <div class="processing-indicator">
      <span class="spinner"></span>
      Processing in progress...
    </div>
  {/if}

  <div class="feed-list" bind:this={feedEl}>
    {#each [...captions].reverse() as entry}
      <div class="caption-entry">
        <div class="caption-header">
          <span class="timestamp">[{entry.time}]</span>
          <span class="frame-info mono">
            {entry.frame >= 0 ? `Frame ${entry.frame}` : 'System'}
          </span>
          {#if entry.video_ts}
            <span class="video-ts mono">| Video {entry.video_ts}</span>
          {/if}
        </div>
        <div class="caption-text">{entry.caption}</div>
      </div>
    {/each}

    {#if captions.length === 0 && !isProcessing}
      <div class="empty-state text-muted">
        No captions yet. Set a video path and click Start Processing.
      </div>
    {/if}
  </div>
</div>

<style>
  .caption-feed {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .processing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.65rem 1rem;
    background: var(--bg-elevated);
    border-radius: var(--radius);
    color: var(--accent);
    font-size: 0.9rem;
  }
  .feed-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    max-height: 500px;
    overflow-y: auto;
  }
  .caption-entry {
    padding: 0.6rem 0.85rem;
    background: var(--bg-surface);
    border-radius: var(--radius);
    border-left: 3px solid var(--accent);
  }
  .caption-header {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    font-size: 0.8rem;
    margin-bottom: 0.2rem;
    flex-wrap: wrap;
  }
  .timestamp {
    color: var(--accent);
    font-weight: 500;
  }
  .frame-info {
    color: var(--text-muted);
  }
  .video-ts {
    color: var(--text-muted);
  }
  .caption-text {
    font-size: 0.95rem;
    line-height: 1.5;
  }
  .empty-state {
    text-align: center;
    padding: 2rem;
    font-style: italic;
  }
</style>
