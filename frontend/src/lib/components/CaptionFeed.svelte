<script lang="ts">
  import type { Caption } from '$lib/stores/processing';

  interface Props {
    captions: Caption[];
    isProcessing: boolean;
  }

  let { captions, isProcessing }: Props = $props();

  let feedEl: HTMLDivElement | undefined = $state();
  let autoScroll = $state(true);

  $effect(() => {
    if (feedEl && autoScroll && captions.length) {
      feedEl.scrollTop = feedEl.scrollHeight;
    }
  });

  function handleScroll() {
    if (!feedEl) return;
    const nearBottom = feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight < 60;
    if (nearBottom) {
      if (!autoScroll) autoScroll = true;
    } else {
      autoScroll = false;
    }
  }

  function resume() {
    autoScroll = true;
    if (feedEl) {
      feedEl.scrollTop = feedEl.scrollHeight;
    }
  }
</script>

<div class="caption-panel">
  <div class="feed-list" bind:this={feedEl} onscroll={handleScroll}>
    {#if captions.length === 0}
      <div class="empty-state text-muted">
        {#if isProcessing}
          Waiting for the first caption...
        {:else}
          No captions yet. Set a video path and click Start Processing.
        {/if}
      </div>
    {/if}

    {#each captions as entry}
      <div class="caption-entry">
        <div class="caption-header">
          <span class="timestamp">[{entry.time}]</span>
          <span class="frame-info mono">{entry.frame >= 0 ? `Frame ${entry.frame}` : 'System'}</span>
          {#if entry.video_ts}
            <span class="video-ts mono">| Video {entry.video_ts}</span>
          {/if}
        </div>
        <div class="caption-text">{entry.caption}</div>
        {#if entry.yolo_tracks && entry.yolo_tracks.length > 0}
          <div class="yolo-tracks">
            {#each entry.yolo_tracks as t}
              <span class="track-chip">
                {t.class} → {t.direction}
              </span>
            {/each}
          </div>
        {:else if entry.movement_summary}
          <div class="yolo-objects">
            <span class="yolo-label">Movement:</span>
            {entry.movement_summary}
          </div>
        {/if}
        {#if entry.yolo_objects && entry.yolo_objects.length > 0}
          <div class="yolo-objects">
            <span class="yolo-label">Objects:</span>
            {entry.yolo_objects.join(', ')}
          </div>
        {/if}
      </div>
    {/each}
  </div>

  {#if !autoScroll && captions.length > 0}
    <button class="jump-btn" onclick={resume}>
      <span class="jump-icon">⌄</span>
      Jump to latest
    </button>
  {/if}
</div>

<style>
  .caption-panel {
    position: relative;
    display: flex;
    flex-direction: column;
  }
  .feed-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    max-height: 300px;
    min-height: 90px;
    overflow-y: auto;
    padding: 0.25rem 0.5rem 0.5rem 0.25rem;
    scrollbar-gutter: stable;
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
  .yolo-objects {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
    font-style: italic;
  }
  .yolo-label {
    color: var(--accent);
    font-weight: 500;
    font-style: normal;
  }
  .yolo-tracks {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.35rem;
  }
  .track-chip {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: rgba(99, 179, 255, 0.12);
    color: var(--accent);
    border: 1px solid rgba(99, 179, 255, 0.35);
    white-space: nowrap;
  }
  .empty-state {
    text-align: center;
    padding: 2rem;
    font-style: italic;
  }
  .jump-btn {
    position: absolute;
    bottom: 1rem;
    right: 1rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    background: var(--accent);
    color: #fff;
    font-size: 0.82rem;
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  }
  .jump-btn:hover {
    background: var(--accent-hover);
  }
  .jump-icon {
    font-size: 1rem;
    line-height: 1;
  }
</style>
