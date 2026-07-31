<script lang="ts">
  import type { QueryResult } from '$lib/stores/query';

  interface Props {
    result: QueryResult;
    index: number;
    onview: (source: string, timestampMs: number) => void;
  }

  let { result, index, onview }: Props = $props();
  let open = $state(false);

  let meta = $derived(result.metadata);
</script>

<details bind:open>
  <summary>
    #{index} &mdash; Score: {result.score.toFixed(4)} &mdash;
    {meta.video_name || 'N/A'} &mdash;
    {meta.video_timestamp || 'N/A'}
  </summary>
  <div class="content">
    <div class="meta-grid">
      <div class="meta-col">
        <div class="meta-row"><span class="label">Model:</span> <span class="mono">{meta.model_type || 'N/A'}</span></div>
        <div class="meta-row"><span class="label">Video Name:</span> <span class="mono">{meta.video_name || 'N/A'}</span></div>
        <div class="meta-row"><span class="label">Video ID:</span> <span class="mono">{meta.video_id || 'N/A'}</span></div>
      </div>
      <div class="meta-col">
        <div class="meta-row"><span class="label">Video Timestamp:</span> <span class="mono">{meta.video_timestamp || 'N/A'}</span></div>
        <div class="meta-row"><span class="label">Wall-Clock Time:</span> <span class="mono">{meta.current_time || 'N/A'}</span></div>
        <div class="meta-row"><span class="label">Frame Index:</span> <span class="mono">{meta.frame_index ?? 'N/A'}</span></div>
      </div>
    </div>
    <hr />
    <div class="caption-text">
      <strong>Caption:</strong> {meta.caption || result.document}
    </div>
    {#if meta.movement_summary}
      <div class="movement-summary">
        <strong>Movement:</strong> {meta.movement_summary}
      </div>
    {/if}
    {#if meta.yolo_objects && meta.yolo_objects.length > 0}
      <div class="yolo-objects">
        <strong>Detected Objects:</strong> {meta.yolo_objects.join(', ')}
      </div>
    {/if}
    {#if meta.source}
      <div class="view-row">
        <button class="btn-primary view-btn" onclick={() => onview(meta.source, meta.video_timestamp_ms)}>
          View
        </button>
      </div>
    {/if}
  </div>
</details>

<style>
  .meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 0.75rem;
  }
  .meta-col {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .meta-row {
    font-size: 0.9rem;
  }
  .label {
    color: var(--text-muted);
  }
  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0.75rem 0;
  }
  .caption-text {
    font-size: 0.95rem;
    line-height: 1.6;
  }
  .yolo-objects {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
  }
  .movement-summary {
    font-size: 0.85rem;
    color: var(--accent);
    margin-top: 0.4rem;
  }
  .view-row {
    margin-top: 0.75rem;
  }
  .view-btn {
    font-size: 0.85rem;
    padding: 0.4rem 1rem;
  }
</style>
