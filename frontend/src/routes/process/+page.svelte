<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { processing } from '$lib/stores/processing';
  import { getModels } from '$lib/api';
  import type { ModelInfo } from '$lib/api';

  import ModelSelector from '$lib/components/ModelSelector.svelte';
  import PathInput from '$lib/components/PathInput.svelte';
  import AdvancedSettings from '$lib/components/AdvancedSettings.svelte';
  import CaptionFeed from '$lib/components/CaptionFeed.svelte';
  import LiveVideoFeed from '$lib/components/LiveVideoFeed.svelte';

  let models = $state<ModelInfo[]>([]);
  let collections = $state<Record<string, string>>({});
  let selectedModel = $state('lfm2.5');
  let videoPath = $state('');
  let useCustomPrompt = $state(false);
  let systemPrompt = $state('');

  // Advanced settings
  let captionInterval = $state(1.0);
  let maxNewTokens = $state(500);
  let maxFrames = $state(0);
  let fps = $state(20);
  let clipDuration = $state(4.0);

  // Detection settings
  let activityDetectionEnabled = $state(false);
  let activityThreshold = $state(0.85);
  let yoloEnabled = $state(false);
  let yoloModel = $state('yolov8n.pt');
  let yoloConfidence = $state(0.5);
  let yoloTracking = $state(true);

  let isProcessing = $state(false);
  let captions = $state<any[]>([]);
  let statusMessage = $state('');
  let errorMessage = $state('');
  let videoName = $state('');
  let modelName = $state('');
  let frame = $state<ArrayBuffer | null>(null);

  const unsubIsProcessing = processing.isProcessing.subscribe((v) => (isProcessing = v));
  const unsubCaptions = processing.captions.subscribe((v) => (captions = v));
  const unsubStatus = processing.statusMessage.subscribe((v) => (statusMessage = v));
  const unsubError = processing.errorMessage.subscribe((v) => (errorMessage = v));
  const unsubVideoName = processing.videoName.subscribe((v) => (videoName = v));
  const unsubModelName = processing.modelName.subscribe((v) => (modelName = v));
  const unsubFrame = processing.frame.subscribe((v) => (frame = v));

  let currentModel = $derived(models.find((m) => m.type === selectedModel));
  let defaultPrompt = $derived(currentModel?.default_prompt ?? '');
  let startDisabled = $derived(isProcessing || !videoPath.trim());

  onMount(async () => {
    try {
      const res = await getModels();
      models = res.models;
      collections = res.collections;
    } catch (e) {
      console.error('Failed to load models:', e);
    }
    processing.connect();
  });

  onDestroy(() => {
    unsubIsProcessing();
    unsubCaptions();
    unsubStatus();
    unsubError();
    unsubVideoName();
    unsubModelName();
    unsubFrame();
  });

  function handleStart() {
    const collectionName = collections[selectedModel] || `captions_${selectedModel}`;
    processing.startProcessing({
      video_path: videoPath,
      model_type: selectedModel,
      system_prompt: useCustomPrompt ? systemPrompt || null : null,
      caption_interval: captionInterval,
      max_new_tokens: maxNewTokens,
      max_frames: maxFrames,
      collection_name: collectionName,
      fps,
      clip_duration: clipDuration,
      activity_detection_enabled: activityDetectionEnabled,
      activity_detection_threshold: activityThreshold,
      yolo_enabled: yoloEnabled,
      yolo_model: yoloModel,
      yolo_confidence: yoloConfidence,
      yolo_tracking: yoloEnabled && yoloTracking,
      stream_width: 960
    });
  }

  function handleStop() {
    processing.stopProcessing();
    processing.reset();
  }
</script>

<svelte:head>
  <title>Video Processing — Semantic Video Search</title>
</svelte:head>

{#if !isProcessing}
  <!-- ── Setup view ─────────────────────────────────────────────── -->
  <div class="page-header">
    <h1>Video Processing</h1>
    <p class="page-subtitle">
      Caption a video with a Vision-Language Model, detect objects with YOLO, and watch it all live.
    </p>
  </div>

  <div class="card setup-card">
    <div class="control-row">
      <div class="path-col">
        <PathInput bind:value={videoPath} />
      </div>
      <div class="model-col">
        <ModelSelector
          {models}
          selected={selectedModel}
          onchange={(v) => (selectedModel = v)}
        />
      </div>
    </div>

    <div class="prompt-section">
      <label class="checkbox-label">
        <input type="checkbox" bind:checked={useCustomPrompt} />
        Use custom system prompt
      </label>

      {#if useCustomPrompt}
        <textarea
          bind:value={systemPrompt}
          rows="5"
          placeholder={defaultPrompt || 'Enter a system prompt...'}
        ></textarea>
      {:else if defaultPrompt}
        <div class="text-muted">Default prompt: <em>{defaultPrompt}</em></div>
      {:else}
        <div class="text-muted">No system prompt for this model.</div>
      {/if}
    </div>

    <div class="divider"></div>

    <div class="detection-grid">
      <div class="detection-card">
        <label class="checkbox-label">
          <input type="checkbox" bind:checked={activityDetectionEnabled} />
          Activity Detection (SSIM)
        </label>
        {#if activityDetectionEnabled}
          <div class="slider-row">
            <label class="slider-label" for="activity-threshold">
              Threshold: {activityThreshold.toFixed(2)}
            </label>
            <input
              id="activity-threshold"
              type="range"
              min="0.5"
              max="0.99"
              step="0.01"
              bind:value={activityThreshold}
            />
          </div>
        {/if}
      </div>

      <div class="detection-card">
        <label class="checkbox-label">
          <input type="checkbox" bind:checked={yoloEnabled} />
          YOLO Object Detection
        </label>
        {#if yoloEnabled}
          <div class="slider-row">
            <label class="slider-label" for="yolo-model">Model:</label>
            <select id="yolo-model" bind:value={yoloModel}>
              <option value="yolov8n.pt">YOLOv8 Nano (fastest)</option>
              <option value="yolov8s.pt">YOLOv8 Small</option>
              <option value="yolov8m.pt">YOLOv8 Medium</option>
            </select>
          </div>
          <div class="slider-row">
            <label class="slider-label" for="yolo-confidence">
              Confidence: {yoloConfidence.toFixed(2)}
            </label>
            <input
              id="yolo-confidence"
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              bind:value={yoloConfidence}
            />
          </div>
          <label class="checkbox-label">
            <input
              type="checkbox"
              bind:checked={yoloTracking}
              disabled={!yoloEnabled}
            />
            Object Tracking (movement trails &amp; direction)
          </label>
        {/if}
      </div>
    </div>

    <div class="divider"></div>

    <AdvancedSettings
      bind:captionInterval
      bind:maxNewTokens
      bind:maxFrames
      bind:fps
      bind:clipDuration
    />

    {#if errorMessage}
      <div class="error-banner">{errorMessage}</div>
    {/if}

    {#if statusMessage && !errorMessage}
      <div class="status-banner">{statusMessage}</div>
    {/if}

    <button class="btn-primary btn-lg full-width" onclick={handleStart} disabled={startDisabled}>
      Start Processing
    </button>
  </div>
{:else}
  <!-- ── Processing view ────────────────────────────────────────── -->
  <div class="processing-topbar">
    <div class="processing-meta">
      <div class="processing-title">
        <span class="meta-label">Video</span>
        <span class="meta-value">{videoName}</span>
      </div>
      <div class="processing-title">
        <span class="meta-label">Captioning Model</span>
        <span class="meta-value mono">{modelName}</span>
      </div>
      <span class="badge badge-live"><span class="live-dot"></span>Processing</span>
    </div>
    <button class="btn-danger btn-lg" onclick={handleStop}>Stop Processing</button>
  </div>

  {#if errorMessage}
    <div class="error-banner">{errorMessage}</div>
  {/if}

  <LiveVideoFeed {frame} {videoName} />

  <div class="captions-section">
    <div class="captions-header">
      <h2>Live Captions</h2>
      <span class="badge badge-count">{captions.length} captions</span>
    </div>
    <CaptionFeed {captions} {isProcessing} />
  </div>
{/if}

<style>
  .page-header {
    margin-bottom: 1.5rem;
  }
  .page-header h1 {
    font-size: 1.5rem;
    margin-bottom: 0.25rem;
  }
  .page-subtitle {
    color: var(--text-muted);
    font-size: 0.92rem;
  }
  .card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow);
  }
  .setup-card {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }
  .control-row {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 1rem;
  }
  .path-col,
  .model-col {
    display: flex;
    flex-direction: column;
  }
  .prompt-section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  textarea {
    resize: vertical;
  }
  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    color: var(--text);
    cursor: pointer;
    margin: 0;
  }
  .checkbox-label input[type="checkbox"] {
    accent-color: var(--accent);
    width: auto;
  }
  .detection-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  .detection-card {
    padding: 1rem;
    background: var(--bg);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .slider-row {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .slider-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 0;
  }
  .slider-row input[type="range"] {
    width: 100%;
    accent-color: var(--accent);
  }
  .slider-row select {
    padding: 0.3rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-elevated);
    color: var(--text);
    font-size: 0.85rem;
  }
  .divider {
    border-top: 1px solid var(--border);
  }
  .error-banner {
    padding: 0.6rem 1rem;
    background: rgba(224, 85, 85, 0.12);
    border: 1px solid var(--error);
    border-radius: var(--radius);
    color: var(--error);
  }
  .status-banner {
    padding: 0.6rem 1rem;
    background: rgba(108, 140, 255, 0.1);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    color: var(--accent);
    font-size: 0.9rem;
  }
  .processing-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
  }
  .processing-meta {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
  }
  .processing-title {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .meta-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
  }
  .meta-value {
    font-size: 1.05rem;
    font-weight: 600;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .badge-live {
    background: rgba(224, 85, 85, 0.14);
    border: 1px solid rgba(224, 85, 85, 0.5);
    color: #ff8a8a;
    text-transform: uppercase;
  }
  .badge-count {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    color: var(--text-muted);
  }
  .live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--error);
    animation: live-pulse 1.4s ease-in-out infinite;
  }
  .captions-section {
    margin-top: 1.25rem;
  }
  .captions-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
  }
  .captions-header h2 {
    font-size: 1.05rem;
  }
  @keyframes live-pulse {
    0%,
    100% {
      opacity: 1;
      box-shadow: 0 0 0 0 rgba(224, 85, 85, 0.5);
    }
    50% {
      opacity: 0.6;
      box-shadow: 0 0 0 5px rgba(224, 85, 85, 0);
    }
  }
  @media (max-width: 700px) {
    .control-row {
      grid-template-columns: 1fr;
    }
    .detection-grid {
      grid-template-columns: 1fr;
    }
    .processing-topbar {
      align-items: flex-start;
    }
  }
</style>
