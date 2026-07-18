<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { processing } from '$lib/stores/processing';
  import { getModels } from '$lib/api';
  import type { ModelInfo } from '$lib/api';

  import ModelSelector from '$lib/components/ModelSelector.svelte';
  import PathInput from '$lib/components/PathInput.svelte';
  import AdvancedSettings from '$lib/components/AdvancedSettings.svelte';
  import CaptionFeed from '$lib/components/CaptionFeed.svelte';

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

  let isProcessing = $state(false);
  let captions = $state<any[]>([]);
  let statusMessage = $state('');
  let errorMessage = $state('');

  const unsubIsProcessing = processing.isProcessing.subscribe((v) => (isProcessing = v));
  const unsubCaptions = processing.captions.subscribe((v) => (captions = v));
  const unsubStatus = processing.statusMessage.subscribe((v) => (statusMessage = v));
  const unsubError = processing.errorMessage.subscribe((v) => (errorMessage = v));

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
      yolo_confidence: yoloConfidence
    });
  }

  function handleStop() {
    processing.stopProcessing();
  }
</script>

<svelte:head>
  <title>Video Processing — Semantic Video Search</title>
</svelte:head>

<h1>Process Video</h1>

<div class="top-controls">
  <div class="control-row">
    <div class="path-col">
      <PathInput bind:value={videoPath} disabled={isProcessing} />
    </div>
    <div class="model-col">
      <ModelSelector
        {models}
        selected={selectedModel}
        disabled={isProcessing}
        onchange={(v) => (selectedModel = v)}
      />
    </div>
  </div>
</div>

<hr />

<div class="prompt-section">
  <label class="checkbox-label">
    <input type="checkbox" bind:checked={useCustomPrompt} disabled={isProcessing} />
    Use custom system prompt
  </label>

  {#if useCustomPrompt}
    <textarea
      bind:value={systemPrompt}
      rows="5"
      disabled={isProcessing}
      placeholder={defaultPrompt || 'Enter a system prompt...'}
    ></textarea>
  {:else if defaultPrompt}
    <div class="text-muted">Default prompt: <em>{defaultPrompt}</em></div>
  {:else}
    <div class="text-muted">No system prompt for this model.</div>
  {/if}
</div>

<AdvancedSettings
  bind:captionInterval
  bind:maxNewTokens
  bind:maxFrames
  bind:fps
  bind:clipDuration
  disabled={isProcessing}
/>

<div class="detection-section">
  <h3>Detection Settings</h3>

  <div class="detection-grid">
    <div class="detection-card">
      <label class="checkbox-label">
        <input type="checkbox" bind:checked={activityDetectionEnabled} disabled={isProcessing} />
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
            disabled={isProcessing}
          />
        </div>
      {/if}
    </div>

    <div class="detection-card">
      <label class="checkbox-label">
        <input type="checkbox" bind:checked={yoloEnabled} disabled={isProcessing} />
        YOLO Object Detection
      </label>
      {#if yoloEnabled}
        <div class="slider-row">
          <label class="slider-label" for="yolo-model">
            Model:
          </label>
          <select id="yolo-model" bind:value={yoloModel} disabled={isProcessing}>
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
            disabled={isProcessing}
          />
        </div>
      {/if}
    </div>
  </div>
</div>

<hr />

<div class="button-row">
  <button class="btn-primary full-width" onclick={handleStart} disabled={startDisabled}>
    Start Processing
  </button>
  <button class="btn-danger full-width" onclick={handleStop} disabled={!isProcessing}>
    Stop Processing
  </button>
</div>

<hr />

<h2>Live Captions</h2>

{#if errorMessage}
  <div class="error-banner">{errorMessage}</div>
{/if}

{#if statusMessage && !errorMessage}
  <div class="status-banner">{statusMessage}</div>
{/if}

<CaptionFeed {captions} {isProcessing} />

<style>
  h1 {
    font-size: 1.4rem;
    margin-bottom: 1rem;
  }
  h2 {
    font-size: 1.15rem;
    margin-bottom: 0.5rem;
  }
  .top-controls {
    margin-bottom: 0.5rem;
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
  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
  }
  .prompt-section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
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
  }
  .checkbox-label input[type="checkbox"] {
    accent-color: var(--accent);
  }
  .detection-section {
    margin-bottom: 1rem;
  }
  .detection-section h3 {
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
    color: var(--text-muted);
  }
  .detection-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  .detection-card {
    padding: 0.75rem;
    background: var(--bg-surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .slider-row {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .slider-label {
    font-size: 0.8rem;
    color: var(--text-muted);
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
  .button-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }
  .error-banner {
    padding: 0.6rem 1rem;
    background: rgba(224, 85, 85, 0.12);
    border: 1px solid var(--error);
    border-radius: var(--radius);
    color: var(--error);
    margin-bottom: 0.5rem;
  }
  .status-banner {
    padding: 0.6rem 1rem;
    background: rgba(108, 140, 255, 0.1);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    color: var(--accent);
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }
  @media (max-width: 700px) {
    .control-row {
      grid-template-columns: 1fr;
    }
    .button-row {
      grid-template-columns: 1fr;
    }
  }
</style>
