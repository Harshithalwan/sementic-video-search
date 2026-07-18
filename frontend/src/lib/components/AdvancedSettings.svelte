<script lang="ts">
  interface Props {
    captionInterval: number;
    maxNewTokens: number;
    maxFrames: number;
    fps: number;
    clipDuration: number;
    disabled?: boolean;
  }

  let {
    captionInterval = $bindable(),
    maxNewTokens = $bindable(),
    maxFrames = $bindable(),
    fps = $bindable(),
    clipDuration = $bindable(),
    disabled = false
  }: Props = $props();
</script>

<details>
  <summary>Advanced Settings</summary>
  <div class="content">
    <div class="grid-3">
      <div class="field">
        <label for="caption-interval">Caption Interval (sec)</label>
        <input
          id="caption-interval"
          type="range"
          min="0.25"
          max="5"
          step="0.25"
          bind:value={captionInterval}
          {disabled}
        />
        <span class="value-display mono">{captionInterval}</span>
      </div>
      <div class="field">
        <label for="max-tokens">Max New Tokens</label>
        <input
          id="max-tokens"
          type="range"
          min="8"
          max="500"
          step="8"
          bind:value={maxNewTokens}
          {disabled}
        />
        <span class="value-display mono">{maxNewTokens}</span>
      </div>
      <div class="field">
        <label for="max-frames">Max Captions (0 = unlimited)</label>
        <input
          id="max-frames"
          type="number"
          min="0"
          bind:value={maxFrames}
          {disabled}
        />
      </div>
    </div>
    <div class="grid-2">
      <div class="field">
        <label for="fps">Sampling FPS (video-native)</label>
        <input
          id="fps"
          type="range"
          min="1"
          max="30"
          step="1"
          bind:value={fps}
          {disabled}
        />
        <span class="value-display mono">{fps}</span>
      </div>
      <div class="field">
        <label for="clip-duration">Clip Duration (sec, video-native)</label>
        <input
          id="clip-duration"
          type="range"
          min="1"
          max="10"
          step="0.5"
          bind:value={clipDuration}
          {disabled}
        />
        <span class="value-display mono">{clipDuration}</span>
      </div>
    </div>
  </div>
</details>

<style>
  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .value-display {
    font-size: 0.8rem;
    color: var(--accent);
    text-align: center;
  }
  input[type="range"] {
    accent-color: var(--accent);
  }
  @media (max-width: 700px) {
    .grid-3,
    .grid-2 {
      grid-template-columns: 1fr;
    }
  }
</style>
