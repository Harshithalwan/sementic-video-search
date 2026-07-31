<script lang="ts">
  interface Props {
    frame: ArrayBuffer | null;
    videoName: string;
  }

  let { frame, videoName }: Props = $props();

  let canvasEl: HTMLCanvasElement | undefined = $state();
  let hasVideo = $state(false);

  let bitmap: ImageBitmap | null = null;
  let ctx: CanvasRenderingContext2D | null = null;
  let drawSeq = 0;
  let resizeObserver: ResizeObserver | null = null;

  async function draw(buf: ArrayBuffer) {
    const seq = ++drawSeq;
    let next: ImageBitmap;
    try {
      next = await createImageBitmap(new Blob([buf], { type: 'image/jpeg' }));
    } catch {
      return;
    }
    if (seq !== drawSeq) {
      next.close();
      return;
    }
    if (bitmap) bitmap.close();
    bitmap = next;
    hasVideo = true;
    render();
  }

  function render() {
    if (!canvasEl || !bitmap) return;
    const canvas = canvasEl;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx = ctx ?? canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    const scale = Math.min(w / bitmap.width, h / bitmap.height);
    const dw = bitmap.width * scale;
    const dh = bitmap.height * scale;
    ctx.drawImage(bitmap, (w - dw) / 2, (h - dh) / 2, dw, dh);
  }

  $effect(() => {
    if (frame) draw(frame);
  });

  $effect(() => {
    if (!canvasEl) return;
    if (!resizeObserver) {
      resizeObserver = new ResizeObserver(() => render());
    }
    resizeObserver.observe(canvasEl);
    return () => {
      resizeObserver?.disconnect();
      resizeObserver = null;
    };
  });

  $effect(() => {
    return () => {
      if (bitmap) bitmap.close();
    };
  });
</script>

<div class="video-card">
  <div class="video-stage">
    <canvas bind:this={canvasEl} class="video-canvas"></canvas>

    {#if !hasVideo}
      <div class="video-placeholder">
        <span class="placeholder-spinner"></span>
        <span>Waiting for video feed...</span>
      </div>
    {/if}

    <div class="video-hud">
      <span class="hud-live"><span class="live-dot"></span>LIVE</span>
      <span class="hud-name">{videoName}</span>
    </div>
  </div>
</div>

<style>
  .video-card {
    background: #000;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  }
  .video-stage {
    position: relative;
    aspect-ratio: 16 / 9;
    width: 100%;
    background: #05060a;
  }
  .video-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
  }
  .video-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 0.75rem;
    color: var(--text-muted);
    font-size: 0.9rem;
  }
  .placeholder-spinner {
    width: 28px;
    height: 28px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .video-hud {
    position: absolute;
    top: 0.75rem;
    left: 0.75rem;
    right: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    pointer-events: none;
  }
  .hud-live {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(224, 85, 85, 0.16);
    border: 1px solid rgba(224, 85, 85, 0.55);
    color: #ff8a8a;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--error);
    animation: live-pulse 1.4s ease-in-out infinite;
  }
  .hud-name {
    max-width: 60%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.55);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.72rem;
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
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
