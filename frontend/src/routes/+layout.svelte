<script lang="ts">
  import '../app.css';
  import { page } from '$app/stores';

  let { children } = $props();
</script>

<nav class="topnav">
  <div class="nav-brand">
    <a href="/process">Semantic Video Search</a>
  </div>
  <div class="nav-links">
    <a href="/process" class:active={$page.url.pathname.startsWith('/process')}>Video Processing</a>
    <a href="/query" class:active={$page.url.pathname.startsWith('/query')}>Query Captions</a>
  </div>
</nav>

<main class="container">
  {@render children()}
</main>

<style>
  .topnav {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 0.75rem 2rem;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .nav-brand a {
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--text);
    text-decoration: none;
  }
  .nav-links {
    display: flex;
    gap: 0.25rem;
  }
  .nav-links a {
    padding: 0.4rem 0.85rem;
    border-radius: var(--radius);
    color: var(--text-muted);
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    transition: background 0.15s, color 0.15s;
  }
  .nav-links a:hover {
    background: var(--bg-elevated);
    color: var(--text);
    text-decoration: none;
  }
  .nav-links a.active {
    background: var(--accent);
    color: #fff;
  }
  .container {
    max-width: 960px;
    margin: 0 auto;
    padding: 2rem;
  }
</style>
