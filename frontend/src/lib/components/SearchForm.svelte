<script lang="ts">
  import { getCollectionVideos } from '$lib/api';

  interface Props {
    query: string;
    topK: number;
    collection: string;
    collections: Record<string, string>;
    loading: boolean;
    onsearch: () => void;
  }

  let {
    query = $bindable(),
    topK = $bindable(),
    collection = $bindable(),
    collections,
    loading,
    onsearch
  }: Props = $props();

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') onsearch();
  }
</script>

<div class="search-form">
  <div class="search-row">
    <div class="query-field">
      <label for="search-query">Search Query</label>
      <input
        id="search-query"
        type="text"
        bind:value={query}
        placeholder="e.g. a person walking near a red car"
        onkeydown={handleKeydown}
      />
    </div>
    <div class="topk-field">
      <label for="top-k">Top K</label>
      <input id="top-k" type="number" min="1" max="100" bind:value={topK} />
    </div>
  </div>

  <div class="search-row">
    <div class="collection-field">
      <label for="collection">Collection</label>
      <select id="collection" bind:value={collection}>
        {#each Object.values(collections) as c}
          <option value={c}>{c.replace('captions_', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</option>
        {/each}
      </select>
    </div>
    <div></div>
  </div>

  <button class="btn-primary full-width" onclick={onsearch} disabled={loading || !query.trim()}>
    {#if loading}
      <span class="spinner"></span> Searching...
    {:else}
      Search
    {/if}
  </button>
</div>

<style>
  .search-form {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
  }
  .search-row {
    display: grid;
    grid-template-columns: 3fr 1fr;
    gap: 0.75rem;
  }
  .query-field,
  .topk-field,
  .collection-field {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
  }
</style>
