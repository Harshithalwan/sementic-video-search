<script lang="ts">
  import { onMount } from 'svelte';
  import { searchCaptions, getModels, getCollectionVideos } from '$lib/api';
  import type { ModelInfo, QueryResult } from '$lib/api';
  import { query } from '$lib/stores/query';

  import SearchForm from '$lib/components/SearchForm.svelte';
  import FilterPanel from '$lib/components/FilterPanel.svelte';
  import ResultCard from '$lib/components/ResultCard.svelte';
  import VideoPlayer from '$lib/components/VideoPlayer.svelte';

  let models = $state<ModelInfo[]>([]);
  let collections = $state<Record<string, string>>({});
  let videoNames = $state<string[]>([]);

  // Search fields
  let searchQuery = $state('');
  let topK = $state(5);
  let collection = $state('captions_lfm2.5');

  // Filter fields
  let videoNameFilter = $state('');
  let videoIdFilter = $state('');
  let tsFrom = $state(0);
  let tsTo = $state(0);
  let timeFrom = $state('');
  let timeTo = $state('');

  // Results
  let results = $state<QueryResult[]>([]);
  let loading = $state(false);
  let error = $state('');
  let resultCount = $state(0);

  // Video player modal
  let playerShow = $state(false);
  let playerSource = $state('');
  let playerTimestampMs = $state(0);

  function handleViewVideo(source: string, timestampMs: number) {
    playerSource = source;
    playerTimestampMs = timestampMs;
    playerShow = true;
  }

  const unsubResults = query.results.subscribe((v) => (results = v));
  const unsubLoading = query.loading.subscribe((v) => (loading = v));
  const unsubError = query.error.subscribe((v) => (error = v));

  onMount(async () => {
    try {
      const res = await getModels();
      models = res.models;
      collections = res.collections;
      collection = Object.values(collections)[0] || 'captions_lfm2.5';
      await loadVideoNames();
    } catch (e) {
      console.error('Failed to load models:', e);
    }
  });

  async function loadVideoNames() {
    try {
      const colName = collection || Object.values(collections)[0];
      if (!colName) return;
      const res = await getCollectionVideos(colName);
      videoNames = res.video_names;
    } catch {
      videoNames = [];
    }
  }

  async function handleSearch() {
    if (!searchQuery.trim()) {
      error = 'Please enter a search query.';
      return;
    }

    loading = true;
    error = '';
    query.loading.set(true);
    query.error.set('');

    try {
      const params: any = {
        query: searchQuery,
        top_k: topK,
        collection
      };

      if (videoNameFilter) params.video_name = videoNameFilter;
      if (videoIdFilter.trim()) params.video_id = videoIdFilter.trim();
      if (tsFrom > 0) params.ts_from = tsFrom;
      if (tsTo > 0) params.ts_to = tsTo;
      if (timeFrom) {
        const [h, m, s] = timeFrom.split(':').map(Number);
        params.time_from = h * 3600 + m * 60 + s;
      }
      if (timeTo) {
        const [h, m, s] = timeTo.split(':').map(Number);
        params.time_to = h * 3600 + m * 60 + s;
      }

      const res = await searchCaptions(params);

      if (res.error) {
        error = res.error;
        query.error.set(res.error);
        results = [];
        query.results.set([]);
      } else {
        results = res.results;
        resultCount = res.results.length;
        query.results.set(res.results);
      }
    } catch (e: any) {
      error = e.message || 'Search failed';
      query.error.set(error);
      results = [];
      query.results.set([]);
    } finally {
      loading = false;
      query.loading.set(false);
    }
  }

  $effect(() => {
    collection;
    loadVideoNames();
  });
</script>

<svelte:head>
  <title>Query Captions — Semantic Video Search</title>
</svelte:head>

<h1>Query Captions</h1>

<SearchForm
  bind:query={searchQuery}
  bind:topK
  bind:collection
  {collections}
  {loading}
  onsearch={handleSearch}
/>

<div style="margin-top: 0.85rem;">
  <FilterPanel
    bind:videoNameFilter
    bind:videoIdFilter
    bind:tsFrom
    bind:tsTo
    bind:timeFrom
    bind:timeTo
    {videoNames}
  />
</div>

<hr />

{#if error}
  <div class="error-banner">{error}</div>
{/if}

{#if results.length > 0}
  <div class="result-count text-success">
    Found {resultCount} result(s)
  </div>
  <div class="results-list">
    {#each results as r, i}
      <ResultCard result={r} index={i + 1} onview={handleViewVideo} />
    {/each}
  </div>
{:else if !loading && !error && results.length === 0 && searchQuery}
  <div class="empty-state text-muted">No matching captions found.</div>
{/if}

<VideoPlayer
  videoSource={playerSource}
  timestampMs={playerTimestampMs}
  show={playerShow}
  onclose={() => (playerShow = false)}
/>

<style>
  h1 {
    font-size: 1.4rem;
    margin-bottom: 1rem;
  }
  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
  }
  .error-banner {
    padding: 0.6rem 1rem;
    background: rgba(224, 85, 85, 0.12);
    border: 1px solid var(--error);
    border-radius: var(--radius);
    color: var(--error);
    margin-bottom: 0.5rem;
  }
  .result-count {
    margin-bottom: 0.75rem;
    font-weight: 500;
  }
  .results-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .empty-state {
    text-align: center;
    padding: 2rem;
    font-style: italic;
  }
</style>
