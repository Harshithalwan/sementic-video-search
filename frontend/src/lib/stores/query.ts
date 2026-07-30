import { writable } from 'svelte/store';
import type { CaptionMetadata } from '$lib/api';

export interface QueryResult {
  score: number;
  document: string;
  metadata: CaptionMetadata;
}

function createQueryStore() {
  const results = writable<QueryResult[]>([]);
  const loading = writable(false);
  const error = writable('');

  return {
    results,
    loading,
    error
  };
}

export const query = createQueryStore();
