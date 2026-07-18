import { writable } from 'svelte/store';

export interface QueryResult {
  score: number;
  document: string;
  metadata: Record<string, any>;
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
