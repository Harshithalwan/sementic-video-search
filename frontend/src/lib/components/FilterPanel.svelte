<script lang="ts">
  interface Props {
    videoNameFilter: string;
    videoIdFilter: string;
    tsFrom: number;
    tsTo: number;
    timeFrom: string;
    timeTo: string;
    videoNames: string[];
    disabled?: boolean;
  }

  let {
    videoNameFilter = $bindable(),
    videoIdFilter = $bindable(),
    tsFrom = $bindable(),
    tsTo = $bindable(),
    timeFrom = $bindable(),
    timeTo = $bindable(),
    videoNames,
    disabled = false
  }: Props = $props();
</script>

<details>
  <summary>Advanced Filters</summary>
  <div class="content">
    <div class="grid-2">
      <div class="field">
        <label for="video-name-filter">Filter by Video Name</label>
        <select id="video-name-filter" bind:value={videoNameFilter} {disabled}>
          <option value="">All</option>
          {#each videoNames as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </div>
      <div class="field">
        <label for="video-id-filter">Video ID</label>
        <input
          id="video-id-filter"
          type="text"
          bind:value={videoIdFilter}
          placeholder="Exact match"
          {disabled}
        />
      </div>
    </div>

    <div class="grid-2">
      <div class="field">
        <label for="ts-from">Video Timestamp From (ms)</label>
        <input id="ts-from" type="number" min="0" step="100" bind:value={tsFrom} {disabled} />
      </div>
      <div class="field">
        <label for="ts-to">Video Timestamp To (ms)</label>
        <input id="ts-to" type="number" min="0" step="100" bind:value={tsTo} {disabled} />
      </div>
    </div>

    <div class="grid-2">
      <div class="field">
        <label for="time-from">Wall-Clock Time From</label>
        <input id="time-from" type="time" bind:value={timeFrom} {disabled} />
      </div>
      <div class="field">
        <label for="time-to">Wall-Clock Time To</label>
        <input id="time-to" type="time" bind:value={timeTo} {disabled} />
      </div>
    </div>
  </div>
</details>

<style>
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }
  .grid-2:last-child {
    margin-bottom: 0;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  @media (max-width: 600px) {
    .grid-2 {
      grid-template-columns: 1fr;
    }
  }
</style>
