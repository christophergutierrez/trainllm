<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import KPICard from '../components/KPICard.svelte';
  import ChartContainer from '../components/ChartContainer.svelte';

  let runs: any[] = [];
  let trendChart: any = null;
  let selectedRun: any = null;

  onMount(async () => {
    try {
      runs = await api.runs.list();
    } catch {}
  });

  async function selectRun(run: any) {
    selectedRun = selectedRun?.id === run.id ? null : run;
  }
</script>

<div class="runs-page">
  <div class="master-pane">
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Model</th>
          <th>Composite</th>
          <th>Similarity</th>
          <th>Records</th>
        </tr>
      </thead>
      <tbody>
        {#each runs as run}
          <tr
            class="run-row"
            class:selected={selectedRun?.id === run.id}
            on:click={() => selectRun(run)}
          >
            <td>{run.timestamp}</td>
            <td class="model-name">{run.model || '—'}</td>
            <td>{(run.avg_composite_score ?? run.avg_score ?? 0).toFixed(3)}</td>
            <td>{(run.avg_score ?? 0).toFixed(3)}</td>
            <td>{run.num_records}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <div class="detail-pane">
    {#if selectedRun}
      <h3>Run: {selectedRun.id}</h3>
      <div class="kpi-row">
        <KPICard label="Composite" value={(selectedRun.avg_composite_score ?? selectedRun.avg_score).toFixed(3)} />
        <KPICard label="Records" value={selectedRun.num_records} />
      </div>
      {#if selectedRun.band_counts}
        <div class="band-summary">
          {#each Object.entries(selectedRun.band_counts) as [band, count]}
            <span class="band-chip {band.toLowerCase()}">{band}: {count}</span>
          {/each}
        </div>
      {/if}
    {:else}
      <div class="detail-empty">Select a run to view details</div>
    {/if}
  </div>
</div>

<style>
  .runs-page {
    padding: 1rem;
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 1rem;
    height: 100%;
    overflow: hidden;
  }
  .master-pane {
    overflow-y: auto;
    min-height: 0;
  }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { text-align: left; padding: 0.5rem; color: #64748b; border-bottom: 1px solid #2a2a4a; position: sticky; top: 0; background: #0f0f23; }
  td { padding: 0.5rem; color: #cbd5e1; border-bottom: 1px solid #1e293b; }
  .run-row { cursor: pointer; }
  .run-row:hover { background: #1e293b; }
  .run-row.selected { background: #1e3a5f; }
  .model-name { font-size: 0.75rem; color: #94a3b8; }
  .detail-pane {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 1rem;
    overflow-y: auto;
    min-height: 0;
    position: sticky;
    top: 1rem;
    align-self: start;
  }
  .detail-pane h3 { margin: 0 0 0.8rem; font-size: 0.85rem; color: #94a3b8; }
  .detail-empty { color: #475569; font-size: 0.8rem; text-align: center; padding: 2rem 0; }
  .kpi-row { display: flex; gap: 0.8rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
  .band-summary { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .band-chip {
    padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;
  }
  .band-chip.excellent { background: #064e3b; color: #6ee7b7; }
  .band-chip.good { background: #1e3a5f; color: #93c5fd; }
  .band-chip.partial { background: #451a03; color: #fcd34d; }
  .band-chip.poor { background: #450a0a; color: #fca5a5; }
  @media (max-width: 768px) {
    .runs-page { grid-template-columns: 1fr; }
    .detail-pane { position: static; }
  }
</style>
