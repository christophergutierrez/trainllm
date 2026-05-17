<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import KPICard from '../components/KPICard.svelte';
  import ChartContainer from '../components/ChartContainer.svelte';

  let runs: any[] = [];
  let selectedEval: string = '';
  let evalData: any = null;
  let records: any[] = [];
  let bandChart: any = null;
  let scoreChart: any = null;
  let selectedRecord: any = null;
  let filterBand = '';
  let total = 0;

  onMount(async () => {
    try {
      runs = await api.runs.list();
      if (runs.length > 0) {
        selectedEval = runs[0].id;
        await loadEval();
      }
    } catch {}
  });

  async function loadEval() {
    if (!selectedEval) return;
    evalData = null;
    bandChart = null;
    scoreChart = null;
    records = [];
    total = 0;
    try {
      evalData = await api.evals.get(selectedEval);
      const params: Record<string, string> = { limit: '50' };
      if (filterBand) params.band = filterBand;
      const res = await api.evals.records(selectedEval, params);
      records = res.records;
      total = res.total;
    } catch {}
    try { bandChart = await api.evals.bandChart(selectedEval); } catch {}
    try { scoreChart = await api.evals.scoreChart(selectedEval); } catch {}
  }

  function selectRecord(rec: any) {
    selectedRecord = selectedRecord?.id === rec.id ? null : rec;
  }
</script>

<div class="evals-page">
  <div class="controls">
    <select bind:value={selectedEval} on:change={loadEval}>
      {#each runs as run}
        <option value={run.id}>{run.timestamp} — {run.model || run.id}</option>
      {/each}
    </select>
    <select bind:value={filterBand} on:change={loadEval}>
      <option value="">All Bands</option>
      <option value="EXCELLENT">Excellent</option>
      <option value="GOOD">Good</option>
      <option value="PARTIAL">Partial</option>
      <option value="POOR">Poor</option>
    </select>
  </div>

  {#if evalData}
    <div class="kpi-row">
      <KPICard label="Composite" value={(evalData.summary?.avg_composite_score ?? evalData.summary?.avg_score ?? 0).toFixed(3)} />
      <KPICard label="Structural" value={(evalData.summary?.avg_structural_score ?? 0).toFixed(3)} />
      <KPICard label="Similarity" value={(evalData.summary?.avg_score ?? 0).toFixed(3)} />
      <KPICard label="Records" value={total} />
    </div>

    <div class="charts-row">
      <ChartContainer title="Band Distribution" figure={bandChart} />
      <ChartContainer title="Score Distribution" figure={scoreChart} />
    </div>
  {/if}

  <div class="records-table">
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Band</th>
          <th>Composite</th>
          <th>Structural</th>
          <th>Similarity</th>
        </tr>
      </thead>
      <tbody>
        {#each records as rec}
          <tr class="record-row" class:selected={selectedRecord?.id === rec.id} on:click={() => selectRecord(rec)}>
            <td>{rec.id}</td>
            <td><span class="band-badge {rec.band.toLowerCase()}">{rec.band}</span></td>
            <td>{(rec.composite_score ?? rec.score ?? 0).toFixed(3)}</td>
            <td>{(rec.structural_score ?? 0).toFixed(3)}</td>
            <td>{(rec.score ?? 0).toFixed(3)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  {#if selectedRecord}
    <div class="detail-panel">
      <h3>Record: {selectedRecord.id}</h3>
      <div class="detail-grid">
        <div class="detail-col">
          <div class="detail-label">Expected</div>
          <pre class="code-block">{selectedRecord.expected || '—'}</pre>
        </div>
        <div class="detail-col">
          <div class="detail-label">Generated</div>
          <pre class="code-block">{selectedRecord.generated || '—'}</pre>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .evals-page { padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
  .controls { display: flex; gap: 0.8rem; }
  .controls select {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 6px;
    padding: 0.4rem 0.8rem; color: #e2e8f0; font-size: 0.8rem;
  }
  .kpi-row { display: flex; gap: 0.8rem; flex-wrap: wrap; }
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
  .records-table { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { text-align: left; padding: 0.5rem; color: #64748b; border-bottom: 1px solid #2a2a4a; }
  td { padding: 0.5rem; color: #cbd5e1; border-bottom: 1px solid #1e293b; }
  .record-row { cursor: pointer; }
  .record-row:hover { background: #1e293b; }
  .record-row.selected { background: #1e3a5f; }
  .band-badge {
    padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;
  }
  .band-badge.excellent { background: #064e3b; color: #6ee7b7; }
  .band-badge.good { background: #1e3a5f; color: #93c5fd; }
  .band-badge.partial { background: #451a03; color: #fcd34d; }
  .band-badge.poor { background: #450a0a; color: #fca5a5; }
  .detail-panel {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 1rem;
  }
  .detail-panel h3 { margin: 0 0 0.8rem; font-size: 0.85rem; color: #94a3b8; }
  .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .detail-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; margin-bottom: 0.3rem; }
  .code-block {
    background: #0f0f23; border: 1px solid #2a2a4a; border-radius: 6px;
    padding: 0.6rem; font-size: 0.75rem; color: #e2e8f0; overflow-x: auto;
    white-space: pre-wrap; max-height: 300px; overflow-y: auto;
  }
</style>
