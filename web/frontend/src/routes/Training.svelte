<script lang="ts">
  import { onMount } from 'svelte';
  import { latestLoss, latestStep, latestLR, trainingEvents, isTraining } from '../lib/stores';
  import { api } from '../lib/api';
  import KPICard from '../components/KPICard.svelte';
  import ChartContainer from '../components/ChartContainer.svelte';

  let lossChart: any = null;
  let convergence: any = null;
  let error = '';

  onMount(async () => {
    try {
      convergence = await api.diagnostics.convergence();
    } catch (e: any) {
      // No convergence data yet — that's fine
    }
  });

  $: stepsPerSec = $trainingEvents.length > 1
    ? (($trainingEvents[$trainingEvents.length - 1]?.step - $trainingEvents[0]?.step) /
       Math.max(1, $trainingEvents.length - 1)).toFixed(1)
    : '—';
</script>

<div class="training-page">
  <div class="kpi-row">
    <KPICard label="Loss" value={$latestLoss?.toFixed(4) ?? '—'} trend={$isTraining ? 'down' : null} />
    <KPICard label="Step" value={$latestStep || '—'} />
    <KPICard label="LR" value={$latestLR ? $latestLR.toExponential(1) : '—'} />
    <KPICard label="Steps/s" value={stepsPerSec} />
  </div>

  <div class="charts-row">
    <ChartContainer title="Loss Curve" figure={lossChart} />
  </div>

  {#if convergence}
    <div class="convergence-summary">
      <h3>Convergence</h3>
      <div class="stats-grid">
        <div class="stat">
          <span class="stat-label">First Loss</span>
          <span class="stat-value">{convergence.first_loss}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Final Loss</span>
          <span class="stat-value">{convergence.final_loss}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Best Loss</span>
          <span class="stat-value">{convergence.best_loss}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Total Steps</span>
          <span class="stat-value">{convergence.total_steps}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Rate</span>
          <span class="stat-value">{convergence.convergence_rate}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Early Stop</span>
          <span class="stat-value">{convergence.stopped_early ? 'Yes' : 'No'}</span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .training-page { padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
  .kpi-row { display: flex; gap: 0.8rem; flex-wrap: wrap; }
  .charts-row { display: grid; grid-template-columns: 1fr; gap: 0.8rem; }
  .convergence-summary {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 1rem;
  }
  .convergence-summary h3 { margin: 0 0 0.8rem; font-size: 0.85rem; color: #94a3b8; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.8rem; }
  .stat-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; }
  .stat-value { display: block; font-size: 1rem; font-weight: 600; color: #e2e8f0; }
</style>
