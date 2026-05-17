<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import { pipelineErrors, pipelineWarnings } from '../lib/stores';
  import ChartContainer from '../components/ChartContainer.svelte';

  let timingChart: any = null;
  let timing: any[] = [];
  let convergence: any = null;
  let health: any = null;
  let error = '';

  $: alerts = [
    ...$pipelineErrors.map(e => ({ ...e, level: 'error' })),
    ...$pipelineWarnings.map(w => ({ ...w, level: 'warning' })),
  ].sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''));

  onMount(async () => {
    try {
      const res = await api.diagnostics.timing();
      timing = res.steps || [];
    } catch {}

    try {
      timingChart = await api.diagnostics.timingChart();
    } catch {}

    try {
      convergence = await api.diagnostics.convergence();
    } catch {}

    try {
      health = await api.diagnostics.pipelineHealth();
    } catch {}
  });
</script>

<div class="diag-page">
  <h2>Diagnostics</h2>

  <div class="charts-section">
    <ChartContainer title="Step Timing" figure={timingChart} />
  </div>

  {#if timing.length > 0}
    <div class="timing-table">
      <h3>Step Durations</h3>
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Duration</th>
            <th>Start</th>
          </tr>
        </thead>
        <tbody>
          {#each timing as step}
            <tr>
              <td>{step.step}</td>
              <td>{step.duration_sec}s</td>
              <td class="ts">{step.start}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  {#if health}
    <div class="health-section">
      <h3>Pipeline Health</h3>
      <div class="health-grid">
        <div class="health-item">
          <span class="health-label">Status</span>
          <span class="health-value" class:training={health.status === 'training'} class:error={health.status === 'error'}>{health.status}</span>
        </div>
        <div class="health-item">
          <span class="health-label">Step</span>
          <span class="health-value">{health.step}</span>
        </div>
        {#if health.loss !== null}
          <div class="health-item">
            <span class="health-label">Loss</span>
            <span class="health-value">{health.loss?.toFixed(4)}</span>
          </div>
        {/if}
        {#if health.last_event_age_sec !== null}
          <div class="health-item">
            <span class="health-label">Last event</span>
            <span class="health-value">{health.last_event_age_sec}s ago</span>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  {#if alerts.length > 0}
    <div class="alerts-section">
      <h3>Alerts Timeline</h3>
      <div class="alerts-list">
        {#each alerts as alert}
          <div class="alert-item" class:error={alert.level === 'error'} class:warning={alert.level === 'warning'}>
            <span class="alert-badge">{alert.level === 'error' ? 'ERR' : 'WARN'}</span>
            <span class="alert-code">{alert.code}</span>
            <span class="alert-msg">{alert.message}</span>
            {#if alert.timestamp}
              <span class="alert-ts">{new Date(alert.timestamp).toLocaleTimeString()}</span>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  {#if convergence}
    <div class="convergence-section">
      <h3>Latest Convergence</h3>
      <pre class="code-block">{JSON.stringify(convergence, null, 2)}</pre>
    </div>
  {/if}
</div>

<style>
  .diag-page { padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
  h2 { margin: 0; font-size: 1rem; color: #e2e8f0; }
  h3 { margin: 0 0 0.5rem; font-size: 0.85rem; color: #94a3b8; }
  .charts-section { display: grid; grid-template-columns: 1fr; gap: 0.8rem; }
  .timing-table, .convergence-section {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 1rem;
  }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { text-align: left; padding: 0.4rem; color: #64748b; border-bottom: 1px solid #2a2a4a; }
  td { padding: 0.4rem; color: #cbd5e1; border-bottom: 1px solid #1e293b; }
  .ts { font-size: 0.7rem; color: #64748b; }
  .code-block {
    background: #0f0f23; border: 1px solid #2a2a4a; border-radius: 6px;
    padding: 0.8rem; font-size: 0.75rem; color: #e2e8f0; overflow-x: auto;
    max-height: 300px; overflow-y: auto;
  }
  .health-section, .alerts-section {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 1rem;
  }
  .health-grid { display: flex; gap: 1.5rem; flex-wrap: wrap; }
  .health-item { display: flex; flex-direction: column; gap: 0.2rem; }
  .health-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; }
  .health-value { font-size: 0.85rem; color: #e2e8f0; font-weight: 500; }
  .health-value.training { color: #3b82f6; }
  .health-value.error { color: #ef4444; }
  .alerts-list { display: flex; flex-direction: column; gap: 0.4rem; }
  .alert-item {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.8rem;
  }
  .alert-item.error { background: #1c0f0f; border-left: 3px solid #ef4444; }
  .alert-item.warning { background: #1c1a0f; border-left: 3px solid #f59e0b; }
  .alert-badge {
    font-size: 0.65rem; font-weight: 700; padding: 0.1rem 0.3rem;
    border-radius: 4px; text-transform: uppercase;
  }
  .alert-item.error .alert-badge { background: #7f1d1d; color: #fca5a5; }
  .alert-item.warning .alert-badge { background: #78350f; color: #fde68a; }
  .alert-code { font-weight: 600; color: #cbd5e1; }
  .alert-msg { flex: 1; color: #94a3b8; }
  .alert-ts { font-size: 0.7rem; color: #64748b; }
</style>
