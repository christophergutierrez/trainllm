<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import ChartContainer from '../components/ChartContainer.svelte';

  let timingChart: any = null;
  let timing: any[] = [];
  let convergence: any = null;
  let error = '';

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
</style>
