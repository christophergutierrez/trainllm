<script lang="ts">
  export let label: string;
  export let value: string | number;
  export let trend: 'up' | 'down' | 'flat' | null = null;
  export let unit: string = '';
</script>

<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">
    {value}{unit}
    {#if trend === 'up'}
      <span class="trend up">↑</span>
    {:else if trend === 'down'}
      <span class="trend down">↓</span>
    {/if}
  </div>
</div>

<style>
  .kpi-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    min-width: 120px;
  }
  .kpi-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 500;
  }
  .kpi-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 0.2rem;
  }
  .trend { font-size: 0.9rem; margin-left: 0.3rem; }
  .trend.up { color: #10b981; }
  .trend.down { color: #ef4444; }
</style>
