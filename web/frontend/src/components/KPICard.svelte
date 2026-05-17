<script lang="ts">
  export let label: string;
  export let value: string | number;
  export let trend: 'up' | 'down' | 'flat' | null = null;
  export let unit: string = '';
  export let hint: string = '';
</script>

<div class="kpi-card">
  <div class="kpi-label">
    {label}
    {#if hint}
      <span class="hint-wrap">
        <span class="hint-icon">?</span>
        <span class="hint-tooltip">{hint}</span>
      </span>
    {/if}
  </div>
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
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }
  .hint-wrap {
    position: relative;
    display: inline-flex;
  }
  .hint-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: #2a2a4a;
    color: #64748b;
    font-size: 0.55rem;
    font-weight: 700;
    cursor: help;
    flex-shrink: 0;
  }
  .hint-wrap:hover .hint-icon {
    background: #3b82f6;
    color: white;
  }
  .hint-tooltip {
    display: none;
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    background: #1e293b;
    border: 1px solid #3b82f6;
    color: #e2e8f0;
    font-size: 0.68rem;
    font-weight: 400;
    text-transform: none;
    padding: 0.4rem 0.6rem;
    border-radius: 5px;
    white-space: normal;
    width: max-content;
    max-width: 220px;
    line-height: 1.4;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  .hint-tooltip::after {
    content: '';
    position: absolute;
    bottom: 100%;
    left: 6px;
    border: 5px solid transparent;
    border-bottom-color: #3b82f6;
  }
  .hint-wrap:hover .hint-tooltip {
    display: block;
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
