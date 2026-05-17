<script lang="ts">
  import { onMount, afterUpdate } from 'svelte';

  export let figure: any = null;
  export let title: string = '';

  let container: HTMLDivElement;
  let Plotly: any = null;

  onMount(async () => {
    Plotly = await import('plotly.js-dist-min');
    if (figure) render();
  });

  afterUpdate(() => {
    if (figure && Plotly) render();
  });

  function render() {
    if (!container || !Plotly || !figure) return;
    const config = { responsive: true, displayModeBar: false };
    Plotly.react(container, figure.data, figure.layout, config);
  }
</script>

<div class="chart-container">
  {#if title}
    <div class="chart-title">{title}</div>
  {/if}
  <div class="chart" bind:this={container}>
    {#if !figure}
      <div class="placeholder">No data</div>
    {/if}
  </div>
</div>

<style>
  .chart-container {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 0.8rem;
    overflow: hidden;
  }
  .chart-title {
    font-size: 0.75rem;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
  }
  .chart { width: 100%; min-height: 200px; }
  .placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: #475569;
    font-size: 0.85rem;
  }
</style>
