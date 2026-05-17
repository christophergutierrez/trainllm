<script lang="ts">
  import { latestLoss, latestStep, isTraining, agentStatus, pipelineErrors, pipelineWarnings } from '../lib/stores';
  import StatusPill from './StatusPill.svelte';

  export let onToggleAgent: () => void;

  $: healthColor = $pipelineErrors.length > 0 ? 'red' : $pipelineWarnings.length > 0 ? 'amber' : 'green';
</script>

<header class="header">
  <div class="header-left">
    <h1 class="title">trainLLM</h1>
    <span class="health-dot" class:green={healthColor === 'green'} class:amber={healthColor === 'amber'} class:red={healthColor === 'red'}
          title={healthColor === 'red' ? 'Pipeline error' : healthColor === 'amber' ? 'Pipeline warning' : 'Healthy'}></span>
    <StatusPill />
  </div>

  <div class="header-center">
    {#if $isTraining}
      <span class="metric">Step <strong>{$latestStep}</strong></span>
      {#if $latestLoss !== null}
        <span class="metric">Loss <strong>{$latestLoss.toFixed(4)}</strong></span>
      {/if}
    {:else}
      <span class="metric idle">Idle</span>
    {/if}
  </div>

  <div class="header-right">
    <button class="agent-toggle" on:click={onToggleAgent} title="Toggle Agent Panel">
      <span class="agent-icon">AI</span>
      <span class="agent-dot" class:connected={$agentStatus === 'connected'}></span>
    </button>
  </div>
</header>

<style>
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1rem;
    background: #1a1a2e;
    border-bottom: 1px solid #2a2a4a;
    height: 48px;
  }
  .header-left { display: flex; align-items: center; gap: 0.6rem; }
  .health-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #64748b;
    flex-shrink: 0;
  }
  .health-dot.green { background: #10b981; }
  .health-dot.amber { background: #f59e0b; }
  .health-dot.red { background: #ef4444; animation: pulse-red 1.5s infinite; }
  @keyframes pulse-red {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  .header-center { display: flex; gap: 1.5rem; }
  .header-right { display: flex; align-items: center; }
  .title { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin: 0; }
  .metric { color: #94a3b8; font-size: 0.85rem; }
  .metric strong { color: #e2e8f0; }
  .metric.idle { color: #64748b; }
  .agent-toggle {
    background: #2a2a4a;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    color: #e2e8f0;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .agent-toggle:hover { background: #3a3a5a; }
  .agent-icon { font-size: 0.8rem; font-weight: 600; }
  .agent-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #64748b;
  }
  .agent-dot.connected { background: #10b981; }
</style>
