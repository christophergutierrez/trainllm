<script lang="ts">
  export let steps: Array<{ name: string; status: 'pending' | 'active' | 'done' | 'error'; duration?: number }> = [];

  function formatDuration(sec: number | undefined): string {
    if (!sec) return '';
    if (sec < 60) return `${sec}s`;
    return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  }
</script>

<div class="pipeline">
  {#each steps as step, i}
    <div class="step" class:active={step.status === 'active'} class:done={step.status === 'done'} class:error={step.status === 'error'}>
      <div class="step-indicator">
        {#if step.status === 'done'}
          <span class="check">✓</span>
        {:else if step.status === 'error'}
          <span class="x">✕</span>
        {:else if step.status === 'active'}
          <span class="pulse"></span>
        {:else}
          <span class="dot"></span>
        {/if}
      </div>
      <div class="step-info">
        <span class="step-name">{step.name}</span>
        {#if step.duration}
          <span class="step-duration">{formatDuration(step.duration)}</span>
        {/if}
      </div>
    </div>
    {#if i < steps.length - 1}
      <div class="connector" class:done={step.status === 'done'}></div>
    {/if}
  {/each}
</div>

<style>
  .pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 0.8rem 1rem;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    overflow-x: auto;
  }
  .step {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.5rem;
    border-radius: 6px;
    white-space: nowrap;
  }
  .step.active { background: #1e3a5f; }
  .step.done { opacity: 0.7; }
  .step.error { background: #450a0a; }
  .step-indicator { width: 16px; height: 16px; display: flex; align-items: center; justify-content: center; }
  .check { color: #10b981; font-size: 0.8rem; font-weight: bold; }
  .x { color: #ef4444; font-size: 0.8rem; font-weight: bold; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #475569; }
  .pulse {
    width: 8px; height: 8px; border-radius: 50%; background: #3b82f6;
    animation: pulse-anim 1.5s infinite;
  }
  @keyframes pulse-anim {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.4); opacity: 0.6; }
  }
  .step-name { font-size: 0.75rem; color: #e2e8f0; font-weight: 500; }
  .step-duration { font-size: 0.65rem; color: #64748b; }
  .connector {
    width: 24px; height: 2px; background: #334155; flex-shrink: 0;
  }
  .connector.done { background: #10b981; }
</style>
