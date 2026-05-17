<script lang="ts">
  import { activeTab } from '../lib/stores';

  const tabs = [
    { id: 'training', label: 'Training', icon: '⚡' },
    { id: 'evals', label: 'Evals', icon: '📊' },
    { id: 'runs', label: 'Runs', icon: '📋' },
    { id: 'diagnostics', label: 'Diag', icon: '🔧' },
  ];
</script>

<nav class="sidebar">
  {#each tabs as tab}
    <button
      class="nav-item"
      class:active={$activeTab === tab.id}
      on:click={() => activeTab.set(tab.id)}
    >
      <span class="icon">{tab.icon}</span>
      <span class="label">{tab.label}</span>
    </button>
  {/each}
</nav>

<style>
  .sidebar {
    display: flex;
    flex-direction: column;
    width: 72px;
    background: #0f0f23;
    border-right: 1px solid #2a2a4a;
    padding: 0.5rem 0;
  }
  .nav-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
    padding: 0.6rem 0;
    border: none;
    background: none;
    color: #64748b;
    cursor: pointer;
    font-size: 0.65rem;
    transition: all 0.15s;
  }
  .nav-item:hover { color: #94a3b8; background: #1a1a2e; }
  .nav-item.active { color: #e2e8f0; background: #1a1a2e; border-right: 2px solid #3b82f6; }
  .icon { font-size: 1.2rem; }
  .label { font-weight: 500; }
</style>
