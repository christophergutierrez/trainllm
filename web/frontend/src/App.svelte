<script lang="ts">
  import { onMount } from 'svelte';
  import { activeTab, initStores } from './lib/stores';
  import Header from './components/Header.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import AgentPanel from './components/AgentPanel.svelte';
  import Training from './routes/Training.svelte';
  import Evals from './routes/Evals.svelte';
  import Runs from './routes/Runs.svelte';
  import Diagnostics from './routes/Diagnostics.svelte';

  let showAgent = false;

  onMount(() => {
    initStores();
  });
</script>

<div class="app">
  <Header onToggleAgent={() => showAgent = !showAgent} />
  <div class="body">
    <Sidebar />
    <main class="content">
      {#if $activeTab === 'training'}
        <Training />
      {:else if $activeTab === 'evals'}
        <Evals />
      {:else if $activeTab === 'runs'}
        <Runs />
      {:else if $activeTab === 'diagnostics'}
        <Diagnostics />
      {/if}
    </main>
    <AgentPanel visible={showAgent} />
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f0f23;
    color: #e2e8f0;
  }
  :global(*) { box-sizing: border-box; }
  .app { display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  .body { display: flex; flex: 1; overflow: hidden; }
  .content { flex: 1; overflow-y: auto; }
</style>
