<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import KPICard from '../components/KPICard.svelte';

  let bundles: any[] = [];
  let selectedManifest: any = null;
  let error = '';

  onMount(async () => {
    try {
      const data = await api.bundles.list();
      bundles = (data.bundles || []).reverse();
    } catch (e: any) {
      error = e.message;
    }
  });

  async function showManifest(name: string) {
    if (selectedManifest?.bundle_name === name) {
      selectedManifest = null;
      return;
    }
    try {
      selectedManifest = await api.bundles.manifest(name);
    } catch {}
  }

  function download(name: string) {
    window.open(api.bundles.downloadUrl(name), '_blank');
  }
</script>

<div class="bundles-page">
  <div class="page-header">
    <h2>Bundles</h2>
    <span class="subtitle">Versioned adapter artifacts ready for deployment</span>
  </div>

  {#if error}
    <div class="error">{error}</div>
  {/if}

  {#if bundles.length === 0 && !error}
    <div class="empty">No bundles yet. Run <code>python bundle.py</code> to create one.</div>
  {/if}

  <div class="kpi-row">
    <KPICard label="Total Bundles" value={bundles.length} />
    <KPICard label="Latest Version" value={bundles.length > 0 ? `v${bundles[0].version}` : '—'} />
    <KPICard label="Best Score" value={bundles.length > 0 ? Math.max(...bundles.map((b: any) => b.score || 0)).toFixed(4) : '—'} />
  </div>

  <div class="bundles-list">
    <table>
      <thead>
        <tr>
          <th>Bundle</th>
          <th>Base Model</th>
          <th>Score</th>
          <th>Size</th>
          <th>Created</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {#each bundles as b}
          <tr class:selected={selectedManifest?.bundle_name === b.bundle_name}>
            <td class="bundle-name">{b.bundle_name}</td>
            <td class="model">{b.base_model}</td>
            <td>
              <span class="score" class:excellent={b.score >= 0.95} class:good={b.score >= 0.8 && b.score < 0.95}>
                {b.score?.toFixed(4) ?? '—'}
              </span>
            </td>
            <td>{b.size_mb} MB</td>
            <td class="ts">{b.created?.slice(0, 10)}</td>
            <td class="actions">
              <button class="btn-sm" on:click={() => showManifest(b.bundle_name)}>Manifest</button>
              <button class="btn-sm btn-primary" on:click={() => download(b.bundle_name)}>Download</button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  {#if selectedManifest}
    <div class="manifest-panel">
      <h3>Manifest: {selectedManifest.bundle_name}</h3>
      <div class="manifest-grid">
        <div class="manifest-section">
          <div class="section-title">Provenance</div>
          <dl>
            <dt>Base Model</dt><dd>{selectedManifest.base_model}</dd>
            <dt>Git SHA</dt><dd class="mono">{selectedManifest.git_sha?.slice(0, 12)}</dd>
            <dt>Config Hash</dt><dd class="mono">{selectedManifest.config_hash}</dd>
            <dt>Dirty</dt><dd>{selectedManifest.git_dirty ? 'Yes' : 'No'}</dd>
          </dl>
        </div>
        <div class="manifest-section">
          <div class="section-title">LoRA Config</div>
          <dl>
            <dt>Rank</dt><dd>{selectedManifest.lora?.rank}</dd>
            <dt>Alpha</dt><dd>{selectedManifest.lora?.alpha}</dd>
            <dt>Modules</dt><dd>{selectedManifest.lora?.target_modules?.join(', ')}</dd>
          </dl>
        </div>
        <div class="manifest-section">
          <div class="section-title">Eval</div>
          <dl>
            <dt>ID</dt><dd class="mono">{selectedManifest.eval?.eval_id}</dd>
            <dt>Score</dt><dd>{selectedManifest.eval?.score}</dd>
            <dt>Records</dt><dd>{selectedManifest.eval?.num_records}</dd>
          </dl>
        </div>
      </div>
      <details>
        <summary>Raw JSON</summary>
        <pre class="code-block">{JSON.stringify(selectedManifest, null, 2)}</pre>
      </details>
    </div>
  {/if}
</div>

<style>
  .bundles-page { padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
  .page-header { display: flex; align-items: baseline; gap: 0.8rem; }
  h2 { margin: 0; font-size: 1rem; color: #e2e8f0; }
  .subtitle { font-size: 0.75rem; color: #64748b; }
  .kpi-row { display: flex; gap: 0.8rem; flex-wrap: wrap; }
  .error { color: #ef4444; font-size: 0.8rem; padding: 0.5rem; background: #450a0a; border-radius: 6px; }
  .empty { color: #64748b; font-size: 0.85rem; padding: 2rem; text-align: center; }
  .empty code { background: #1a1a2e; padding: 0.2rem 0.4rem; border-radius: 4px; color: #e2e8f0; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { text-align: left; padding: 0.5rem; color: #64748b; border-bottom: 1px solid #2a2a4a; }
  td { padding: 0.5rem; color: #cbd5e1; border-bottom: 1px solid #1e293b; }
  tr.selected { background: #1e3a5f; }
  .bundle-name { font-weight: 600; color: #e2e8f0; }
  .model { font-size: 0.75rem; color: #94a3b8; }
  .ts { font-size: 0.7rem; color: #64748b; }
  .score { font-weight: 600; }
  .score.excellent { color: #6ee7b7; }
  .score.good { color: #93c5fd; }
  .actions { display: flex; gap: 0.3rem; }
  .btn-sm {
    padding: 0.2rem 0.5rem; font-size: 0.7rem; border-radius: 4px;
    border: 1px solid #2a2a4a; background: #1a1a2e; color: #94a3b8; cursor: pointer;
  }
  .btn-sm:hover { background: #2a2a4a; color: #e2e8f0; }
  .btn-primary { background: #1e3a5f; border-color: #3b82f6; color: #93c5fd; }
  .btn-primary:hover { background: #2563eb; color: white; }
  .manifest-panel {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 1rem;
  }
  .manifest-panel h3 { margin: 0 0 0.8rem; font-size: 0.85rem; color: #94a3b8; }
  .manifest-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 0.8rem; }
  .manifest-section { }
  .section-title { font-size: 0.7rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 0.4rem; }
  dl { margin: 0; font-size: 0.78rem; }
  dt { color: #64748b; display: inline; }
  dt::after { content: ': '; }
  dd { color: #e2e8f0; display: inline; margin: 0; }
  dd::after { content: '\A'; white-space: pre; }
  .mono { font-family: monospace; font-size: 0.72rem; }
  details { margin-top: 0.5rem; }
  summary { font-size: 0.75rem; color: #64748b; cursor: pointer; }
  .code-block {
    background: #0f0f23; border: 1px solid #2a2a4a; border-radius: 6px;
    padding: 0.6rem; font-size: 0.7rem; color: #e2e8f0; overflow-x: auto;
    max-height: 300px; overflow-y: auto; margin-top: 0.5rem;
  }
</style>
