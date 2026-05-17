<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import KPICard from '../components/KPICard.svelte';
  import ChartContainer from '../components/ChartContainer.svelte';

  let models: any[] = [];
  let selected: Set<string> = new Set();
  let error = '';
  let detailModel: any = null;
  let detailLoading = false;

  // Comparison state
  let comparing = false;
  let comparisonData: any = null;
  let comparisonLoading = false;

  onMount(async () => {
    try {
      const data = await api.models.list();
      models = (data.models || []).reverse();
    } catch (e: any) {
      error = e.message;
    }
  });

  function toggleSelect(name: string) {
    if (selected.has(name)) {
      selected.delete(name);
    } else {
      selected.add(name);
    }
    selected = new Set(selected);
  }

  function selectAll() {
    if (selected.size === models.length) {
      selected = new Set();
    } else {
      selected = new Set(models.map((m: any) => m.model_name || m.bundle_name));
    }
  }

  async function showDetail(name: string) {
    if (detailModel?.model_name === name || detailModel?.bundle_name === name) {
      detailModel = null;
      return;
    }
    detailLoading = true;
    try {
      detailModel = await api.models.diagnostics(name);
    } catch {
      detailModel = null;
    }
    detailLoading = false;
  }

  async function startCompare() {
    comparing = true;
    comparisonLoading = true;
    comparisonData = null;
    try {
      comparisonData = await api.models.compare([...selected]);
    } catch (e: any) {
      error = e.message;
    }
    comparisonLoading = false;
  }

  function backToList() {
    comparing = false;
    comparisonData = null;
  }

  function download(name: string) {
    window.open(api.models.downloadUrl(name), '_blank');
  }

  function modelName(m: any): string {
    return m.model_name || m.bundle_name || '';
  }

  function bestScore(): string {
    if (!models.length) return '—';
    return Math.max(...models.map((m: any) => m.score || 0)).toFixed(4);
  }

  function deltaClass(a: number | null, b: number | null): string {
    if (a == null || b == null) return '';
    if (a > b) return 'better';
    if (a < b) return 'worse';
    return '';
  }

  function fmt(v: any, digits = 4): string {
    if (v == null) return '—';
    if (typeof v === 'number') return v.toFixed(digits);
    return String(v);
  }
</script>

{#if comparing}
  <!-- ========== COMPARISON VIEW ========== -->
  <div class="models-page">
    <div class="compare-header">
      <button class="btn-back" on:click={backToList}>← Back to list</button>
      <h2>Comparing {selected.size} Models</h2>
    </div>

    {#if comparisonLoading}
      <div class="loading">Loading comparison data...</div>
    {:else if comparisonData}
      <div class="compare-kpis">
        {#each comparisonData.models as m, i}
          <div class="model-kpi" style="border-left: 3px solid {['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#ec4899'][i % 6]}">
            <div class="kpi-name">{m.model_name}</div>
            <div class="kpi-score">{fmt(m.score)}</div>
            <div class="kpi-meta">{m.base_model} &middot; v{m.version}</div>
          </div>
        {/each}
      </div>

      <div class="chart-grid">
        {#if comparisonData.charts.loss_overlay}
          <ChartContainer figure={comparisonData.charts.loss_overlay} title="Loss Curves" />
        {/if}
        {#if comparisonData.charts.score_comparison}
          <ChartContainer figure={comparisonData.charts.score_comparison} title="Composite Score" />
        {/if}
        {#if comparisonData.charts.band_comparison}
          <ChartContainer figure={comparisonData.charts.band_comparison} title="Band Distribution" />
        {/if}
        {#if comparisonData.charts.convention_comparison}
          <ChartContainer figure={comparisonData.charts.convention_comparison} title="Convention Scores" />
        {/if}
      </div>

      <!-- Metrics table -->
      <div class="metrics-panel">
        <div class="section-title">Metrics Comparison</div>
        <div class="metrics-table-wrap">
          <table class="metrics-table">
            <thead>
              <tr>
                <th>Metric</th>
                {#each comparisonData.models as m}
                  <th>{m.model_name}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="metric-label">Composite Score</td>
                {#each comparisonData.models as m, i}
                  <td class={i > 0 ? deltaClass(m.score, comparisonData.models[0].score) : ''}>{fmt(m.score)}</td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">Eval Records</td>
                {#each comparisonData.models as m}
                  <td>{m.num_records || '—'}</td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">First Loss</td>
                {#each comparisonData.models as m}
                  <td>{fmt(m.convergence?.first_loss, 4)}</td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">Best Loss</td>
                {#each comparisonData.models as m, i}
                  <td class={i > 0 ? deltaClass(comparisonData.models[0].convergence?.best_loss, m.convergence?.best_loss) : ''}>
                    {fmt(m.convergence?.best_loss, 5)}
                  </td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">Final Loss</td>
                {#each comparisonData.models as m, i}
                  <td class={i > 0 ? deltaClass(comparisonData.models[0].convergence?.final_loss, m.convergence?.final_loss) : ''}>
                    {fmt(m.convergence?.final_loss, 5)}
                  </td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">Total Steps</td>
                {#each comparisonData.models as m}
                  <td>{m.convergence?.total_steps ?? '—'}</td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">Best Step</td>
                {#each comparisonData.models as m}
                  <td>{m.convergence?.best_step ?? '—'}</td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">Early Stopped</td>
                {#each comparisonData.models as m}
                  <td>{m.convergence?.stopped_early != null ? (m.convergence.stopped_early ? 'Yes' : 'No') : '—'}</td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">Convergence Rate</td>
                {#each comparisonData.models as m}
                  <td>{fmt(m.convergence?.convergence_rate, 6)}</td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">LoRA Rank</td>
                {#each comparisonData.models as m}
                  <td>{m.lora?.rank ?? '—'}</td>
                {/each}
              </tr>
              <tr>
                <td class="metric-label">LoRA Alpha</td>
                {#each comparisonData.models as m}
                  <td>{m.lora?.alpha ?? '—'}</td>
                {/each}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    {/if}
  </div>

{:else}
  <!-- ========== LIST VIEW ========== -->
  <div class="models-page">
    <div class="page-header">
      <div class="header-left">
        <h2>Models</h2>
        <span class="subtitle">Versioned adapters with diagnostics and comparison</span>
      </div>
      {#if selected.size >= 2}
        <button class="btn-compare" on:click={startCompare}>
          Compare ({selected.size})
        </button>
      {/if}
    </div>

    {#if error}
      <div class="error">{error}</div>
    {/if}

    {#if models.length === 0 && !error}
      <div class="empty">No models yet. Run <code>python bundle.py</code> to create one.</div>
    {/if}

    {#if models.length > 0}
      <div class="kpi-row">
        <KPICard label="Total Models" value={models.length} />
        <KPICard label="Latest Version" value={`v${models[0].version}`} />
        <KPICard label="Best Score" value={bestScore()} />
      </div>

      <div class="models-list">
        <table>
          <thead>
            <tr>
              <th class="col-check">
                <input type="checkbox" checked={selected.size === models.length} on:change={selectAll} />
              </th>
              <th>Model</th>
              <th>Base Model</th>
              <th>Score</th>
              <th>Size</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {#each models as m}
              {@const name = modelName(m)}
              <tr
                class:selected={selected.has(name)}
                class:detail-open={modelName(detailModel) === name}
              >
                <td class="col-check">
                  <input
                    type="checkbox"
                    checked={selected.has(name)}
                    on:change={() => toggleSelect(name)}
                  />
                </td>
                <td class="model-name" on:click={() => showDetail(name)}>{name}</td>
                <td class="base-model">{m.base_model}</td>
                <td>
                  <span class="score" class:excellent={m.score >= 0.95} class:good={m.score >= 0.8 && m.score < 0.95}>
                    {m.score?.toFixed(4) ?? '—'}
                  </span>
                </td>
                <td>{m.size_mb} MB</td>
                <td class="ts">{m.created?.slice(0, 10)}</td>
                <td class="actions">
                  <button class="btn-sm" on:click={() => showDetail(name)}>Details</button>
                  <button class="btn-sm btn-primary" on:click={() => download(name)}>Download</button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <!-- Single model detail panel -->
    {#if detailModel && !detailLoading}
      <div class="detail-panel">
        <h3>{detailModel.model_name}</h3>
        <div class="detail-grid">
          <div class="detail-section">
            <div class="section-title">Provenance</div>
            <dl>
              <dt>Base Model</dt><dd>{detailModel.base_model}</dd>
              <dt>Git SHA</dt><dd class="mono">{detailModel.git_sha?.slice(0, 12)}</dd>
              <dt>Config Hash</dt><dd class="mono">{detailModel.config_hash}</dd>
            </dl>
          </div>
          <div class="detail-section">
            <div class="section-title">LoRA Config</div>
            <dl>
              <dt>Rank</dt><dd>{detailModel.lora?.rank}</dd>
              <dt>Alpha</dt><dd>{detailModel.lora?.alpha}</dd>
              <dt>Modules</dt><dd>{detailModel.lora?.target_modules?.join(', ')}</dd>
            </dl>
          </div>
          <div class="detail-section">
            <div class="section-title">Evaluation</div>
            <dl>
              <dt>Score</dt><dd>{fmt(detailModel.score)}</dd>
              <dt>Records</dt><dd>{detailModel.num_records}</dd>
              {#if detailModel.band_counts}
                <dt>Bands</dt>
                <dd>
                  {#each Object.entries(detailModel.band_counts) as [band, count]}
                    <span class="band-tag band-{band.toLowerCase()}">{band}: {count}</span>
                  {/each}
                </dd>
              {/if}
            </dl>
          </div>
          {#if detailModel.convergence}
            <div class="detail-section">
              <div class="section-title">Convergence</div>
              <dl>
                <dt>First Loss</dt><dd>{fmt(detailModel.convergence.first_loss, 4)}</dd>
                <dt>Best Loss</dt><dd>{fmt(detailModel.convergence.best_loss, 5)}</dd>
                <dt>Final Loss</dt><dd>{fmt(detailModel.convergence.final_loss, 5)}</dd>
                <dt>Total Steps</dt><dd>{detailModel.convergence.total_steps}</dd>
                <dt>Early Stopped</dt><dd>{detailModel.convergence.stopped_early ? 'Yes' : 'No'}</dd>
              </dl>
            </div>
          {/if}
          {#if detailModel.train_data}
            <div class="detail-section">
              <div class="section-title">Training Data</div>
              <dl>
                <dt>Path</dt><dd class="mono">{detailModel.train_data.path?.split('/').pop()}</dd>
                <dt>SHA256</dt><dd class="mono">{detailModel.train_data.sha256?.slice(0, 16)}...</dd>
                <dt>Size</dt><dd>{(detailModel.train_data.size / 1024 / 1024).toFixed(1)} MB</dd>
              </dl>
            </div>
          {/if}
        </div>
        {#if detailModel.convention_breakdown?.length}
          <details>
            <summary>Convention Breakdown ({detailModel.convention_breakdown.length})</summary>
            <div class="conv-table-wrap">
              <table class="conv-table">
                <thead><tr><th>Convention</th><th>Avg</th><th>N</th></tr></thead>
                <tbody>
                  {#each detailModel.convention_breakdown.sort((a, b) => a.avg - b.avg) as c}
                    <tr>
                      <td>{c.convention}</td>
                      <td class:warn={c.avg < 0.9}>{c.avg.toFixed(3)}</td>
                      <td>{c.n}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          </details>
        {/if}
      </div>
    {/if}
    {#if detailLoading}
      <div class="loading">Loading model details...</div>
    {/if}
  </div>
{/if}

<style>
  .models-page { padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }

  /* Header */
  .page-header { display: flex; align-items: center; justify-content: space-between; }
  .header-left { display: flex; align-items: baseline; gap: 0.8rem; }
  h2 { margin: 0; font-size: 1rem; color: #e2e8f0; }
  .subtitle { font-size: 0.75rem; color: #64748b; }

  /* Compare button */
  .btn-compare {
    padding: 0.4rem 1rem; font-size: 0.8rem; font-weight: 600; border-radius: 6px;
    border: none; background: #2563eb; color: white; cursor: pointer;
    transition: background 0.15s;
  }
  .btn-compare:hover { background: #1d4ed8; }

  /* KPIs */
  .kpi-row { display: flex; gap: 0.8rem; flex-wrap: wrap; }

  /* Status */
  .error { color: #ef4444; font-size: 0.8rem; padding: 0.5rem; background: #450a0a; border-radius: 6px; }
  .empty { color: #64748b; font-size: 0.85rem; padding: 2rem; text-align: center; }
  .empty code { background: #1a1a2e; padding: 0.2rem 0.4rem; border-radius: 4px; color: #e2e8f0; }
  .loading { color: #94a3b8; font-size: 0.85rem; padding: 2rem; text-align: center; }

  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { text-align: left; padding: 0.5rem; color: #64748b; border-bottom: 1px solid #2a2a4a; }
  td { padding: 0.5rem; color: #cbd5e1; border-bottom: 1px solid #1e293b; }
  tr.selected { background: rgba(37, 99, 235, 0.1); }
  tr.detail-open { background: #1e3a5f; }
  .col-check { width: 32px; text-align: center; }
  .model-name { font-weight: 600; color: #e2e8f0; cursor: pointer; }
  .model-name:hover { text-decoration: underline; color: #93c5fd; }
  .base-model { font-size: 0.75rem; color: #94a3b8; }
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

  /* Detail panel */
  .detail-panel {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 1rem;
  }
  .detail-panel h3 { margin: 0 0 0.8rem; font-size: 0.85rem; color: #94a3b8; }
  .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 0.8rem; }
  .detail-section { }
  .section-title { font-size: 0.7rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 0.4rem; }
  dl { margin: 0; font-size: 0.78rem; }
  dt { color: #64748b; display: inline; }
  dt::after { content: ': '; }
  dd { color: #e2e8f0; display: inline; margin: 0; }
  dd::after { content: '\A'; white-space: pre; }
  .mono { font-family: monospace; font-size: 0.72rem; }

  .band-tag {
    display: inline-block; font-size: 0.65rem; padding: 0.1rem 0.35rem;
    border-radius: 3px; margin-right: 0.3rem; font-weight: 600;
  }
  .band-excellent { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; }
  .band-good { background: rgba(59, 130, 246, 0.2); color: #93c5fd; }
  .band-partial { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
  .band-poor { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
  .band-error { background: rgba(107, 114, 128, 0.2); color: #9ca3af; }

  details { margin-top: 0.5rem; }
  summary { font-size: 0.75rem; color: #64748b; cursor: pointer; }
  .conv-table-wrap { max-height: 250px; overflow-y: auto; margin-top: 0.5rem; }
  .conv-table { font-size: 0.75rem; }
  .conv-table td.warn { color: #fbbf24; }

  /* ========== COMPARISON VIEW ========== */
  .compare-header { display: flex; align-items: center; gap: 1rem; }
  .btn-back {
    padding: 0.3rem 0.8rem; font-size: 0.75rem; border-radius: 4px;
    border: 1px solid #2a2a4a; background: #1a1a2e; color: #94a3b8; cursor: pointer;
  }
  .btn-back:hover { background: #2a2a4a; color: #e2e8f0; }

  .compare-kpis { display: flex; gap: 0.8rem; flex-wrap: wrap; }
  .model-kpi {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px;
    padding: 0.8rem 1rem; flex: 1; min-width: 160px;
  }
  .kpi-name { font-size: 0.8rem; font-weight: 600; color: #e2e8f0; }
  .kpi-score { font-size: 1.4rem; font-weight: 700; color: #6ee7b7; margin: 0.2rem 0; }
  .kpi-meta { font-size: 0.65rem; color: #64748b; }

  .chart-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 0.8rem;
  }

  /* Metrics comparison table */
  .metrics-panel {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 1rem;
  }
  .metrics-table-wrap { overflow-x: auto; }
  .metrics-table { font-size: 0.78rem; }
  .metrics-table th { font-weight: 600; color: #e2e8f0; min-width: 100px; }
  .metric-label { color: #94a3b8; font-weight: 500; }
  td.better { color: #6ee7b7; }
  td.worse { color: #fca5a5; }
</style>
