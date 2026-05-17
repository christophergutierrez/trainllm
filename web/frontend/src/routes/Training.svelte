<script lang="ts">
  import { onMount } from 'svelte';
  import { latestLoss, latestStep, latestLR, trainingEvents, isTraining } from '../lib/stores';
  import { api } from '../lib/api';
  import KPICard from '../components/KPICard.svelte';
  import ChartContainer from '../components/ChartContainer.svelte';
  import Hint from '../components/Hint.svelte';

  let historicChart: any = null;
  let logScale = false;
  let maxSteps = 0;
  let runConfig: any = null;

  onMount(async () => {
    try {
      runConfig = await api.config.get();
      maxSteps = runConfig?.training?.max_steps || 0;
    } catch {}
    try {
      const runs = await api.runs.list();
      if (runs.length > 0) {
        historicChart = await api.runs.lossChart(runs[0].id);
      }
    } catch {}
  });

  $: liveChart = $trainingEvents.length >= 2 ? buildLiveChart($trainingEvents, logScale) : null;
  $: displayChart = liveChart || applyScale(historicChart, logScale);

  function buildLiveChart(events: any[], useLog: boolean) {
    const steps = events.map(e => e.step).filter(Boolean);
    const losses = events.map(e => e.value).filter((v: any) => v != null);
    if (steps.length < 2) return null;
    return {
      data: [{
        x: steps,
        y: losses,
        type: 'scatter',
        mode: 'lines',
        name: 'Train Loss',
        line: { color: '#3b82f6', width: 2 },
      }],
      layout: {
        xaxis: { title: { text: 'Step' }, color: '#94a3b8', gridcolor: '#2a2a4a' },
        yaxis: { title: { text: 'Loss' }, type: useLog ? 'log' : 'linear', color: '#94a3b8', gridcolor: '#2a2a4a' },
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 50, r: 20, t: 20, b: 40 },
        height: 300,
        legend: { orientation: 'h', yanchor: 'bottom', y: 1.02 },
      },
    };
  }

  function applyScale(chart: any, useLog: boolean) {
    if (!chart) return null;
    return {
      ...chart,
      layout: {
        ...chart.layout,
        yaxis: { ...(chart.layout?.yaxis || {}), type: useLog ? 'log' : 'linear' },
      },
    };
  }

  $: secPerStep = (() => {
    const evts = $trainingEvents.filter((e: any) => e.step && e.timestamp);
    if (evts.length < 2) return null;
    const last = evts[evts.length - 1];
    const prev = evts[evts.length - 2];
    const dt = (new Date(last.timestamp).getTime() - new Date(prev.timestamp).getTime()) / 1000;
    const ds = last.step - prev.step;
    if (ds <= 0) return null;
    return dt / ds;
  })();

  $: progress = (maxSteps > 0 && $latestStep > 0)
    ? Math.round(($latestStep / maxSteps) * 100)
    : null;

  $: elapsed = (() => {
    const evts = $trainingEvents.filter((e: any) => e.timestamp);
    if (evts.length < 2) return null;
    const first = new Date(evts[0].timestamp).getTime();
    const last = new Date(evts[evts.length - 1].timestamp).getTime();
    return (last - first) / 1000;
  })();

  $: etaMax = (() => {
    if (!secPerStep || !maxSteps || !$latestStep) return null;
    return (maxSteps - $latestStep) * secPerStep;
  })();

  function fmtDuration(seconds: number | null): string {
    if (seconds == null) return '—';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  }
</script>

<div class="training-page">
  <div class="kpi-row">
    <KPICard
      label="Loss"
      value={$latestLoss?.toFixed(4) ?? '—'}
      trend={$isTraining ? 'down' : null}
      hint="Current training loss at the latest logged step"
    />
    <KPICard
      label="Step"
      value={$latestStep || '—'}
      hint="Gradient updates completed so far"
    />
    <KPICard
      label="Progress"
      value={progress != null ? `${progress}%` : '—'}
      hint="Percent of max_steps completed. Training may end earlier — early stopping halts when loss plateaus."
    />
    <KPICard
      label="Elapsed"
      value={fmtDuration(elapsed)}
      hint="Wall-clock time since training steps began (excludes model loading)"
    />
    <KPICard
      label="Remaining"
      value={fmtDuration(etaMax)}
      hint="Worst-case time left assuming all max_steps run. Early stopping typically ends sooner."
    />
    <KPICard
      label="s/step"
      value={secPerStep ? secPerStep.toFixed(1) : '—'}
      hint="Seconds per gradient update (wall-clock). Lower is faster."
    />
    <KPICard
      label="LR"
      value={$latestLR ? $latestLR.toExponential(1) : '—'}
      hint="Current learning rate from the cosine scheduler"
    />
  </div>

  <div class="charts-row">
    <div class="chart-wrapper">
      {#if liveChart}
        <span class="live-badge">LIVE</span>
      {/if}
      <button class="scale-toggle" on:click={() => logScale = !logScale}>
        {logScale ? 'LOG' : 'LIN'}
      </button>
      <ChartContainer title="Loss Curve" figure={displayChart} />
    </div>
  </div>

  {#if runConfig}
    {@const t = runConfig.training || {}}
    {@const d = runConfig.data || {}}
    <div class="run-config">
      <h3>Run Config</h3>
      <div class="config-sections">
        <div class="config-group">
          <span class="group-label">Model</span>
          <div class="config-items">
            <span class="cfg-item">{runConfig.model || '—'}</span>
            {#if t.load_in_fp8}
              <span class="cfg-item tag">FP8</span>
              <Hint text="8-bit floating point quantization. Reduces memory ~50% vs FP16 with minimal quality loss. Native on Blackwell GPUs." />
            {:else if t.load_in_4bit}
              <span class="cfg-item tag">4-bit</span>
              <Hint text="4-bit NormalFloat quantization (QLoRA). Maximum memory savings at the cost of some precision." />
            {/if}
          </div>
        </div>

        <div class="config-group">
          <span class="group-label">LoRA</span>
          <div class="config-items">
            <span class="cfg-item">r={t.lora_rank || 16}</span>
            <Hint text="Rank — number of trainable dimensions per layer. Higher = more capacity but slower and more VRAM." />
            <span class="cfg-item">α={t.lora_alpha || 32}</span>
            <Hint text="Alpha — scaling factor for LoRA updates. Typically 2× rank. Controls how strongly adapter weights influence the output." />
            {#if t.use_rslora !== false}
              <span class="cfg-item tag">rsLoRA</span>
              <Hint text="Rank-Stabilized LoRA. Uses α/√r scaling instead of α/r, allowing stable training at higher ranks without retuning LR." />
            {/if}
            {#if t.lora_init && t.lora_init !== 'gaussian'}
              <span class="cfg-item tag">{t.lora_init}</span>
            {/if}
          </div>
        </div>

        <div class="config-group">
          <span class="group-label">Optimizer</span>
          <div class="config-items">
            <span class="cfg-item">{t.optimizer || 'adamw_torch'}</span>
            <Hint text="Weight update algorithm. adamw_8bit uses quantized optimizer states for ~30% less VRAM vs standard AdamW." />
            <span class="cfg-item">{t.lr_scheduler || 'cosine'}</span>
            <Hint text="How learning rate changes over training. Cosine decays smoothly from peak to near-zero; WSD holds stable then decays." />
            <span class="cfg-item">LR {t.learning_rate || 2e-4}</span>
            <Hint text="Peak learning rate. Step size for weight updates — too high causes instability, too low causes slow convergence." />
          </div>
        </div>

        <div class="config-group">
          <span class="group-label">Batch</span>
          <div class="config-items">
            <span class="cfg-item">bs={t.batch_size || 2}</span>
            <Hint text="Micro-batch size — samples per GPU per forward pass. Limited by VRAM." />
            <span class="cfg-item">× ga={t.gradient_accumulation_steps || 4}</span>
            <Hint text="Gradient accumulation — simulates a larger batch by accumulating gradients over multiple forward passes before updating weights." />
            <span class="cfg-item">seq={t.max_seq_length || 2048}</span>
            <Hint text="Maximum sequence length in tokens. Longer sequences use quadratically more memory (attention)." />
          </div>
        </div>

        <div class="config-group">
          <span class="group-label">Dataset</span>
          <div class="config-items">
            <span class="cfg-item">{(d.train || '').split('/').pop() || '—'}</span>
          </div>
        </div>

        <div class="config-group">
          <span class="group-label">Features</span>
          <div class="config-items">
            {#if t.neftune_noise_alpha && t.neftune_noise_alpha > 0}
              <span class="cfg-item tag">NEFTune α={t.neftune_noise_alpha}</span>
              <Hint text="Adds uniform noise to embeddings during training. Acts as regularizer, improves instruction-following 10-30%." />
            {/if}
            {#if t.train_on_responses_only !== false}
              <span class="cfg-item tag">Response-only loss</span>
              <Hint text="Only computes loss on assistant tokens, ignoring system/user prompts. Focuses all learning capacity on generating correct outputs." />
            {/if}
            {#if t.eval_during_training !== false}
              <span class="cfg-item tag">Eval split</span>
              <Hint text="Holds out 5% of data for validation. Monitors overfitting by evaluating on unseen examples every save_steps." />
            {/if}
            <span class="cfg-item tag">Plateau detect</span>
            <Hint text="Stops training early when loss stops improving. Saves time and prevents overfitting past the point of diminishing returns." />
          </div>
        </div>

        <div class="config-group">
          <span class="group-label">Stopping</span>
          <div class="config-items">
            <span class="cfg-item">max {t.max_steps || 2000} steps</span>
            <Hint text="Hard ceiling on gradient updates. Training ends here even if still improving. Actual stop is usually earlier via plateau detection." />
            <span class="cfg-item">patience=200 δ=0.002</span>
            <Hint text="Plateau detector triggers if loss doesn't improve by at least 0.002 for 200 consecutive steps." />
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .training-page { padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
  .kpi-row { display: flex; gap: 0.8rem; flex-wrap: wrap; }
  .charts-row { display: grid; grid-template-columns: 1fr; gap: 0.8rem; }
  .chart-wrapper { position: relative; }
  .live-badge {
    position: absolute; top: 0.6rem; right: 0.8rem; z-index: 10;
    background: #dc2626; color: white; font-size: 0.6rem; font-weight: 700;
    padding: 0.15rem 0.4rem; border-radius: 3px; letter-spacing: 0.05em;
    animation: pulse 2s infinite;
  }
  .scale-toggle {
    position: absolute; top: 0.6rem; right: 3.5rem; z-index: 10;
    background: #1e293b; color: #94a3b8; font-size: 0.6rem; font-weight: 700;
    padding: 0.15rem 0.5rem; border-radius: 3px; border: 1px solid #2a2a4a;
    cursor: pointer; letter-spacing: 0.05em;
  }
  .scale-toggle:hover { background: #2a2a4a; color: #e2e8f0; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }
  .run-config {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 1rem;
  }
  .run-config h3 { margin: 0 0 0.8rem; font-size: 0.85rem; color: #94a3b8; }
  .config-sections { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.8rem; }
  .config-group { display: flex; flex-direction: column; gap: 0.3rem; }
  .group-label { font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 600; }
  .config-items { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }
  .cfg-item { font-size: 0.8rem; color: #e2e8f0; }
  .cfg-item.tag {
    background: #2a2a4a;
    color: #94a3b8;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.68rem;
    font-weight: 500;
  }
</style>
