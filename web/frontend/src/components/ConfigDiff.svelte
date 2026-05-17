<script lang="ts">
  export let left: Record<string, any> = {};
  export let right: Record<string, any> = {};
  export let leftLabel: string = 'Before';
  export let rightLabel: string = 'After';

  type DiffLine = { key: string; leftVal: string; rightVal: string; changed: boolean };

  $: diffLines = computeDiff(left, right);

  function computeDiff(a: Record<string, any>, b: Record<string, any>): DiffLine[] {
    const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
    return keys.map(key => {
      const lv = a[key] !== undefined ? JSON.stringify(a[key]) : '—';
      const rv = b[key] !== undefined ? JSON.stringify(b[key]) : '—';
      return { key, leftVal: lv, rightVal: rv, changed: lv !== rv };
    });
  }
</script>

<div class="config-diff">
  <div class="diff-header">
    <span class="col-label">{leftLabel}</span>
    <span class="col-label">{rightLabel}</span>
  </div>
  {#each diffLines as line}
    <div class="diff-row" class:changed={line.changed}>
      <span class="diff-key">{line.key}</span>
      <span class="diff-val left">{line.leftVal}</span>
      <span class="diff-val right">{line.rightVal}</span>
    </div>
  {/each}
</div>

<style>
  .config-diff {
    background: #0f0f23;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    font-size: 0.75rem;
    overflow: hidden;
  }
  .diff-header {
    display: grid;
    grid-template-columns: 1fr 1fr;
    padding: 0.4rem 0.6rem 0.4rem calc(30% + 0.6rem);
    background: #1a1a2e;
    border-bottom: 1px solid #2a2a4a;
  }
  .col-label { color: #64748b; font-weight: 600; }
  .diff-row {
    display: grid;
    grid-template-columns: 30% 1fr 1fr;
    padding: 0.3rem 0.6rem;
    border-bottom: 1px solid #1e293b;
  }
  .diff-row.changed { background: #1e293b; }
  .diff-key { color: #94a3b8; font-weight: 500; }
  .diff-val { color: #cbd5e1; font-family: monospace; }
  .diff-row.changed .diff-val.left { color: #fca5a5; }
  .diff-row.changed .diff-val.right { color: #6ee7b7; }
</style>
