<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let columns: Array<{ key: string; label: string; width?: string }> = [];
  export let rows: any[] = [];
  export let selectedId: string | null = null;
  export let sortBy: string = '';
  export let sortDesc: boolean = true;

  const dispatch = createEventDispatcher();

  function handleSort(key: string) {
    if (sortBy === key) {
      sortDesc = !sortDesc;
    } else {
      sortBy = key;
      sortDesc = true;
    }
    dispatch('sort', { sortBy, sortDesc });
  }

  function handleRowClick(row: any) {
    dispatch('select', row);
  }

  function formatCell(value: any): string {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'number') return value.toFixed(3);
    return String(value);
  }
</script>

<div class="table-wrapper">
  <table>
    <thead>
      <tr>
        {#each columns as col}
          <th
            style={col.width ? `width: ${col.width}` : ''}
            on:click={() => handleSort(col.key)}
            class:sorted={sortBy === col.key}
          >
            {col.label}
            {#if sortBy === col.key}
              <span class="sort-arrow">{sortDesc ? '↓' : '↑'}</span>
            {/if}
          </th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each rows as row}
        <tr
          class:selected={selectedId === row.id}
          on:click={() => handleRowClick(row)}
        >
          {#each columns as col}
            <td>
              {#if col.key === 'band'}
                <span class="band-badge {(row[col.key] || '').toLowerCase()}">{row[col.key] || '—'}</span>
              {:else}
                {formatCell(row[col.key])}
              {/if}
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .table-wrapper { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th {
    text-align: left; padding: 0.5rem; color: #64748b;
    border-bottom: 1px solid #2a2a4a; cursor: pointer; user-select: none;
  }
  th:hover { color: #94a3b8; }
  th.sorted { color: #e2e8f0; }
  .sort-arrow { font-size: 0.7rem; margin-left: 0.2rem; }
  td { padding: 0.5rem; color: #cbd5e1; border-bottom: 1px solid #1e293b; }
  tr { cursor: pointer; transition: background 0.1s; }
  tr:hover { background: #1e293b; }
  tr.selected { background: #1e3a5f; }
  .band-badge {
    padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;
  }
  .band-badge.excellent { background: #064e3b; color: #6ee7b7; }
  .band-badge.good { background: #1e3a5f; color: #93c5fd; }
  .band-badge.partial { background: #451a03; color: #fcd34d; }
  .band-badge.poor { background: #450a0a; color: #fca5a5; }
  .band-badge.error { background: #1e293b; color: #64748b; }
</style>
