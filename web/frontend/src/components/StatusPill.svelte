<script lang="ts">
  import { isTraining } from '../lib/stores';
</script>

<span class="pill" class:active={$isTraining}>
  <span class="dot"></span>
  {$isTraining ? 'Training' : 'Idle'}
</span>

<style>
  .pill {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    background: #1e293b;
    color: #64748b;
    border: 1px solid #334155;
  }
  .pill.active {
    background: #064e3b;
    color: #6ee7b7;
    border-color: #065f46;
  }
  .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #64748b;
  }
  .pill.active .dot {
    background: #10b981;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
</style>
