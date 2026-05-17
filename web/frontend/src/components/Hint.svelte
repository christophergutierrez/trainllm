<script lang="ts">
  export let text: string;
</script>

<span class="hint-wrap">
  <span class="hint-icon">?</span>
  <span class="hint-tooltip">{text}</span>
</span>

<style>
  .hint-wrap {
    position: relative;
    display: inline-flex;
    margin-left: 0.2rem;
  }
  .hint-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: #2a2a4a;
    color: #64748b;
    font-size: 0.55rem;
    font-weight: 700;
    cursor: help;
    flex-shrink: 0;
  }
  .hint-wrap:hover .hint-icon {
    background: #3b82f6;
    color: white;
  }
  .hint-tooltip {
    display: none;
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    background: #1e293b;
    border: 1px solid #3b82f6;
    color: #e2e8f0;
    font-size: 0.68rem;
    font-weight: 400;
    text-transform: none;
    padding: 0.4rem 0.6rem;
    border-radius: 5px;
    white-space: normal;
    width: max-content;
    max-width: 220px;
    line-height: 1.4;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  .hint-tooltip::after {
    content: '';
    position: absolute;
    bottom: 100%;
    left: 6px;
    border: 5px solid transparent;
    border-bottom-color: #3b82f6;
  }
  .hint-wrap:hover .hint-tooltip {
    display: block;
  }
</style>
