<script lang="ts">
  import { agentStatus, agentMessages, recommendations } from '../lib/stores';
  import { agentWs } from '../lib/ws';

  export let visible = false;

  let input = '';

  function sendCommand() {
    if (!input.trim()) return;
    agentWs.send({ content: input.trim() });
    agentMessages.update(msgs => [...msgs, { type: 'user_command', content: input.trim() }]);
    input = '';
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendCommand();
    }
  }
</script>

{#if visible}
  <aside class="agent-panel">
    <div class="panel-header">
      <span class="panel-title">Agent</span>
      <span class="status-badge" class:connected={$agentStatus === 'connected'}>
        {$agentStatus}
      </span>
    </div>

    <div class="messages">
      {#if $recommendations.length > 0}
        <div class="section-label">Recommendations</div>
        {#each $recommendations as rec}
          <div class="rec" class:high={rec.priority === 'high'}>
            <div class="rec-title">{rec.title}</div>
            <div class="rec-detail">{rec.detail}</div>
          </div>
        {/each}
      {/if}

      {#each $agentMessages as msg}
        <div class="message" class:user={msg.type === 'user_command'} class:alert={msg.type === 'pipeline_alert'}>
          <span class="msg-content">{msg.content}</span>
        </div>
      {/each}
    </div>

    <div class="input-area">
      <input
        type="text"
        bind:value={input}
        on:keydown={handleKeydown}
        placeholder="Command the agent..."
      />
      <button on:click={sendCommand}>Send</button>
    </div>
  </aside>
{/if}

<style>
  .agent-panel {
    width: 320px;
    background: #0f0f23;
    border-left: 1px solid #2a2a4a;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #2a2a4a;
  }
  .panel-title { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; }
  .status-badge {
    font-size: 0.7rem;
    padding: 0.15rem 0.4rem;
    border-radius: 8px;
    background: #1e293b;
    color: #64748b;
  }
  .status-badge.connected { background: #064e3b; color: #6ee7b7; }
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    margin-top: 0.5rem;
  }
  .rec {
    padding: 0.5rem;
    background: #1a1a2e;
    border-radius: 6px;
    border-left: 3px solid #3b82f6;
  }
  .rec.high { border-left-color: #ef4444; }
  .rec-title { font-size: 0.8rem; font-weight: 500; color: #e2e8f0; }
  .rec-detail { font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem; }
  .message {
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    background: #1a1a2e;
    font-size: 0.8rem;
    color: #cbd5e1;
  }
  .message.user { background: #1e3a5f; align-self: flex-end; }
  .message.alert { background: #3b1111; border-left: 3px solid #ef4444; color: #fca5a5; }
  .input-area {
    display: flex;
    gap: 0.4rem;
    padding: 0.5rem;
    border-top: 1px solid #2a2a4a;
  }
  .input-area input {
    flex: 1;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    color: #e2e8f0;
    font-size: 0.8rem;
  }
  .input-area input:focus { outline: none; border-color: #3b82f6; }
  .input-area button {
    background: #3b82f6;
    border: none;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    color: white;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .input-area button:hover { background: #2563eb; }
</style>
