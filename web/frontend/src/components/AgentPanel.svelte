<script lang="ts">
  import { get } from 'svelte/store';
  import { agentStatus, agentMessages, agentPending, recommendations,
           latestLoss, latestStep, latestEvalLoss, isTraining, trainingDone,
           pipelineErrors, pipelineWarnings, activeTab } from '../lib/stores';
  import { agentWs } from '../lib/ws';

  export let visible = false;

  let input = '';

  function sendCommand() {
    if (!input.trim()) return;
    if ($agentStatus === 'disabled') return;

    const context = {
      page: get(activeTab),
      step: get(latestStep),
      latestLoss: get(latestLoss),
      latestEvalLoss: get(latestEvalLoss),
      isTraining: get(isTraining),
      trainingDone: get(trainingDone),
      errors: get(pipelineErrors).slice(-5),
      warnings: get(pipelineWarnings).slice(-5),
    };

    agentWs.send({ content: input.trim(), context });
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
      <span class="status-badge" class:available={$agentStatus === 'available'} class:disabled={$agentStatus === 'disabled'}>
        {#if $agentStatus === 'disabled'}AI Disabled{:else}{$agentStatus}{/if}
      </span>
    </div>

    {#if $agentStatus === 'disabled'}
      <div class="disabled-notice">
        <span class="disabled-icon">⊘</span>
        <p>AI doesn't work if it wasn't spawned from an agent.</p>
        <p class="hint">Start training from a Claude Code session to enable.</p>
      </div>
    {:else}
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

        {#if $agentPending}
          <div class="pending">
            <span class="pending-dot"></span>
            Message sent — response may take a moment depending on what the agent is doing.
          </div>
        {/if}
      </div>

      <div class="input-area">
        <input
          type="text"
          bind:value={input}
          on:keydown={handleKeydown}
          placeholder="Ask the agent..."
          disabled={$agentStatus !== 'available'}
        />
        <button on:click={sendCommand} disabled={$agentStatus !== 'available'}>Send</button>
      </div>
    {/if}
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
  .status-badge.available { background: #064e3b; color: #6ee7b7; }
  .status-badge.disabled { background: #1e293b; color: #64748b; }
  .disabled-notice {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem;
    text-align: center;
    gap: 0.5rem;
  }
  .disabled-icon { font-size: 2rem; opacity: 0.4; }
  .disabled-notice p { font-size: 0.8rem; color: #64748b; margin: 0; }
  .disabled-notice .hint { font-size: 0.72rem; color: #475569; }
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
  .pending {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    color: #64748b;
    padding: 0.4rem 0.6rem;
    font-style: italic;
  }
  .pending-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #3b82f6;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
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
  .input-area input:disabled { opacity: 0.5; cursor: not-allowed; }
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
  .input-area button:disabled { opacity: 0.5; cursor: not-allowed; background: #3b82f6; }
</style>
