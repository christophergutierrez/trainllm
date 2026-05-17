/** Svelte stores — reactive state driven by WebSocket events. */

import { writable, derived } from 'svelte/store';
import { trainingWs, agentWs } from './ws';

// Training state
export const trainingEvents = writable<any[]>([]);
export const latestLoss = writable<number | null>(null);
export const latestStep = writable<number>(0);
export const latestLR = writable<number | null>(null);
export const cycleLog = writable<string[]>([]);
export const isTraining = writable(false);

// Agent state
export const agentStatus = writable<string>('disconnected');
export const agentMessages = writable<Array<{ type: string; content: string }>>([]);
export const recommendations = writable<any[]>([]);

// Active tab
export const activeTab = writable<string>('training');

// Initialize WebSocket subscriptions
export function initStores() {
  trainingWs.subscribe((event) => {
    if (event.type === 'log') {
      cycleLog.update(logs => [...logs.slice(-200), event.content]);
    } else if (event.event === 'loss') {
      latestLoss.set(event.value);
      latestStep.set(event.step);
      latestLR.set(event.lr);
      trainingEvents.update(events => [...events, event]);
      isTraining.set(true);
    } else if (event.type === 'cycle_end') {
      isTraining.set(false);
    }
  });

  agentWs.onStatus((connected) => {
    agentStatus.set(connected ? 'connected' : 'disconnected');
  });

  agentWs.subscribe((msg) => {
    if (msg.type === 'recommendations') {
      recommendations.set(msg.data || []);
    } else if (msg.type === 'agent_response' || msg.type === 'agent_thinking') {
      agentMessages.update(msgs => [...msgs.slice(-50), msg]);
    }
  });
}
