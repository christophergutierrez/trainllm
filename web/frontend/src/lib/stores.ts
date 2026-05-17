/** Svelte stores — reactive state driven by WebSocket events. */

import { writable, derived } from 'svelte/store';
import { trainingWs, agentWs } from './ws';

// Training state
export const trainingEvents = writable<any[]>([]);
export const evalEvents = writable<any[]>([]);
export const latestLoss = writable<number | null>(null);
export const latestEvalLoss = writable<number | null>(null);
export const latestStep = writable<number>(0);
export const latestLR = writable<number | null>(null);
export const cycleLog = writable<string[]>([]);
export const isTraining = writable(false);
export const trainingDone = writable(false);
export const pipelineErrors = writable<any[]>([]);
export const pipelineWarnings = writable<any[]>([]);

// Agent state
export const agentStatus = writable<'available' | 'disabled' | 'disconnected'>('disconnected');
export const agentMessages = writable<Array<{ type: string; content: string; id?: string }>>([]);
export const agentPending = writable<boolean>(false);
export const recommendations = writable<any[]>([]);

// Active tab
export const activeTab = writable<string>('training');

// Initialize WebSocket subscriptions
export function initStores() {
  trainingWs.subscribe((event) => {
    if (event.type === 'log') {
      cycleLog.update(logs => [...logs.slice(-200), event.content]);
    } else if (event.event === 'step_start' && event.step === 'train') {
      trainingEvents.set([]);
      evalEvents.set([]);
      latestLoss.set(null);
      latestEvalLoss.set(null);
      latestStep.set(0);
      latestLR.set(null);
      isTraining.set(true);
      trainingDone.set(false);
    } else if (event.event === 'loss') {
      latestLoss.set(event.value);
      latestStep.set(event.step);
      latestLR.set(event.lr);
      trainingEvents.update(events => {
        if (events.length > 0 && events[events.length - 1].step >= event.step) return events;
        return [...events, event];
      });
      isTraining.set(true);
      trainingDone.set(false);
    } else if (event.event === 'eval_loss') {
      latestEvalLoss.set(event.value);
      evalEvents.update(events => {
        if (events.length > 0 && events[events.length - 1].step >= event.step) return events;
        return [...events, event];
      });
    } else if (event.event === 'error') {
      pipelineErrors.update(errs => [...errs.slice(-19), event]);
    } else if (event.event === 'warning') {
      pipelineWarnings.update(warns => [...warns.slice(-19), event]);
    } else if (event.event === 'step_end' && event.step === 'train') {
      isTraining.set(false);
      trainingDone.set(true);
    } else if (event.type === 'cycle_end') {
      isTraining.set(false);
      trainingDone.set(true);
    }
  });

  agentWs.onStatus((connected) => {
    if (!connected) agentStatus.set('disconnected');
  });

  agentWs.subscribe((msg) => {
    if (msg.type === 'agent_status') {
      agentStatus.set(msg.status as 'available' | 'disabled');
    } else if (msg.type === 'message_ack') {
      agentPending.set(true);
    } else if (msg.type === 'agent_response') {
      agentPending.set(false);
      agentMessages.update(msgs => [...msgs.slice(-50), msg]);
    } else if (msg.type === 'recommendations') {
      recommendations.set(msg.data || []);
    } else if (msg.type === 'pipeline_alert') {
      agentMessages.update(msgs => [...msgs.slice(-50), {
        type: 'pipeline_alert',
        content: `[${msg.code}] ${msg.message}`,
      }]);
    }
  });
}
