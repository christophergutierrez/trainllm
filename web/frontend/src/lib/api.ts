/** Thin REST client — all data shaping is done server-side. */

const BASE = import.meta.env.VITE_API_URL || '';

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE}${path}`);
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function post<T>(path: string, body?: object): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
  return resp.json();
}

export const api = {
  health: () => get<{ status: string; adapter: string }>('/api/health'),
  runs: {
    list: () => get<any[]>('/api/runs'),
    get: (id: string) => get<any>(`/api/runs/${id}`),
    lossChart: (id: string) => get<any>(`/api/runs/${id}/loss-chart`),
  },
  evals: {
    get: (id: string) => get<any>(`/api/evals/${id}`),
    records: (id: string, params?: Record<string, string>) => {
      const qs = params ? '?' + new URLSearchParams(params).toString() : '';
      return get<any>(`/api/evals/${id}/records${qs}`);
    },
    bandChart: (id: string) => get<any>(`/api/evals/${id}/charts/bands`),
    scoreChart: (id: string) => get<any>(`/api/evals/${id}/charts/scores`),
  },
  diagnostics: {
    timing: () => get<any>('/api/diagnostics/timing'),
    timingChart: () => get<any>('/api/diagnostics/timing/chart'),
    convergence: () => get<any>('/api/diagnostics/convergence'),
    gpu: () => get<any>('/api/diagnostics/gpu'),
    pipelineHealth: () => get<any>('/api/diagnostics/pipeline-health'),
  },
  config: {
    get: () => get<any>('/api/config'),
    training: () => get<any>('/api/config/training'),
  },
  training: {
    state: () => get<any>('/api/training/state'),
  },
  cycle: {
    status: () => get<any>('/api/cycle/status'),
    start: (opts?: { skip_train?: boolean; skip_judge?: boolean }) =>
      post<any>(`/api/cycle/start?${new URLSearchParams(opts as any || {})}`),
    stop: () => post<any>('/api/cycle/stop'),
  },
  models: {
    list: () => get<any>('/api/models'),
    manifest: (name: string) => get<any>(`/api/models/${name}`),
    diagnostics: (name: string) => get<any>(`/api/models/${name}/diagnostics`),
    compare: (names: string[]) => get<any>(`/api/models/compare?names=${names.join(',')}`),
    downloadUrl: (name: string) => `${BASE}/api/models/${name}/download`,
  },
};
