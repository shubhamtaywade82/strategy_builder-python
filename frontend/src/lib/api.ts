const BASE = "";  // same-origin; Vite proxies /api in dev, FastAPI serves in prod

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText;
    try { const b = await res.json(); msg = b?.error?.message ?? b?.detail ?? msg; } catch { /* noop */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export async function apiGet<T>(path: string): Promise<T> {
  return handle<T>(await fetch(`${BASE}${path}`, { credentials: "include" }));
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return handle<T>(await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(body),
  }));
}
