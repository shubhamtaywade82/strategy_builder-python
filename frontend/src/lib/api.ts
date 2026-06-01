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

export async function apiPostStream<T>(
  path: string,
  body: unknown,
  onChunk: (chunk: T) => void,
): Promise<void> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let msg = res.statusText;
    try { const b = await res.json(); msg = b?.error?.message ?? b?.detail ?? msg; } catch { /* noop */ }
    throw new Error(msg);
  }

  if (!res.body) {
    throw new Error("No response body");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const jsonStr = line.slice(6);
        if (jsonStr) {
          const parsed = JSON.parse(jsonStr);
          onChunk(parsed);
        }
      }
    }
  }

  // Process any remaining buffered data
  if (buffer.startsWith("data: ")) {
    const jsonStr = buffer.slice(6);
    if (jsonStr) {
      const parsed = JSON.parse(jsonStr);
      onChunk(parsed);
    }
  }
}
