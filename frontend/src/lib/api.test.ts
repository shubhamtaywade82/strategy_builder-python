import { describe, it, expect, vi } from "vitest";
import { apiPost } from "./api";

describe("apiPost", () => {
  it("posts JSON and returns parsed body", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "content-type": "application/json" } })
    ) as any;
    const out = await apiPost<{ ok: boolean }>("/api/ping", {});
    expect(out.ok).toBe(true);
  });

  it("throws normalized error on non-2xx", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ error: { message: "boom" } }), { status: 400 })
    ) as any;
    await expect(apiPost("/api/x", {})).rejects.toThrow("boom");
  });
});
