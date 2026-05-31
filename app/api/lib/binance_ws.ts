/**
 * Server-side Binance WebSocket proxy.
 * Proxies Binance futures WebSocket streams through the server so the
 * browser doesn't need to open a direct WS to fstream.binance.com.
 * Used by the /api/ws/:symbol route in boot.ts.
 */

import type { ServerWebSocket } from "bun";

// We re-export a Hono middleware handler that upgrades HTTP → WS and
// bridges to the upstream Binance WS. This only runs in Bun; in Node
// the browser connects directly (see useBinanceTicker.ts which connects
// to wss://fstream.binance.com/ws directly from the client).

export const FSTREAM_BASE = "wss://fstream.binance.com/ws";

export function buildBinanceStreamUrl(symbol: string): string {
  const sym = symbol.toLowerCase();
  return `${FSTREAM_BASE}/${sym}@markPrice/${sym}@ticker`;
}
