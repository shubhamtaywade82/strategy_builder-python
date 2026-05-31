/**
 * Market data proxy endpoints.
 * Proxies Binance REST / provides real-time price data server-side.
 */

import { z } from "zod";
import { createRouter, publicQuery } from "./middleware";

const FAPI = "https://fapi.binance.com";

async function binanceFetch(path: string): Promise<any> {
  const res = await fetch(`${FAPI}${path}`, {
    headers: { "Accept": "application/json" },
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error(`Binance API error: ${res.status}`);
  return res.json();
}

export const marketRouter = createRouter({
  // Mark price + funding rate for a single symbol
  markPrice: publicQuery
    .input(z.object({ symbol: z.string().regex(/^[A-Z0-9]{3,12}$/) }))
    .query(async ({ input }) => {
      const data = await binanceFetch(`/fapi/v1/premiumIndex?symbol=${input.symbol}`);
      return {
        symbol: data.symbol as string,
        markPrice: parseFloat(data.markPrice),
        indexPrice: parseFloat(data.indexPrice),
        fundingRate: parseFloat(data.lastFundingRate),
        nextFundingTime: data.nextFundingTime as number,
        time: data.time as number,
      };
    }),

  // 24h stats
  ticker24h: publicQuery
    .input(z.object({ symbol: z.string().regex(/^[A-Z0-9]{3,12}$/) }))
    .query(async ({ input }) => {
      const data = await binanceFetch(`/fapi/v1/ticker/24hr?symbol=${input.symbol}`);
      return {
        symbol: data.symbol as string,
        priceChange: parseFloat(data.priceChange),
        priceChangePct: parseFloat(data.priceChangePercent),
        high24h: parseFloat(data.highPrice),
        low24h: parseFloat(data.lowPrice),
        volume24h: parseFloat(data.volume),
        quoteVolume24h: parseFloat(data.quoteVolume),
        lastPrice: parseFloat(data.lastPrice),
        openPrice: parseFloat(data.openPrice),
        count: data.count as number,
      };
    }),

  // Recent klines (last N bars) for a given timeframe — real data
  klines: publicQuery
    .input(
      z.object({
        symbol: z.string().regex(/^[A-Z0-9]{3,12}$/),
        interval: z.enum(["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]),
        limit: z.number().int().min(1).max(1500).default(100),
      })
    )
    .query(async ({ input }) => {
      const data: any[][] = await binanceFetch(
        `/fapi/v1/klines?symbol=${input.symbol}&interval=${input.interval}&limit=${input.limit}`
      );
      return data.map((row) => ({
        openTime: row[0] as number,
        open: parseFloat(row[1]),
        high: parseFloat(row[2]),
        low: parseFloat(row[3]),
        close: parseFloat(row[4]),
        volume: parseFloat(row[5]),
        closeTime: row[6] as number,
        quoteVolume: parseFloat(row[7]),
        nTrades: row[8] as number,
        takerBuyBase: parseFloat(row[9]),
        takerBuyQuote: parseFloat(row[10]),
      }));
    }),

  // Top traded perpetuals (useful for symbol picker)
  topSymbols: publicQuery.query(async () => {
    const data: any[] = await binanceFetch("/fapi/v1/ticker/24hr");
    return data
      .filter((d) => d.symbol.endsWith("USDT") && parseFloat(d.quoteVolume) > 0)
      .sort((a, b) => parseFloat(b.quoteVolume) - parseFloat(a.quoteVolume))
      .slice(0, 30)
      .map((d) => ({
        symbol: d.symbol as string,
        lastPrice: parseFloat(d.lastPrice),
        priceChangePct: parseFloat(d.priceChangePercent),
        quoteVolume: parseFloat(d.quoteVolume),
      }));
  }),
});
