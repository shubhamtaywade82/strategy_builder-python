import { useState, useEffect, useRef, useCallback } from "react";

export interface TickerData {
  symbol: string;
  markPrice: number;
  indexPrice: number;
  fundingRate: number;
  nextFundingTime: number;
  change24h: number;
  high24h: number;
  low24h: number;
  volume24h: number;
  connected: boolean;
}

const FAPI_BASE = "https://fapi.binance.com";
const WS_BASE = "wss://fstream.binance.com/stream?streams=";

async function fetchInitialTicker(symbol: string): Promise<Partial<TickerData>> {
  const sym = symbol.toUpperCase();

  const [markRes, statsRes] = await Promise.all([
    fetch(`${FAPI_BASE}/fapi/v1/premiumIndex?symbol=${sym}`),
    fetch(`${FAPI_BASE}/fapi/v1/ticker/24hr?symbol=${sym}`),
  ]);

  const [markData, statsData] = await Promise.all([
    markRes.json(),
    statsRes.json(),
  ]);

  return {
    symbol: sym,
    markPrice: parseFloat(markData.markPrice ?? "0"),
    indexPrice: parseFloat(markData.indexPrice ?? "0"),
    fundingRate: parseFloat(markData.lastFundingRate ?? "0"),
    nextFundingTime: markData.nextFundingTime ?? 0,
    change24h: parseFloat(statsData.priceChangePercent ?? "0"),
    high24h: parseFloat(statsData.highPrice ?? "0"),
    low24h: parseFloat(statsData.lowPrice ?? "0"),
    volume24h: parseFloat(statsData.quoteVolume ?? "0"),
  };
}

export function useBinanceTicker(symbol: string): TickerData {
  const [ticker, setTicker] = useState<TickerData>({
    symbol: symbol.toUpperCase(),
    markPrice: 0,
    indexPrice: 0,
    fundingRate: 0,
    nextFundingTime: 0,
    change24h: 0,
    high24h: 0,
    low24h: 0,
    volume24h: 0,
    connected: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback((sym: string) => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }

    const symLower = sym.toLowerCase();
    // Combined stream: mark price (1s) + 24hr ticker (1s)
    const ws = new WebSocket(`${WS_BASE}${symLower}@markPrice/${symLower}@ticker`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setTicker((prev) => ({ ...prev, connected: true }));
    };

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return;
      try {
        const msg = JSON.parse(evt.data as string);
        // Binance combined stream wraps in { stream, data }
        const data = msg.data ?? msg;
        const eventType = data.e;

        if (eventType === "markPriceUpdate") {
          setTicker((prev) => ({
            ...prev,
            symbol: data.s,
            markPrice: parseFloat(data.p ?? "0"),
            indexPrice: parseFloat(data.i ?? "0"),
            fundingRate: parseFloat(data.r ?? "0"),
            nextFundingTime: data.T ?? 0,
            connected: true,
          }));
        } else if (eventType === "24hrTicker") {
          setTicker((prev) => ({
            ...prev,
            change24h: parseFloat(data.P ?? "0"),
            high24h: parseFloat(data.h ?? "0"),
            low24h: parseFloat(data.l ?? "0"),
            volume24h: parseFloat(data.q ?? "0"),
            connected: true,
          }));
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setTicker((prev) => ({ ...prev, connected: false }));
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setTicker((prev) => ({ ...prev, connected: false }));
      // Reconnect after 3s on unexpected close
      reconnectRef.current = setTimeout(() => {
        if (mountedRef.current) connect(sym);
      }, 3000);
    };
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    const sym = symbol.toUpperCase();

    // Fetch REST data immediately for fast initial display
    fetchInitialTicker(sym)
      .then((data) => {
        if (mountedRef.current) {
          setTicker((prev) => ({ ...prev, ...data }));
        }
      })
      .catch(() => {});

    // Start WebSocket
    connect(sym);

    return () => {
      mountedRef.current = false;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [symbol, connect]);

  return ticker;
}
