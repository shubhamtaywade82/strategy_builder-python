import { useMutation, useQuery } from "@tanstack/react-query";
import { apiGet, apiPost } from "@/lib/api";
import type { ResearchResult } from "@/types";

export interface RunResearchInput {
  symbol: string; days: number; leverage: number; horizon: number; rrs: string[];
}

export function useResearchRun(opts?: {
  onSuccess?: (d: ResearchResult) => void; onError?: (e: Error) => void;
}) {
  return useMutation<ResearchResult, Error, RunResearchInput>({
    mutationFn: (input) => apiPost<ResearchResult>("/api/research/run", input),
    onSuccess: opts?.onSuccess,
    onError: opts?.onError,
  });
}

export function useAiHealth() {
  return useQuery({
    queryKey: ["ai", "health"],
    queryFn: () => apiGet<{ ok: boolean; models: string[]; message: string }>("/api/ai/health"),
  });
}

export interface ChatInput {
  messages: { role: string; content: string }[]; model?: string; temperature?: number; sessionId?: string;
}

export function useAiChat(opts?: {
  onSuccess?: (d: { content: string; model: string }) => void; onError?: (e: Error) => void;
}) {
  return useMutation<{ content: string; model: string }, Error, ChatInput>({
    mutationFn: (input) => apiPost("/api/ai/chat", input),
    onSuccess: opts?.onSuccess,
    onError: opts?.onError,
  });
}

export interface GenStrategyInput {
  symbol: string; rr: string; features: string[]; metrics: Record<string, number>;
  model?: string; sessionId?: string;
}

export function useGenerateStrategy(opts?: {
  onSuccess?: (d: { analysis: string }) => void; onError?: (e: Error) => void;
}) {
  return useMutation<{ analysis: string }, Error, GenStrategyInput>({
    mutationFn: (input) => apiPost("/api/ai/generate-strategy", input),
    onSuccess: opts?.onSuccess,
    onError: opts?.onError,
  });
}
