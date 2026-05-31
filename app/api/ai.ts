import { z } from "zod";
import { createRouter, publicQuery } from "./middleware";
import {
  chatCompletion,
  generateStrategy,
  analyzeMarket,
  listModels,
  healthCheck,
  type OllamaMessage,
} from "./lib/ollama";
import { getDb } from "./queries/connection";
import { chatMessages, researchSessions, strategyInsights } from "../db/schema";
import { eq, desc } from "drizzle-orm";

export const aiRouter = createRouter({
  // Health check for Ollama
  health: publicQuery.query(async () => {
    const ok = await healthCheck();
    const models = ok ? await listModels() : [];
    return { ok, models, message: ok ? "Ollama connected" : "Ollama unavailable - check OLLAMA_URL" };
  }),

  // Chat with AI assistant
  chat: publicQuery
    .input(
      z.object({
        messages: z.array(
          z.object({
            role: z.enum(["system", "user", "assistant"]),
            content: z.string(),
          })
        ),
        model: z.string().optional().default("llama3.1"),
        temperature: z.number().optional().default(0.7),
        sessionId: z.string().optional(),
      })
    )
    .mutation(async ({ input }) => {
      const response = await chatCompletion(
        input.messages as OllamaMessage[],
        input.model,
        { temperature: input.temperature }
      );

      // Save to DB if sessionId provided
      if (input.sessionId) {
        const db = getDb();
        await db.insert(chatMessages).values({
          sessionId: input.sessionId,
          role: "user",
          content: input.messages[input.messages.length - 1].content,
          model: input.model,
        });
        await db.insert(chatMessages).values({
          sessionId: input.sessionId,
          role: "assistant",
          content: response.message.content,
          model: input.model,
        });
      }

      return {
        content: response.message.content,
        model: input.model,
        timing: {
          total: response.total_duration,
          promptTokens: response.prompt_eval_count,
          completionTokens: response.eval_count,
        },
      };
    }),

  // Get chat history for a session
  history: publicQuery
    .input(z.object({ sessionId: z.string() }))
    .query(async ({ input }) => {
      const db = getDb();
      const messages = await db
        .select()
        .from(chatMessages)
        .where(eq(chatMessages.sessionId, input.sessionId))
        .orderBy(chatMessages.createdAt);
      return messages;
    }),

  // Generate strategy analysis with AI
  generateStrategy: publicQuery
    .input(
      z.object({
        symbol: z.string(),
        rr: z.string(),
        features: z.array(z.string()),
        metrics: z.record(z.number()),
        model: z.string().optional().default("llama3.1"),
        sessionId: z.string().optional(),
      })
    )
    .mutation(async ({ input }) => {
      const analysis = await generateStrategy(
        input.symbol,
        input.rr,
        input.features,
        input.metrics,
        input.model
      );

      if (input.sessionId) {
        const db = getDb();
        await db.insert(strategyInsights).values({
          sessionId: input.sessionId,
          strategyName: `${input.symbol}_${input.rr}`,
          insightType: "analysis",
          content: analysis,
          model: input.model,
        });
      }

      return { analysis };
    }),

  // AI market analysis
  marketAnalysis: publicQuery
    .input(
      z.object({
        symbol: z.string(),
        priceData: z.object({
          currentPrice: z.number(),
          change24h: z.number(),
          high24h: z.number(),
          low24h: z.number(),
          volume24h: z.number(),
        }),
        model: z.string().optional().default("llama3.1"),
      })
    )
    .mutation(async ({ input }) => {
      const analysis = await analyzeMarket(
        input.symbol,
        input.priceData,
        input.model
      );
      return { analysis };
    }),

  // Save research session
  saveSession: publicQuery
    .input(
      z.object({
        sessionId: z.string(),
        symbol: z.string(),
        rrConfig: z.string(),
        leverage: z.number().optional().default(10),
        days: z.number().optional().default(60),
        resultJson: z.string().optional(),
        bestStrategy: z.string().optional(),
        winRate: z.number().optional(),
        profitFactor: z.number().optional(),
        expectancy: z.number().optional(),
      })
    )
    .mutation(async ({ input }) => {
      const db = getDb();
      await db.insert(researchSessions).values({
        ...input,
        status: "completed",
      });
      return { success: true };
    }),

  // Get research sessions
  sessions: publicQuery.query(async () => {
    const db = getDb();
    const sessions = await db
      .select()
      .from(researchSessions)
      .orderBy(desc(researchSessions.createdAt))
      .limit(50);
    return sessions;
  }),

  // Get strategy insights
  insights: publicQuery
    .input(z.object({ sessionId: z.string() }))
    .query(async ({ input }) => {
      const db = getDb();
      const insights = await db
        .select()
        .from(strategyInsights)
        .where(eq(strategyInsights.sessionId, input.sessionId))
        .orderBy(desc(strategyInsights.createdAt));
      return insights;
    }),

  // Clear chat history
  clearHistory: publicQuery
    .input(z.object({ sessionId: z.string() }))
    .mutation(async ({ input }) => {
      const db = getDb();
      await db
        .delete(chatMessages)
        .where(eq(chatMessages.sessionId, input.sessionId));
      return { success: true };
    }),
});
