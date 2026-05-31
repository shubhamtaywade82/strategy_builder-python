import {
  mysqlTable,
  serial,
  varchar,
  text,
  timestamp,
  float,
  json,
  int,
} from "drizzle-orm/mysql-core";

// Chat messages with AI assistant
export const chatMessages = mysqlTable("chat_messages", {
  id: serial("id").primaryKey(),
  sessionId: varchar("session_id", { length: 64 }).notNull(),
  role: varchar("role", { length: 20 }).notNull(), // user, assistant, system
  content: text("content").notNull(),
  model: varchar("model", { length: 50 }).default("llama3.1"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
});

// Research sessions (saved strategy research runs)
export const researchSessions = mysqlTable("research_sessions", {
  id: serial("id").primaryKey(),
  sessionId: varchar("session_id", { length: 64 }).notNull().unique(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  rrConfig: varchar("rr_config", { length: 50 }).notNull(),
  leverage: int("leverage").notNull().default(10),
  days: int("days").notNull().default(60),
  status: varchar("status", { length: 20 }).notNull().default("pending"), // pending, running, completed, failed
  resultJson: text("result_json"),
  bestStrategy: text("best_strategy"),
  winRate: float("win_rate"),
  profitFactor: float("profit_factor"),
  expectancy: float("expectancy"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
});

// AI-generated strategy insights
export const strategyInsights = mysqlTable("strategy_insights", {
  id: serial("id").primaryKey(),
  sessionId: varchar("session_id", { length: 64 }).notNull(),
  strategyName: varchar("strategy_name", { length: 100 }).notNull(),
  insightType: varchar("insight_type", { length: 30 }).notNull(), // analysis, optimization, risk_assessment
  content: text("content").notNull(),
  model: varchar("model", { length: 50 }).default("llama3.1"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
});
