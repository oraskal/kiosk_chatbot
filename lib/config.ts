import { z } from "zod";

const appConfigSchema = z.object({
  OPENAI_API_KEY: z.string().optional(),
  OPENAI_CHAT_MODEL: z.string().default("gpt-4.1-mini"),
  OPENAI_EMBEDDING_MODEL: z.string().default("text-embedding-3-small"),
  SUPABASE_URL: z.string().optional(),
  SUPABASE_SERVICE_ROLE_KEY: z.string().optional(),
  VECTOR_TABLE_NAME: z.string().default("support_case_documents"),
  VECTOR_QUERY_NAME: z.string().default("match_support_case_documents"),
  CSV_SOURCE_DIR: z.string().default("data/input"),
  RAG_TOP_K: z.coerce.number().int().min(1).max(20).default(6),
  HIGH_CONFIDENCE_THRESHOLD: z.coerce.number().min(0).max(1).default(0.78),
  LOW_CONFIDENCE_THRESHOLD: z.coerce.number().min(0).max(1).default(0.58),
  NEXT_PUBLIC_APP_NAME: z.string().default("유비케어 상담지원 챗봇"),
});

export type AppConfig = z.infer<typeof appConfigSchema>;
export type RequiredServerConfig = AppConfig & {
  OPENAI_API_KEY: string;
  SUPABASE_URL: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
};

export function getAppConfig(): AppConfig {
  return appConfigSchema.parse(process.env);
}

export function getRequiredServerConfig(): RequiredServerConfig {
  const config = getAppConfig();
  const missing = [
    ["OPENAI_API_KEY", config.OPENAI_API_KEY],
    ["SUPABASE_URL", config.SUPABASE_URL],
    ["SUPABASE_SERVICE_ROLE_KEY", config.SUPABASE_SERVICE_ROLE_KEY],
  ]
    .filter(([, value]) => !value)
    .map(([key]) => key);

  if (missing.length > 0) {
    throw new Error(`필수 환경변수가 비어 있습니다: ${missing.join(", ")}`);
  }

  return {
    ...config,
    OPENAI_API_KEY: config.OPENAI_API_KEY!,
    SUPABASE_URL: config.SUPABASE_URL!,
    SUPABASE_SERVICE_ROLE_KEY: config.SUPABASE_SERVICE_ROLE_KEY!,
  };
}
