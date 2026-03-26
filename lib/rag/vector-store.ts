import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";

import { getRequiredServerConfig } from "@/lib/config";
import { getSupabaseAdminClient } from "@/lib/supabase/admin";

export function createEmbeddings() {
  const config = getRequiredServerConfig();

  return new OpenAIEmbeddings({
    apiKey: config.OPENAI_API_KEY,
    model: config.OPENAI_EMBEDDING_MODEL,
  });
}

export function getSupabaseVectorStore() {
  const config = getRequiredServerConfig();
  const client = getSupabaseAdminClient();

  return new SupabaseVectorStore(createEmbeddings(), {
    client,
    tableName: config.VECTOR_TABLE_NAME,
    queryName: config.VECTOR_QUERY_NAME,
  });
}

export async function clearVectorTable() {
  const client = getSupabaseAdminClient();
  const config = getRequiredServerConfig();

  const { error } = await client.from(config.VECTOR_TABLE_NAME).delete().gte("id", 0);

  if (error) {
    throw new Error(`기존 벡터 문서를 비우지 못했습니다: ${error.message}`);
  }
}
