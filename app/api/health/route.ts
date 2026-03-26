import { NextResponse } from "next/server";

import { getAppConfig, getRequiredServerConfig } from "@/lib/config";
import { getSupabaseAdminClient } from "@/lib/supabase/admin";

export const runtime = "nodejs";

export async function GET() {
  const config = getAppConfig();
  const missing = [
    ["OPENAI_API_KEY", config.OPENAI_API_KEY],
    ["SUPABASE_URL", config.SUPABASE_URL],
    ["SUPABASE_SERVICE_ROLE_KEY", config.SUPABASE_SERVICE_ROLE_KEY],
  ]
    .filter(([, value]) => !value)
    .map(([key]) => key);

  if (missing.length > 0) {
    return NextResponse.json(
      {
        ok: false,
        message: "필수 환경변수가 비어 있습니다.",
        missing,
      },
      { status: 500 },
    );
  }

  try {
    const required = getRequiredServerConfig();
    const client = getSupabaseAdminClient();
    const { count, error } = await client
      .from(required.VECTOR_TABLE_NAME)
      .select("*", { count: "exact", head: true });

    if (error) {
      throw error;
    }

    return NextResponse.json({
      ok: true,
      table: required.VECTOR_TABLE_NAME,
      indexed_documents: count ?? 0,
      model: required.OPENAI_CHAT_MODEL,
      embedding_model: required.OPENAI_EMBEDDING_MODEL,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "헬스체크 중 알 수 없는 오류가 발생했습니다.";

    return NextResponse.json(
      {
        ok: false,
        message,
      },
      { status: 500 },
    );
  }
}
