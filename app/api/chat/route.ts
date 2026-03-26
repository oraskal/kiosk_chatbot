import { NextResponse } from "next/server";
import { z } from "zod";

import { analyzeSupportIssue } from "@/lib/rag/workflow";

export const runtime = "nodejs";

const requestSchema = z.object({
  message: z.string().trim().min(1, "문제상황을 입력해주세요."),
  topK: z.number().int().min(1).max(20).optional(),
});

export async function POST(request: Request) {
  try {
    const json = await request.json();
    const { message, topK } = requestSchema.parse(json);
    const result = await analyzeSupportIssue(message, topK);

    return NextResponse.json(result);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.issues[0]?.message || "입력값이 올바르지 않습니다." }, { status: 400 });
    }

    const message =
      error instanceof Error ? error.message : "챗봇 응답을 생성하는 중 오류가 발생했습니다.";

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
