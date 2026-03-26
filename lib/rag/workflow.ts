import { Document } from "@langchain/core/documents";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

import { getRequiredServerConfig } from "@/lib/config";
import { getSupabaseVectorStore } from "@/lib/rag/vector-store";
import type { ChatApiResponse, ConfidenceLevel, SimilarCaseSummary } from "@/lib/types";

const responseSchema = z.object({
  suspected_causes: z.array(z.string()).min(1).max(4),
  checks: z.array(z.string()).min(2).max(6),
  next_actions: z.array(z.string()).min(2).max(6),
  customer_reply_draft: z.string().min(1),
});

type RetrievedMatch = {
  similarity: number;
  summary: SimilarCaseSummary;
  customerReplyReference: string;
  latestActionReference: string;
  qualityTier: string;
};

function determineConfidenceLevel(score: number | null, highThreshold: number, lowThreshold: number): ConfidenceLevel {
  if (score === null || score < lowThreshold) {
    return "low";
  }

  if (score >= highThreshold) {
    return "high";
  }

  return "medium";
}

function buildConfidenceNote(level: ConfidenceLevel, score: number | null, count: number) {
  if (count === 0 || score === null) {
    return "검색된 사례가 없어 일반적인 확인 포인트 중심으로 안내했습니다.";
  }

  if (level === "high") {
    return `상위 ${count}건의 유사 사례가 비교적 잘 맞았습니다. 최고 유사도 참고값은 ${score.toFixed(2)}입니다.`;
  }

  if (level === "medium") {
    return `일부 유사 사례가 있어 참고 가능한 방향은 제시했지만, 최고 유사도 참고값은 ${score.toFixed(2)}로 추가 확인이 필요합니다.`;
  }

  return `충분히 가까운 사례가 적어 단정하지 않고 확인 포인트 중심으로 정리했습니다. 최고 유사도 참고값은 ${score.toFixed(2)}입니다.`;
}

function buildMostSimilarCaseSummary(match: RetrievedMatch | undefined) {
  if (!match) {
    return "";
  }

  const parts = [
    `증상: ${match.summary.problem_summary || "기록 없음"}`,
    `원인: ${match.summary.root_cause || "기록 없음"}`,
    `조치: ${match.summary.resolution_action || "기록 없음"}`,
    `결과: ${match.summary.resolution_result || "기록 없음"}`,
  ];

  return parts.join(" / ");
}

function formatRetrievedCases(matches: RetrievedMatch[]) {
  if (matches.length === 0) {
    return "검색된 참고 사례 없음";
  }

  return matches
    .map(
      (match, index) =>
        [
          `[사례 ${index + 1}]`,
          `유사도: ${match.similarity.toFixed(2)}`,
          `증상 요약: ${match.summary.problem_summary || "기록 없음"}`,
          `실제 원인: ${match.summary.root_cause || "기록 없음"}`,
          `실제 해결 조치: ${match.summary.resolution_action || "기록 없음"}`,
          `처리 결과: ${match.summary.resolution_result || "기록 없음"}`,
          `병원 안내 참고문안: ${match.customerReplyReference || "기록 없음"}`,
          `최신 조치 참고: ${match.latestActionReference || "기록 없음"}`,
          `품질 계층: ${match.qualityTier}`,
        ].join("\n"),
    )
    .join("\n\n");
}

function toRetrievedMatches(input: Array<[Document, number]>) {
  return input.map(([document, similarity]) => {
    const metadata = document.metadata as Record<string, unknown>;

    return {
      similarity,
      customerReplyReference: String(metadata.customer_reply_reference ?? ""),
      latestActionReference: String(metadata.latest_action_reference ?? ""),
      qualityTier: String(metadata.quality_tier ?? ""),
      summary: {
        case_key: String(metadata.case_key ?? ""),
        clinic_name: String(metadata.clinic_name ?? ""),
        issue_subtype_label: String(metadata.issue_subtype_label ?? ""),
        problem_summary: String(metadata.problem_summary ?? ""),
        root_cause: String(metadata.root_cause ?? ""),
        resolution_action: String(metadata.resolution_action ?? ""),
        resolution_result: String(metadata.resolution_result ?? ""),
        similarity_score: similarity,
      },
    } satisfies RetrievedMatch;
  });
}

function createChatModel() {
  const config = getRequiredServerConfig();

  return new ChatOpenAI({
    apiKey: config.OPENAI_API_KEY,
    model: config.OPENAI_CHAT_MODEL,
    temperature: 0.2,
  });
}

export async function analyzeSupportIssue(message: string, topK?: number): Promise<ChatApiResponse> {
  const config = getRequiredServerConfig();
  const vectorStore = getSupabaseVectorStore();
  const searchResults = await vectorStore.similaritySearchWithScore(message, topK ?? config.RAG_TOP_K);
  const matches = toRetrievedMatches(searchResults);
  const topSimilarity = matches[0]?.similarity ?? null;
  const confidenceLevel = determineConfidenceLevel(
    topSimilarity,
    config.HIGH_CONFIDENCE_THRESHOLD,
    config.LOW_CONFIDENCE_THRESHOLD,
  );

  const llm = createChatModel().withStructuredOutput(responseSchema, {
    name: "support_case_response",
    strict: true,
  });

  const structured = await llm.invoke([
    new SystemMessage(
      [
        "너는 유비케어 병원고객팀 상담사용 내부 지원 챗봇이다.",
        "검색된 사례는 참고자료이지 절대적인 정답이 아니다.",
        "현재 문제상황과 가장 가까운 패턴을 찾아 의심 원인, 우선 확인사항, 권장 대응 방향을 정리하라.",
        "근거가 약하면 단정하지 말고 추가 확인 포인트를 먼저 제시하라.",
        "gt_customer_reply 또는 유사한 안내 문안이 있더라도 그대로 복사하지 말고 현재 문의 상황에 맞게 자연스럽게 다듬어라.",
        "병원 안내용 답변 초안은 상담사가 바로 읽어줄 수 있게 간결하고 공손한 한국어로 작성하라.",
        "불필요한 장문 설명은 피하고 실무에 바로 쓰일 문장만 남겨라.",
      ].join("\n"),
    ),
    new HumanMessage(
      [
        `현재 문의:\n${message}`,
        "",
        `검색된 유사 사례:\n${formatRetrievedCases(matches.slice(0, 4))}`,
        "",
        `검색 신뢰도 등급: ${confidenceLevel}`,
        "출력에는 사례를 그대로 복붙하지 말고, 현재 문의 맥락에 맞는 실무형 답변만 담아라.",
      ].join("\n"),
    ),
  ]);

  const similarCases = matches
    .filter((match) => match.similarity >= config.LOW_CONFIDENCE_THRESHOLD)
    .slice(0, 3)
    .map((match) => match.summary);

  return {
    ...structured,
    confidence_level: confidenceLevel,
    confidence_note: buildConfidenceNote(confidenceLevel, topSimilarity, similarCases.length),
    similar_case_count: similarCases.length,
    top_similarity: topSimilarity,
    most_similar_case_summary: buildMostSimilarCaseSummary(matches[0]),
    similar_cases: similarCases,
    fallback_used: confidenceLevel === "low",
  };
}
