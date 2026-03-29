import { readFile } from "node:fs/promises";
import path from "node:path";

import { Document } from "@langchain/core/documents";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

import { getRequiredServerConfig } from "@/lib/config";
import { getSupabaseVectorStore } from "@/lib/rag/vector-store";
import type { ChatApiResponse, ConfidenceLevel, QueryMode, SimilarCaseSummary } from "@/lib/types";

const responseSchema = z.object({
  suspected_causes: z.array(z.string()).min(0).max(4),
  checks: z.array(z.string()).min(1).max(6),
  next_actions: z.array(z.string()).min(1).max(6),
  guide_overview: z.string(),
  guide_steps: z.array(z.string()).max(8),
});

type RetrievedMatch = {
  sourceType: "case";
  similarity: number;
  summary: SimilarCaseSummary;
  customerReplyReference: string;
  latestActionReference: string;
  qualityTier: string;
};

type RetrievedGuideSnippet = {
  sourceType: "guide";
  similarity: number;
  sectionTitle: string;
  content: string;
  sourceFile: string;
};

type GuideSection = {
  title: string;
  body: string;
  fullText: string;
};

const queryModeSchema = z.object({
  mode: z.enum(["incident", "guide"]),
});

function inferQueryModeFallback(message: string): QueryMode {
  const text = message.toLowerCase();
  const incidentSignals = [
    "오류",
    "에러",
    "오작동",
    "고장",
    "문제",
    "실패",
    "불가",
    "안됨",
    "안 돼",
    "안됩니다",
    "안돼요",
    "멈춤",
    "먹통",
    "출력 안",
    "연결 안",
    "not working",
    "error",
    "fail",
  ];
  const guideSignals = [
    "설정",
    "방법",
    "어떻게",
    "사용법",
    "기능",
    "절차",
    "순서",
    "메뉴",
    "옵션",
    "기준",
    "차이",
    "가능",
    "어디서",
  ];

  const hasIncidentSignal = incidentSignals.some((signal) => text.includes(signal));
  const hasGuideSignal = guideSignals.some((signal) => text.includes(signal));

  if (hasIncidentSignal) {
    return "incident";
  }

  if (hasGuideSignal) {
    return "guide";
  }

  return "incident";
}

async function classifyQueryMode(message: string): Promise<QueryMode> {
  try {
    const classifier = new ChatOpenAI({
      apiKey: getRequiredServerConfig().OPENAI_API_KEY,
      model: "gpt-5.1",
    }).withStructuredOutput(queryModeSchema, {
      name: "support_query_mode",
      strict: true,
    });

    const classified = await classifier.invoke([
      new SystemMessage(
        [
          "너는 상담 질문의 의도를 분류하는 라우터다.",
          "incident: 장애/오류/오작동/실패 대응이 필요한 문의.",
          "guide: 설정 방법/기능 설명/절차 안내 중심 문의.",
          "반드시 mode 필드만 반환하라.",
        ].join("\n"),
      ),
      new HumanMessage(`문의: ${message}`),
    ]);

    return classified.mode;
  } catch {
    return inferQueryModeFallback(message);
  }
}

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

function buildGuideConfidenceNote(level: ConfidenceLevel, score: number | null, count: number) {
  if (count === 0 || score === null) {
    return "관련 기준 가이드 검색 결과가 약해, 확인 가능한 범위에서 일반 절차 중심으로 안내했습니다.";
  }

  if (level === "high") {
    return `기준 가이드 ${count}건이 질문과 잘 맞았습니다. 최고 유사도 참고값은 ${score.toFixed(2)}입니다.`;
  }

  if (level === "medium") {
    return `기준 가이드 일부가 질문과 맞아 참고 절차를 제시했습니다. 최고 유사도 참고값은 ${score.toFixed(2)}입니다.`;
  }

  return `가이드 근거가 충분히 가깝지 않아 단정하지 않고 기본 절차 중심으로 정리했습니다. 최고 유사도 참고값은 ${score.toFixed(2)}입니다.`;
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
          `최신 조치 참고: ${match.latestActionReference || "기록 없음"}`,
          `품질 계층: ${match.qualityTier}`,
        ].join("\n"),
    )
    .join("\n\n");
}

function formatRetrievedGuideSnippets(snippets: RetrievedGuideSnippet[]) {
  if (snippets.length === 0) {
    return "검색된 기준 가이드 문서 없음";
  }

  return snippets
    .map((snippet, index) =>
      [
        `[가이드 ${index + 1}]`,
        `유사도: ${snippet.similarity.toFixed(2)}`,
        `섹션: ${snippet.sectionTitle || "제목 없음"}`,
        `내용:\n${snippet.content}`,
      ].join("\n"),
    )
    .join("\n\n");
}

function normalizeBulletText(text: string) {
  return text
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[-*]\s+/, "")
    .replace(/^(\d+[\)\].:-]\s*)+/, "")
    .replace(/(\.\.\.|…)\s*$/, "")
    .trim();
}

function compactList(items: string[], maxItems: number) {
  const seen = new Set<string>();
  const output: string[] = [];

  for (const item of items) {
    const compacted = normalizeBulletText(item);

    if (!compacted || seen.has(compacted)) {
      continue;
    }

    seen.add(compacted);
    output.push(compacted);

    if (output.length >= maxItems) {
      break;
    }
  }

  return output;
}

async function loadBaselineGuideSections() {
  const guidePath = path.resolve(process.cwd(), "data/input/kiosk_baseline_guide.md");
  const text = await readFile(guidePath, "utf-8");
  const normalized = text.replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  const sections: GuideSection[] = [];

  let currentTitle = "가이드 개요";
  let currentLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("## ")) {
      const body = currentLines.join("\n").trim();
      if (body) {
        sections.push({
          title: currentTitle,
          body,
          fullText: `## ${currentTitle}\n\n${body}`,
        });
      }
      currentTitle = line.replace(/^##\s+/, "").trim() || "제목 없음";
      currentLines = [];
      continue;
    }

    currentLines.push(line);
  }

  const finalBody = currentLines.join("\n").trim();
  if (finalBody) {
    sections.push({
      title: currentTitle,
      body: finalBody,
      fullText: `## ${currentTitle}\n\n${finalBody}`,
    });
  }

  return sections;
}

function selectGuideSectionsByQuery(sections: GuideSection[], query: string) {
  const normalizedQuery = query.toLowerCase();
  const tokens = normalizedQuery
    .split(/[\s/,:()>\-[\]]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 1);
  const wantsFull = normalizedQuery.includes("풀");
  const wantsHalf = normalizedQuery.includes("하프");

  const scored = sections
    .map((section) => {
      const haystack = `${section.title}\n${section.body}`.toLowerCase();
      let score = 0;

      for (const token of tokens) {
        if (section.title.toLowerCase().includes(token)) {
          score += 8;
        }
        if (haystack.includes(token)) {
          score += 2;
        }
      }

      if (normalizedQuery.includes("프린터") && haystack.includes("프린터")) {
        score += 6;
      }
      if (normalizedQuery.includes("네트워크") && haystack.includes("네트워크")) {
        score += 6;
      }
      if (normalizedQuery.includes("ip") && haystack.includes("ip")) {
        score += 6;
      }
      if (wantsFull && haystack.includes("풀 키오스크")) {
        score += 10;
      }
      if (wantsHalf && haystack.includes("하프 키오스크")) {
        score += 10;
      }
      if (wantsFull && haystack.includes("하프 키오스크")) {
        score -= 8;
      }
      if (wantsHalf && haystack.includes("풀 키오스크")) {
        score -= 8;
      }

      return { section, score };
    })
    .sort((a, b) => b.score - a.score);

  const bestScore = scored[0]?.score ?? 0;
  const selected = scored
    .filter((item) => item.score >= Math.max(6, bestScore - 4))
    .slice(0, 2)
    .map((item) => item.section);

  return selected.length > 0 ? selected : scored.slice(0, 1).map((item) => item.section);
}

function splitRetrievedDocuments(input: Array<[Document, number]>) {
  const caseMatches: RetrievedMatch[] = [];
  const guideSnippets: RetrievedGuideSnippet[] = [];

  input.forEach(([document, similarity]) => {
    const metadata = document.metadata as Record<string, unknown>;
    const sourceType = String(metadata.source_type ?? "support_case");

    if (sourceType === "baseline_guide") {
      guideSnippets.push({
        sourceType: "guide",
        similarity,
        sectionTitle: String(metadata.guide_section_title ?? ""),
        sourceFile: String(metadata.source_file ?? ""),
        content: document.pageContent,
      });
      return;
    }

    caseMatches.push({
      sourceType: "case",
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
    });
  });

  return {
    caseMatches,
    guideSnippets,
  };
}

function createChatModel() {
  const config = getRequiredServerConfig();

  return new ChatOpenAI({
    apiKey: config.OPENAI_API_KEY,
    model: config.OPENAI_CHAT_MODEL,
  });
}

export async function analyzeSupportIssue(message: string, topK?: number): Promise<ChatApiResponse> {
  const config = getRequiredServerConfig();
  const vectorStore = getSupabaseVectorStore();
  const queryMode = await classifyQueryMode(message);
  const baseTopK = topK ?? config.RAG_TOP_K;
  const searchTopK = queryMode === "guide" ? Math.max(baseTopK * 3, 18) : baseTopK;
  const searchResults = await vectorStore.similaritySearchWithScore(message, searchTopK);
  const { caseMatches, guideSnippets } = splitRetrievedDocuments(searchResults);
  const topSimilarity =
    queryMode === "incident"
      ? caseMatches[0]?.similarity ?? guideSnippets[0]?.similarity ?? null
      : guideSnippets[0]?.similarity ?? caseMatches[0]?.similarity ?? null;
  const confidenceLevel = determineConfidenceLevel(
    topSimilarity,
    config.HIGH_CONFIDENCE_THRESHOLD,
    config.LOW_CONFIDENCE_THRESHOLD,
  );
  const caseContext = queryMode === "incident" ? caseMatches.slice(0, 4) : [];
  const guideContext = queryMode === "incident" ? guideSnippets.slice(0, 3) : guideSnippets.slice(0, 8);

  if (queryMode === "guide") {
    const sections = await loadBaselineGuideSections();
    const matchedSections = selectGuideSectionsByQuery(sections, message);
    const rawGuideText = matchedSections.map((section) => section.fullText).join("\n\n---\n\n");
    const guideExcerptLines = rawGuideText.split("\n");
    const topGuideSimilarity = guideContext[0]?.similarity ?? 0.8;
    const guideConfidenceLevel = determineConfidenceLevel(
      topGuideSimilarity,
      config.HIGH_CONFIDENCE_THRESHOLD,
      config.LOW_CONFIDENCE_THRESHOLD,
    );
    const guideOverview =
      matchedSections.length > 0
        ? `가이드 원문 섹션: ${matchedSections.map((section) => section.title).join(" / ")}`
        : "가이드 원문 발췌";

    return {
      query_mode: "guide",
      suspected_causes: [],
      checks: [],
      next_actions: [],
      guide_overview: guideOverview,
      guide_steps: guideExcerptLines,
      confidence_level: guideConfidenceLevel,
      confidence_note: buildGuideConfidenceNote(guideConfidenceLevel, topGuideSimilarity, matchedSections.length),
      similar_case_count: 0,
      top_similarity: topGuideSimilarity,
      most_similar_case_summary: "",
      similar_cases: [],
      fallback_used: guideConfidenceLevel === "low",
    };
  }

  const llm = createChatModel().withStructuredOutput(responseSchema, {
    name: "support_case_response",
    strict: true,
  });

  const structured = await llm.invoke([
    new SystemMessage(
      [
        "너는 유비케어 병원고객팀 상담사용 내부 지원 챗봇이다.",
        "질문 맥락을 먼저 파악해 장애 대응 질문인지, 단순 기능/설정 안내 질문인지 구분하라.",
        "검색된 사례와 가이드는 참고자료이지 절대적인 정답이 아니다.",
        "키오스크 메인 기준 가이드(Baseline)는 우선 확인사항/권장 대응 방향의 기준값으로 적극 반영하라.",
        "모드가 incident(장애/오류)라면 현재 문제상황과 가장 가까운 패턴을 찾아 의심 원인, 우선 확인사항, 권장 대응 방향을 정리하라.",
        "모드가 guide(설정/기능 질문)라면 원인 추정을 억지로 만들지 말고 baseline 가이드 기준의 절차/설정 포인트를 중심으로 작성하라.",
        "guide 모드일 때는 guide_overview(핵심 요약)와 guide_steps(실행 순서)를 채워라.",
        "incident 모드일 때도 guide_overview는 빈 문자열(\"\"), guide_steps는 빈 배열([])로 반드시 채워라.",
        "guide 모드일 때 suspected_causes는 빈 배열이어도 된다.",
        "guide 모드에서는 가이드 원문의 구체 값(IP, 포트명, 메뉴 경로, 모델명)을 우선 반영하라.",
        "가이드 원문에 수치/식별값이 있으면 추상적으로 바꾸지 말고 그대로 작성하라.",
        "guide 모드에서는 checks, next_actions를 빈 배열로 출력하라.",
        "특히 원인/확인사항/대응 방향 작성 시, 기준 가이드에서 확인해야 할 항목을 빠뜨리지 마라.",
        "모든 출력은 짧고 건조하게 작성하라. 미사여구/배경설명/권유문 금지.",
        "각 항목은 핵심만 1문장으로 작성하라.",
        "근거가 약하면 단정하지 말고 추가 확인 포인트를 먼저 제시하라.",
        "불필요한 장문 설명은 피하고 실무에 바로 쓰일 문장만 남겨라.",
      ].join("\n"),
    ),
    new HumanMessage(
      [
        `현재 문의:\n${message}`,
        `질의 모드: ${queryMode}`,
        "",
        `검색된 유사 사례:\n${formatRetrievedCases(caseContext)}`,
        "",
        `검색된 키오스크 기준 가이드:\n${formatRetrievedGuideSnippets(guideContext)}`,
        "",
        `검색 신뢰도 등급: ${confidenceLevel}`,
        "guide 모드에서는 장애 원인 단정 표현을 피하고, 기준 절차/설정 포인트를 명확하게 안내하라.",
        "incident 모드에서는 증상과 근거를 연결해 원인 가설과 확인 포인트를 우선 제시하라.",
        "guide_overview에는 한눈에 이해되는 핵심 가이드 요약을 2~4문장으로 작성하라.",
        "guide_steps에는 실제 클릭/설정 순서를 2~6개로 작성하라.",
        "guide_steps에는 가능한 경우 IP/포트/메뉴 경로 같은 구체값을 포함하라.",
        "긴 설명 대신 값/경로/단계만 남겨라.",
        "출력에는 사례/가이드를 그대로 복붙하지 말고, 현재 문의 맥락에 맞는 실무형 답변만 담아라.",
      ].join("\n"),
    ),
  ]);

  const compactedCauses = compactList(structured.suspected_causes, 4);
  const compactedChecks = compactList(structured.checks, 6);
  const compactedActions = compactList(structured.next_actions, 6);
  const similarCases =
    queryMode === "incident"
      ? caseMatches
          .filter((match) => match.similarity >= config.LOW_CONFIDENCE_THRESHOLD)
          .slice(0, 3)
          .map((match) => match.summary)
      : [];
  const confidenceNote =
    queryMode === "incident"
      ? buildConfidenceNote(confidenceLevel, topSimilarity, similarCases.length)
      : buildGuideConfidenceNote(confidenceLevel, topSimilarity, guideContext.length);

  return {
    ...structured,
    query_mode: queryMode,
    suspected_causes: compactedCauses,
    checks: compactedChecks,
    next_actions: compactedActions,
    guide_overview: "",
    guide_steps: [],
    confidence_level: confidenceLevel,
    confidence_note: confidenceNote,
    similar_case_count: similarCases.length,
    top_similarity: topSimilarity,
    most_similar_case_summary: queryMode === "incident" ? buildMostSimilarCaseSummary(caseMatches[0]) : "",
    similar_cases: similarCases,
    fallback_used: confidenceLevel === "low",
  };
}
