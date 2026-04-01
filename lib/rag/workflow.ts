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
  provenance?: "case_history";
  similarity: number;
  summary: SimilarCaseSummary;
  customerReplyReference: string;
  latestActionReference: string;
  qualityTier: string;
};

type RetrievedGuideSnippet = {
  sourceType: "guide";
  provenance?: "baseline_doc" | "official_guide";
  similarity: number;
  sectionTitle: string;
  content: string;
  sourceFile: string;
};

type RelatedGuidePreview = {
  show_related_guide: boolean;
  related_guide_title: string;
  related_guide_excerpt: string;
  related_guide_reason: string;
};

type SourceGrounding = "baseline_doc" | "official_guide" | "case_history" | "inferred";
type AnswerSectionKey = "suspected_causes" | "checks" | "next_actions";

type QuerySignals = {
  payment: boolean;
  printer: boolean;
  peripheral: boolean;
  runtime: boolean;
  screen: boolean;
  reception: boolean;
  fullKiosk: boolean;
  halfKiosk: boolean;
};

type GuideEvidence = {
  text: string;
  normalized: string;
  terms: string[];
  score: number;
  sectionTitle: string;
  provenance: "baseline_doc" | "official_guide";
};

type SectionCandidate = {
  text: string;
  normalized: string;
  provenance: SourceGrounding;
  supportScore: number;
  genericRisk: boolean;
  synthetic: boolean;
};

type GovernedIncidentResponse = {
  suspectedCauses: string[];
  checks: string[];
  nextActions: string[];
  hospitalReply: string;
  fallbackUsed: boolean;
};

const queryModeSchema = z.object({
  mode: z.enum(["incident", "guide"]),
});

const PROVENANCE_PRIORITY: Record<SourceGrounding, number> = {
  baseline_doc: 0,
  official_guide: 1,
  case_history: 2,
  inferred: 3,
};

const PROVENANCE_LABEL: Record<SourceGrounding, string> = {
  baseline_doc: "[기준 문서]",
  official_guide: "[기준 문서]",
  case_history: "[유사 사례]",
  inferred: "[추가 확인]",
};

const MAX_INFERRED_PER_SECTION: Record<AnswerSectionKey, number> = {
  suspected_causes: 1,
  checks: 1,
  next_actions: 1,
};

const SECTION_MAX_ITEMS: Record<AnswerSectionKey, number> = {
  suspected_causes: 4,
  checks: 6,
  next_actions: 6,
};

const GENERIC_TOP_BLOCKLIST = [
  /장치\s*관리자/i,
  /usb\s*인식|인식\s*불량|인식\s*오류/i,
  /드라이버(\s*재설치|\s*업데이트|\s*충돌|\s*문제)?/i,
  /방화벽/i,
  /포트\s*충돌/i,
  /네트워크\s*(문제|장애|오류|불량|불안정)/i,
  /케이블\s*(불량|이상|재결속|재연결)/i,
  /핑\s*테스트/i,
  /윈도우에서\s*확인/i,
  /운영체제|os\s*문제/i,
  /정상\s*인식되는지/i,
];

const TERM_STOP_WORDS = new Set([
  "기준",
  "문서",
  "확인",
  "설정",
  "항목",
  "우선",
  "관련",
  "현재",
  "추가",
  "가이드",
  "절차",
  "대응",
  "방향",
  "원인",
  "복구",
  "원복",
  "다시",
  "기본",
  "값",
  "기본값",
  "기본설정",
  "합니다",
  "해야",
]);

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

function buildConfidenceNote(level: ConfidenceLevel, score: number | null, groundedGuideCount: number, caseCount: number) {
  if (groundedGuideCount === 0 || score === null) {
    return "기준 문서 근거가 약해 문서 기준으로 확인 가능한 항목만 보수적으로 정리했습니다.";
  }

  if (level === "high") {
    return `기준 문서 ${groundedGuideCount}건을 우선 근거로 사용했습니다. 사례 참고 ${caseCount}건, 최고 유사도 ${score.toFixed(2)}입니다.`;
  }

  if (level === "medium") {
    return `기준 문서 근거를 우선 반영했고, 사례 참고는 보조로만 사용했습니다. 최고 유사도 ${score.toFixed(2)}입니다.`;
  }

  return `기준 문서 근거가 충분히 강하지 않아 추정은 줄이고 확인 가능한 설정 항목만 남겼습니다. 최고 유사도 ${score.toFixed(2)}입니다.`;
}

function buildGuideConfidenceNote(level: ConfidenceLevel, score: number | null, count: number) {
  if (count === 0 || score === null) {
    return "현재 확보된 기준 문서에서 직접 대응되는 메뉴나 절차를 찾지 못했습니다.";
  }

  if (level === "high") {
    return `기준 문서 ${count}건이 질문과 잘 맞았습니다. 최고 유사도 ${score.toFixed(2)}입니다.`;
  }

  if (level === "medium") {
    return `기준 문서 일부가 질문과 맞아 해당 메뉴와 설정값 위주로 정리했습니다. 최고 유사도 ${score.toFixed(2)}입니다.`;
  }

  return `기준 문서 근거가 약해 직접 매핑되는 메뉴와 설정값만 제한적으로 안내했습니다. 최고 유사도 ${score.toFixed(2)}입니다.`;
}

function buildMostSimilarCaseSummary(match: RetrievedMatch | undefined, hasGuideGrounding: boolean) {
  if (!match) {
    return "";
  }

  const prefix = hasGuideGrounding ? "사례 참고(기준 문서 우선):" : "사례 참고:";
  const parts = [
    `증상 ${match.summary.problem_summary || "기록 없음"}`,
    `원인 ${match.summary.root_cause || "기록 없음"}`,
    `조치 ${match.summary.resolution_action || "기록 없음"}`,
    `결과 ${match.summary.resolution_result || "기록 없음"}`,
  ];

  return `${prefix} ${parts.join(" / ")}`;
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
        `문서 우선순위: ${snippet.provenance ?? "baseline_doc"}`,
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
    .replace(/^\[(기준 문서|유사 사례|추가 확인)\]\s*/u, "")
    .replace(/(\.\.\.|…)\s*$/, "")
    .trim();
}

function compactList(items: string[], maxItems: number) {
  const seen = new Set<string>();
  const output: string[] = [];

  for (const item of items) {
    const compacted = item.trim();
    const dedupeKey = normalizeBulletText(item);

    if (!compacted || seen.has(dedupeKey)) {
      continue;
    }

    seen.add(dedupeKey);
    output.push(compacted);

    if (output.length >= maxItems) {
      break;
    }
  }

  return output;
}

function detectGuideProvenance(sourceType: string, sourceFile: string) {
  if (sourceType === "baseline_guide" || /baseline|기준/i.test(sourceFile)) {
    return "baseline_doc" as const;
  }

  return "official_guide" as const;
}

function detectDocumentProvenance(document: Document) {
  const metadata = document.metadata as Record<string, unknown>;
  const sourceType = String(metadata.source_type ?? "support_case");
  const sourceFile = String(metadata.source_file ?? "");

  if (sourceType === "support_case") {
    return "case_history" as const;
  }

  return detectGuideProvenance(sourceType, sourceFile);
}

function extractTerms(text: string) {
  return text
    .toLowerCase()
    .split(/[^0-9a-zA-Z가-힣./:\\_-]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 1 && !TERM_STOP_WORDS.has(token));
}

function countOverlap(left: string[], right: string[]) {
  if (left.length === 0 || right.length === 0) {
    return 0;
  }

  const rightSet = new Set(right);
  return [...new Set(left)].filter((item) => rightSet.has(item)).length;
}

function scoreKeywordOverlap(text: string, terms: string[], perHit: number, maxBoost: number) {
  let boost = 0;

  for (const term of terms) {
    if (text.includes(term)) {
      boost += perHit;
    }

    if (boost >= maxBoost) {
      return maxBoost;
    }
  }

  return boost;
}

function buildQuerySignals(message: string): QuerySignals {
  const text = message.toLowerCase();

  return {
    payment: /(결제|카드|승인|ksnet|smartro|vcat|catid|수납)/i.test(text),
    printer: /(프린터|출력|영수증|처방전|인쇄)/i.test(text),
    peripheral: /(스캐너|리더기|바코드|주변기기|단말기|장치|포트|자동설정)/i.test(text),
    runtime: /(실행|시작|프로그램|서비스 사용|자동실행|메인화면|일시정지)/i.test(text),
    screen: /(화면|검은|흰|깨짐|안보임|멈춤|먹통)/i.test(text),
    reception: /(접수|문진|진료실|신규환자)/i.test(text),
    fullKiosk: text.includes("풀"),
    halfKiosk: text.includes("하프"),
  };
}

function createEmptyRelatedGuidePreview(): RelatedGuidePreview {
  return {
    show_related_guide: false,
    related_guide_title: "",
    related_guide_excerpt: "",
    related_guide_reason: "",
  };
}

function getGuideSnippetBody(snippet: RetrievedGuideSnippet) {
  return snippet.content
    .replace(/^문서 유형:.*$/m, "")
    .replace(/^섹션:.*$/m, "")
    .trim();
}

function rerankRetrievedDocuments(input: Array<[Document, number]>, message: string, queryMode: QueryMode) {
  const queryTerms = extractTerms(message);
  const querySignals = buildQuerySignals(message);

  return input
    .map(([document, similarity]) => {
      const provenance = detectDocumentProvenance(document);
      const metadata = document.metadata as Record<string, unknown>;
      const title = String(metadata.guide_section_title ?? metadata.issue_subtype_label ?? "");
      const text = `${title}\n${document.pageContent}`.toLowerCase();
      let weightedScore = similarity * 100;

      weightedScore += provenance === "baseline_doc" ? 42 : provenance === "official_guide" ? 30 : 10;
      weightedScore += queryMode === "incident" ? (provenance === "case_history" ? 6 : 18) : provenance === "case_history" ? -12 : 26;
      weightedScore += scoreKeywordOverlap(text, queryTerms, provenance === "case_history" ? 2 : 3, 32);

      if (querySignals.payment && /(결제|카드|승인|ksnet|smartro|vcat|catid|수납)/i.test(text)) {
        weightedScore += provenance === "case_history" ? 8 : 16;
      }

      if (querySignals.printer && /(프린터|출력|영수증|처방전|sl-m3830nd|checkprinter|172\.25\.123\.99)/i.test(text)) {
        weightedScore += provenance === "case_history" ? 8 : 16;
      }

      if (querySignals.peripheral && /(스캐너|리더기|바코드|주변기기|장치|포트|자동설정|연동)/i.test(text)) {
        weightedScore += provenance === "case_history" ? 6 : 14;
      }

      if (querySignals.runtime && /(서비스 사용|메인화면|테스트 설정|자동실행|플레이어|관리자 모드|일시정지)/i.test(text)) {
        weightedScore += provenance === "case_history" ? 6 : 15;
      }

      if (querySignals.screen && /(화면|플레이어|메인화면|서비스 사용|일시정지)/i.test(text)) {
        weightedScore += provenance === "case_history" ? 5 : 12;
      }

      if (querySignals.reception && /(접수|문진|진료실|신규환자|임시접수)/i.test(text)) {
        weightedScore += provenance === "case_history" ? 6 : 13;
      }

      if (querySignals.fullKiosk && /풀 키오스크/i.test(text)) {
        weightedScore += 8;
      }

      if (querySignals.halfKiosk && /하프 키오스크/i.test(text)) {
        weightedScore += 8;
      }

      if (querySignals.fullKiosk && /하프 키오스크/i.test(text)) {
        weightedScore -= 6;
      }

      if (querySignals.halfKiosk && /풀 키오스크/i.test(text)) {
        weightedScore -= 6;
      }

      return {
        document,
        similarity,
        weightedScore,
      };
    })
    .sort((left, right) => {
      if (right.weightedScore !== left.weightedScore) {
        return right.weightedScore - left.weightedScore;
      }

      return right.similarity - left.similarity;
    })
    .map((item) => [item.document, item.similarity] as [Document, number]);
}

function buildCaseTexts(caseMatches: RetrievedMatch[]) {
  return caseMatches.flatMap((match) => [
    match.summary.problem_summary,
    match.summary.root_cause,
    match.summary.resolution_action,
    match.summary.issue_subtype_label,
  ]);
}

function buildGuideEvidence(guideSnippets: RetrievedGuideSnippet[], message: string, maxItems = 12) {
  const queryTerms = extractTerms(message);
  const evidenceMap = new Map<string, GuideEvidence>();

  for (const snippet of guideSnippets) {
    const body = getGuideSnippetBody(snippet);
    const lines = body
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    for (const line of lines) {
      const normalized = normalizeBulletText(line);

      if (!normalized || normalized.length < 4 || /^(문서 유형|섹션):/i.test(normalized)) {
        continue;
      }

      let score = snippet.similarity * 10;

      if (/^[-*]/.test(line) || /^\d+\./.test(line)) {
        score += 3;
      }

      if (/`[^`]+`/.test(line) || /\b\d{1,3}(?:\.\d{1,3}){3}\b/.test(line) || /ON|OFF/.test(line)) {
        score += 4;
      }

      score += scoreKeywordOverlap(normalized.toLowerCase(), queryTerms, 2, 16);

      const existing = evidenceMap.get(normalized);
      const candidate = {
        text: normalized,
        normalized: normalized.toLowerCase(),
        terms: extractTerms(normalized),
        score,
        sectionTitle: snippet.sectionTitle,
        provenance: snippet.provenance ?? "baseline_doc",
      } satisfies GuideEvidence;

      if (!existing || candidate.score > existing.score) {
        evidenceMap.set(normalized, candidate);
      }
    }
  }

  return [...evidenceMap.values()]
    .sort((left, right) => right.score - left.score)
    .slice(0, maxItems);
}

function scoreSourceSupport(text: string, evidenceText: string, textTerms: string[], evidenceTerms: string[]) {
  const normalizedText = text.toLowerCase();
  const normalizedEvidence = evidenceText.toLowerCase();
  let score = 0;

  if (normalizedEvidence.includes(normalizedText) || normalizedText.includes(normalizedEvidence)) {
    score += 6;
  }

  score += countOverlap(textTerms, evidenceTerms) * 2;

  const literals = normalizedEvidence.match(/`[^`]+`|\b\d{1,3}(?:\.\d{1,3}){3}\b|[a-z]+\.exe|catid/gi) ?? [];
  if (literals.some((literal) => normalizedText.includes(literal.toLowerCase()))) {
    score += 3;
  }

  return score;
}

function isGenericTroubleshooting(text: string, guideCorpus: string) {
  return GENERIC_TOP_BLOCKLIST.some((pattern) => pattern.test(text) && !pattern.test(guideCorpus));
}

function classifyCandidate(
  text: string,
  guideEvidence: GuideEvidence[],
  caseTexts: string[],
  guideCorpus: string,
): SectionCandidate {
  const normalized = normalizeBulletText(text);
  const textTerms = extractTerms(normalized);
  let bestGuideScore = 0;
  let bestGuideProvenance: "baseline_doc" | "official_guide" = "official_guide";
  let bestCaseScore = 0;

  for (const evidence of guideEvidence) {
    const score = scoreSourceSupport(normalized, evidence.text, textTerms, evidence.terms);

    if (score > bestGuideScore) {
      bestGuideScore = score;
      bestGuideProvenance = evidence.provenance;
    }
  }

  for (const caseText of caseTexts) {
    const caseTerms = extractTerms(caseText);
    const score = scoreSourceSupport(normalized, caseText, textTerms, caseTerms);

    if (score > bestCaseScore) {
      bestCaseScore = score;
    }
  }

  const genericRisk = isGenericTroubleshooting(normalized.toLowerCase(), guideCorpus);

  if (bestGuideScore >= 4) {
    return {
      text: normalized,
      normalized: normalized.toLowerCase(),
      provenance: bestGuideProvenance,
      supportScore: bestGuideScore,
      genericRisk,
      synthetic: false,
    };
  }

  if (bestCaseScore >= 4) {
    return {
      text: normalized,
      normalized: normalized.toLowerCase(),
      provenance: "case_history",
      supportScore: bestCaseScore,
      genericRisk,
      synthetic: false,
    };
  }

  return {
    text: normalized,
    normalized: normalized.toLowerCase(),
    provenance: "inferred",
    supportScore: 0,
    genericRisk,
    synthetic: false,
  };
}

function withProvenanceLabel(text: string, provenance: SourceGrounding) {
  return `${PROVENANCE_LABEL[provenance]} ${normalizeBulletText(text)}`.trim();
}

function isPaymentIncidentText(text: string) {
  const normalized = text.toLowerCase();
  const paymentSignals = [
    "카드결제",
    "카드 결제",
    "결제 안",
    "결제불가",
    "결제 불가",
    "결제 오류",
    "결제 실패",
    "카드 승인 실패",
    "승인 요청 실패",
    "카드 인식 불가",
    "카드 인식 안",
    "카드리더기",
    "리더기",
    "멀티패드",
    "결제창",
    "무카드취소",
    "무 카드 취소",
    "수납 직후 취소",
    "승인 취소",
    "van",
    "ksnet",
    "smartro",
    "vcat",
  ];

  return paymentSignals.some((signal) => normalized.includes(signal));
}

function isPaymentIncident(message: string, caseMatches: RetrievedMatch[], guideSnippets: RetrievedGuideSnippet[]) {
  const haystack = [
    message,
    ...caseMatches.flatMap((match) => [
      match.summary.problem_summary,
      match.summary.root_cause,
      match.summary.resolution_action,
      match.summary.issue_subtype_label,
    ]),
    ...guideSnippets.flatMap((snippet) => [snippet.sectionTitle, snippet.content]),
  ]
    .filter(Boolean)
    .join("\n");

  return isPaymentIncidentText(haystack);
}

function buildPaymentPolicyPrompt() {
  return [
    "이 문의가 카드결제/승인 실패/멀티패드 관련 incident라면 결제 도메인 정책을 엄격히 적용하라.",
    "현금 결제, 다른 현금 흐름, 현금 대체 결제를 절대 제안하지 마라.",
    "카드리더기 또는 멀티패드 단독 재부팅을 절대 안내하지 마라.",
    "재부팅이 필요하면 반드시 Windows 포함 키오스크 전체 재부팅만 허용하라.",
    "VAN사 통신망, 카드사 승인망, 프록시/방화벽, 외부 네트워크 경로 문제를 기본 원인이나 유력 가설처럼 제시하지 마라.",
    "결제 incident의 의심 원인은 로컬 baseline 설정 이탈, 키오스크 내부 장치 인식/연결 문제, 로컬 결제 에이전트/드라이버/옵션값 불일치 순으로 우선 제시하라.",
    "우선 확인사항과 권장 대응 방향은 상담사가 통화 중 원격지원으로 바로 확인할 수 있는 항목을 먼저 제시하라.",
    "고객 또는 현장 담당자 조작 요청은 실물 카드 재현 시험이나 전체 기기 재부팅처럼 원격지원으로 대신할 수 없는 경우에만 최소한으로 적어라.",
    "근거가 약하면 과도한 추측 대신 로컬 설정 점검 항목만 좁게 제시하라.",
  ];
}

function rewritePaymentCause(item: string) {
  const normalized = normalizeBulletText(item);

  if (!normalized) {
    return "";
  }

  if (/현금/.test(normalized)) {
    return "";
  }

  if (/(van.?사.*통신망|카드사.*승인망|외부.?망|프록시|방화벽|다른 단말)/i.test(normalized)) {
    return "키오스크 내부 결제 설정과 에이전트 실행 상태가 기본 설정값과 다를 가능성";
  }

  if (/(카드리더기|리더기|멀티패드).*(단독|개별).*(재부팅|재시작)|(카드리더기|리더기|멀티패드).*(재부팅|재시작)/i.test(normalized)) {
    return "키오스크 내부 카드장치 인식 또는 연결 상태가 불안정할 가능성";
  }

  return normalized;
}

function rewritePaymentCheckOrAction(item: string, fallback: string) {
  const normalized = normalizeBulletText(item);

  if (!normalized) {
    return fallback;
  }

  if (/현금/.test(normalized)) {
    return fallback;
  }

  if (/(van.?사.*통신망|카드사.*승인망|외부.?망|프록시|방화벽|다른 단말)/i.test(normalized)) {
    return fallback;
  }

  if (/(카드리더기|리더기|멀티패드).*(단독|개별).*(재부팅|재시작)|(카드리더기|리더기|멀티패드).*(재부팅|재시작)/i.test(normalized)) {
    return "원격지원 중 Windows 포함 키오스크 전체 재부팅 후 결제 에이전트와 장치 인식 상태를 다시 확인합니다.";
  }

  return normalized;
}

function rankPaymentCause(item: string) {
  if (/(기본|세팅|설정|이탈|baseline|van|catid|포트|옵션)/i.test(item)) {
    return 0;
  }

  if (/(장치|인식|연결|접촉|usb|케이블|리더기|멀티패드|카드장치)/i.test(item)) {
    return 1;
  }

  if (/(에이전트|드라이버|프로그램|자동실행|실행 상태)/i.test(item)) {
    return 2;
  }

  return 3;
}

function rankRemoteFirstItem(item: string) {
  if (/(원격지원|원격|상담사|통화 중)/.test(item)) {
    return 0;
  }

  if (/(전체 재부팅|키오스크 전체|windows 포함)/i.test(item)) {
    return 1;
  }

  if (/(현장|실물 카드|카드 승인 재현|물리)/.test(item)) {
    return 2;
  }

  return 0;
}

function sanitizePaymentIncidentResponse(
  suspectedCauses: string[],
  checks: string[],
  nextActions: string[],
) {
  const defaultCauses = [
    "기본 설정값과 다른 결제 설정(VAN, CATID, 포트, 결제 옵션값) 이 반영됐을 가능성",
    "키오스크 내부 카드장치 인식 또는 연결 상태가 불안정할 가능성",
    "결제 에이전트나 드라이버, 자동실행 옵션이 현장 장비와 맞지 않을 가능성",
  ];
  const defaultChecks = [
    "원격지원으로 서비스 사용, 기기 종류, VAN 선택, CATID, 메인화면 연결, 테스트 설정 원복 여부를 확인합니다.",
    "원격지원으로 결제 에이전트 실행 상태와 아이콘, 자동실행, 옵션값이 기본 설정값과 일치하는지 확인합니다.",
    "원격지원으로 결제 프로그램 오류 표시와 결제부 연동 상태를 확인합니다.",
    "필요하면 통화 유지 상태에서 Windows 포함 키오스크 전체 재부팅 후 동일 증상 재현 여부만 다시 확인합니다.",
    "실물 카드 승인 재현은 현장에서만 가능한 경우에 한해 최소 범위로 요청합니다.",
  ];
  const defaultActions = [
    "상담사가 원격지원으로 결제 기본 설정과 에이전트 상태를 먼저 정리한 뒤 재시험 순서를 안내합니다.",
    "설정 이탈이나 에이전트 비정상이 확인되면 기본 설정값에 맞게 복구 후 카드 승인 재시도를 진행합니다.",
    "전체 재부팅 후에도 동일하고 장치 인식 오류가 남으면 키오스크 장치 또는 결제부 점검으로 넘깁니다.",
    "원격지원으로 해결되지 않고 결제부 오류가 반복되면 현장 장치 점검 또는 교체 판단으로 연결합니다.",
  ];

  const sanitizedCauses = compactList(
    [...suspectedCauses.map(rewritePaymentCause), ...defaultCauses]
      .filter(Boolean)
      .sort((left, right) => rankPaymentCause(left) - rankPaymentCause(right)),
    4,
  );
  const sanitizedChecks = compactList(
    [
      ...checks.map((item) =>
        rewritePaymentCheckOrAction(
          item,
          "원격지원으로 키오스크 내부 결제 설정, 에이전트 실행 상태, 장치 인식 상태를 우선 확인합니다.",
        ),
      ),
      ...defaultChecks,
    ]
      .filter(Boolean)
      .sort((left, right) => rankRemoteFirstItem(left) - rankRemoteFirstItem(right)),
    6,
  );
  const sanitizedActions = compactList(
    [
      ...nextActions.map((item) =>
        rewritePaymentCheckOrAction(
          item,
          "상담사가 원격지원으로 확인 가능한 결제 설정과 장치 상태를 먼저 정리한 뒤 필요한 현장 조치를 최소 범위로 안내합니다.",
        ),
      ),
      ...defaultActions,
    ]
      .filter(Boolean)
      .sort((left, right) => rankRemoteFirstItem(left) - rankRemoteFirstItem(right)),
    6,
  );

  return {
    suspectedCauses: sanitizedCauses,
    checks: sanitizedChecks,
    nextActions: sanitizedActions,
  };
}

function buildGuideExcerptKeywords(message: string, answerParts: string[], paymentIncident: boolean) {
  const combined = `${message}\n${answerParts.join("\n")}`.toLowerCase();
  const keywords = new Set<string>();

  if (paymentIncident) {
    ["결제", "카드", "승인", "vcat", "smartro", "ksnet", "van", "catid", "포트", "옵션", "에이전트"].forEach(
      (keyword) => keywords.add(keyword),
    );
  }

  ["프린터", "출력", "접수", "진료실", "음성", "장애인", "메인화면", "서비스 사용", "일시정지"].forEach((keyword) => {
    if (combined.includes(keyword.toLowerCase())) {
      keywords.add(keyword.toLowerCase());
    }
  });

  return [...keywords];
}

function selectRelatedGuideSnippet(
  guideSnippets: RetrievedGuideSnippet[],
  message: string,
  answerParts: string[],
  paymentIncident: boolean,
) {
  if (guideSnippets.length === 0) {
    return undefined;
  }

  const keywords = buildGuideExcerptKeywords(message, answerParts, paymentIncident);

  return guideSnippets
    .map((snippet) => {
      const body = getGuideSnippetBody(snippet).toLowerCase();
      const title = snippet.sectionTitle.toLowerCase();
      let score = snippet.similarity * 100;
      score += (snippet.provenance ?? "baseline_doc") === "baseline_doc" ? 22 : 10;

      for (const keyword of keywords) {
        if (title.includes(keyword)) {
          score += 15;
        }
        if (body.includes(keyword)) {
          score += 6;
        }
      }

      if (paymentIncident && /(결제|ksnet|smartro|vcat|catid|결제 이슈)/i.test(snippet.sectionTitle)) {
        score += 24;
      }

      if (paymentIncident && /(결제|ksnet|smartro|vcat|catid|결제 이슈)/i.test(snippet.content)) {
        score += 16;
      }

      if (/기본 설정 이탈 체크리스트/.test(snippet.sectionTitle)) {
        score += 10;
      }

      return { snippet, score };
    })
    .sort((left, right) => right.score - left.score)[0]?.snippet;
}

function buildFocusedGuideExcerpt(snippet: RetrievedGuideSnippet, message: string, answerParts: string[], paymentIncident: boolean) {
  const body = getGuideSnippetBody(snippet);
  const blocks = body
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
  const keywords = buildGuideExcerptKeywords(message, answerParts, paymentIncident);

  const rankedBlocks = blocks
    .map((block) => {
      const normalizedBlock = block.toLowerCase();
      let score = 0;

      for (const keyword of keywords) {
        if (normalizedBlock.includes(keyword)) {
          score += 5;
        }
      }

      if (/^#+\s/.test(block)) {
        score += 2;
      }

      if (/^- /.test(block) || /^\d+\.\s/.test(block)) {
        score += 1;
      }

      return { block, score };
    })
    .sort((left, right) => right.score - left.score);

  const selectedBlocks = rankedBlocks.filter((item) => item.score > 0).slice(0, 2);
  const excerptSource = (selectedBlocks.length > 0 ? selectedBlocks : rankedBlocks.slice(0, 1))
    .map((item) => item.block)
    .join("\n\n");
  let excerpt = `### ${snippet.sectionTitle}\n\n${excerptSource}`.trim();

  if (excerpt.length > 520) {
    excerpt = `${excerpt.slice(0, 520).trimEnd()}\n...`;
  }

  return excerpt;
}

function buildRelatedGuidePreview(
  message: string,
  suspectedCauses: string[],
  checks: string[],
  nextActions: string[],
  guideSnippets: RetrievedGuideSnippet[],
  paymentIncident: boolean,
  lowThreshold: number,
): RelatedGuidePreview {
  const answerParts = [...suspectedCauses, ...checks, ...nextActions].map((item) => normalizeBulletText(item));
  const hasBaselineSignal = answerParts.some((item) =>
    /(기준 문서|기본 설정|기본값|이탈|원복|포트|서비스 사용|메인화면|에이전트)/.test(item),
  );

  if (!hasBaselineSignal) {
    return createEmptyRelatedGuidePreview();
  }

  const selectedSnippet = selectRelatedGuideSnippet(guideSnippets, message, answerParts, paymentIncident);

  if (!selectedSnippet) {
    return createEmptyRelatedGuidePreview();
  }

  if (
    selectedSnippet.similarity < lowThreshold &&
    !paymentIncident &&
    (selectedSnippet.provenance ?? "baseline_doc") !== "baseline_doc"
  ) {
    return createEmptyRelatedGuidePreview();
  }

  return {
    show_related_guide: true,
    related_guide_title: selectedSnippet.sectionTitle || "관련 기준 가이드",
    related_guide_excerpt: buildFocusedGuideExcerpt(selectedSnippet, message, answerParts, paymentIncident),
    related_guide_reason:
      (selectedSnippet.provenance ?? "baseline_doc") === "baseline_doc"
        ? "답변 상단에 배치한 기준 문서 항목을 바로 대조할 때 참고합니다."
        : paymentIncident
          ? "결제 기본 설정과 에이전트/옵션값을 보조 가이드와 함께 대조할 때 참고합니다."
          : "답변에서 언급한 기준 문서 항목을 보조 가이드와 함께 확인할 때 참고합니다.",
  };
}

function inferBaselinePriorityItem(message: string, guideSnippets: RetrievedGuideSnippet[]) {
  const haystack = `${message}\n${guideSnippets
    .map((snippet) => `${snippet.sectionTitle}\n${snippet.content}`)
    .join("\n")}`.toLowerCase();

  const signals = buildQuerySignals(haystack);

  if (signals.printer && signals.halfKiosk) {
    return "하프 키오스크 프린터 연결 방식과 호스트네임 포트";
  }

  if (signals.printer && signals.fullKiosk) {
    return "풀 키오스크 지정프린터 설정과 172.25.123.99 포트";
  }

  if (signals.printer) {
    return "프린터 종류, 기본 프린터, 포트 설정";
  }

  if (signals.payment && signals.halfKiosk) {
    return "하프 키오스크 SMARTRO/VCAT 옵션값과 자동설정";
  }

  if (signals.payment && signals.fullKiosk) {
    return "풀 키오스크 KSNET 에이전트, VAN 주소, 포트 설정";
  }

  if (signals.payment) {
    return "결제 에이전트, VAN, CATID, 포트 설정";
  }

  if (signals.runtime || signals.screen) {
    return "서비스 사용 ON, 메인화면 연결, 테스트 설정 원복";
  }

  if (signals.reception) {
    return "접수/진료실 관련 운영 옵션과 메인화면 연결";
  }

  if (signals.peripheral) {
    return "장치 종류, 자동설정, 포트, 연동 옵션";
  }

  return "서비스 사용 ON, 기기 종류/VAN 일치, 메인화면 연결, 테스트 설정 원복";
}

function buildBaselinePriorityCheck(message: string, guideSnippets: RetrievedGuideSnippet[]) {
  const priorityItem = inferBaselinePriorityItem(message, guideSnippets);
  return `기본 설정값 기준으로 보면 ${priorityItem} 항목을 우선 확인해야 합니다.`;
}

function buildSectionFallback(
  section: AnswerSectionKey,
  message: string,
  guideSnippets: RetrievedGuideSnippet[],
  hasGuideGrounding: boolean,
) {
  if (!hasGuideGrounding) {
    if (section === "suspected_causes") {
      return "현재 확보된 기준 문서만으로는 특정 원인을 단정하기 어렵습니다.";
    }

    if (section === "checks") {
      return "현재 확보된 기준 문서상 직접 대응되는 메뉴나 설정 항목이 제한적이어서, 실제 메뉴 경로와 설정값을 추가 확인해야 합니다.";
    }

    return "확보된 기준 문서와 현장 설정값을 대조한 뒤 필요한 원복 절차를 다시 안내합니다.";
  }

  if (section === "suspected_causes") {
    return `문서 기준 ${inferBaselinePriorityItem(message, guideSnippets)} 설정 이탈 가능성`;
  }

  if (section === "checks") {
    return buildBaselinePriorityCheck(message, guideSnippets);
  }

  return `문서 기준값으로 ${inferBaselinePriorityItem(message, guideSnippets)} 관련 설정을 원복한 뒤 다시 확인합니다.`;
}

function buildSyntheticCandidates(
  section: AnswerSectionKey,
  message: string,
  guideSnippets: RetrievedGuideSnippet[],
  guideEvidence: GuideEvidence[],
  hasGuideGrounding: boolean,
) {
  const syntheticCandidates: SectionCandidate[] = [];

  const pushCandidate = (text: string, provenance: SourceGrounding, supportScore: number) => {
    const normalized = normalizeBulletText(text);

    if (!normalized) {
      return;
    }

    syntheticCandidates.push({
      text: normalized,
      normalized: normalized.toLowerCase(),
      provenance,
      supportScore,
      genericRisk: false,
      synthetic: true,
    });
  };

  if (section === "checks") {
    if (hasGuideGrounding) {
      pushCandidate(buildBaselinePriorityCheck(message, guideSnippets), "baseline_doc", 999);

      for (const evidence of guideEvidence.slice(0, 2)) {
        pushCandidate(`문서 기준 확인: ${evidence.text}`, evidence.provenance, 20 + evidence.score);
      }
    } else {
      pushCandidate(buildSectionFallback(section, message, guideSnippets, false), "inferred", 6);
    }
  } else {
    pushCandidate(
      buildSectionFallback(section, message, guideSnippets, hasGuideGrounding),
      hasGuideGrounding ? "baseline_doc" : "inferred",
      hasGuideGrounding ? 18 : 6,
    );
  }

  return syntheticCandidates;
}

function composeSection(
  section: AnswerSectionKey,
  items: string[],
  message: string,
  guideSnippets: RetrievedGuideSnippet[],
  guideEvidence: GuideEvidence[],
  caseMatches: RetrievedMatch[],
) {
  const hasGuideGrounding = guideEvidence.length > 0;
  const guideCorpus = guideSnippets
    .map((snippet) => `${snippet.sectionTitle}\n${getGuideSnippetBody(snippet)}`)
    .join("\n")
    .toLowerCase();
  const caseTexts = buildCaseTexts(caseMatches).filter(Boolean);
  const classified = [
    ...buildSyntheticCandidates(section, message, guideSnippets, guideEvidence, hasGuideGrounding),
    ...items.map((item) => classifyCandidate(item, guideEvidence, caseTexts, guideCorpus)),
  ];
  const bestByText = new Map<string, SectionCandidate>();

  for (const candidate of classified) {
    const existing = bestByText.get(candidate.normalized);

    if (!existing) {
      bestByText.set(candidate.normalized, candidate);
      continue;
    }

    const existingKey = `${PROVENANCE_PRIORITY[existing.provenance]}:${existing.synthetic ? 0 : 1}:${-existing.supportScore}`;
    const candidateKey = `${PROVENANCE_PRIORITY[candidate.provenance]}:${candidate.synthetic ? 0 : 1}:${-candidate.supportScore}`;

    if (candidateKey < existingKey) {
      bestByText.set(candidate.normalized, candidate);
    }
  }

  const sorted = [...bestByText.values()].sort((left, right) => {
    const leftIsPriorityCheck = section === "checks" && left.text.startsWith("기본 설정값 기준으로 보면");
    const rightIsPriorityCheck = section === "checks" && right.text.startsWith("기본 설정값 기준으로 보면");

    if (leftIsPriorityCheck !== rightIsPriorityCheck) {
      return leftIsPriorityCheck ? -1 : 1;
    }

    if (PROVENANCE_PRIORITY[left.provenance] !== PROVENANCE_PRIORITY[right.provenance]) {
      return PROVENANCE_PRIORITY[left.provenance] - PROVENANCE_PRIORITY[right.provenance];
    }

    if (left.genericRisk !== right.genericRisk) {
      return left.genericRisk ? 1 : -1;
    }

    if (left.synthetic !== right.synthetic) {
      return left.synthetic ? -1 : 1;
    }

    return right.supportScore - left.supportScore;
  });

  const results: string[] = [];
  let inferredCount = 0;

  for (const candidate of sorted) {
    if (candidate.provenance === "inferred") {
      if (candidate.genericRisk && results.length < 3) {
        continue;
      }

      if (inferredCount >= MAX_INFERRED_PER_SECTION[section]) {
        continue;
      }

      inferredCount += 1;
    }

    results.push(withProvenanceLabel(candidate.text, candidate.provenance));

    if (results.length >= SECTION_MAX_ITEMS[section]) {
      break;
    }
  }

  if (results.length === 0) {
    results.push(withProvenanceLabel(buildSectionFallback(section, message, guideSnippets, hasGuideGrounding), "inferred"));
  }

  return compactList(results, SECTION_MAX_ITEMS[section]);
}

function buildHospitalReply(checks: string[], nextActions: string[], fallbackUsed: boolean) {
  if (fallbackUsed) {
    return "현재 확보된 기준 문서상 바로 특정 가능한 항목이 제한적이어서, 현장 설정값과 메뉴 경로를 추가 확인한 뒤 다시 안내드리겠습니다.";
  }

  const primaryCheck = normalizeBulletText(checks[0] ?? "");
  const primaryAction = normalizeBulletText(nextActions[0] ?? "");

  if (primaryCheck && primaryAction) {
    return `${primaryCheck}부터 기준 문서대로 확인 중이며, 문서 기준값으로 정리한 뒤 다시 안내드리겠습니다.`;
  }

  if (primaryCheck) {
    return `${primaryCheck}부터 기준 문서대로 확인하겠습니다.`;
  }

  return "기준 문서에 나온 설정값과 절차부터 확인한 뒤 다시 안내드리겠습니다.";
}

function applyBaselineFirstPolicy(params: {
  message: string;
  suspectedCauses: string[];
  checks: string[];
  nextActions: string[];
  guideSnippets: RetrievedGuideSnippet[];
  caseMatches: RetrievedMatch[];
}) {
  const guideEvidence = buildGuideEvidence(params.guideSnippets, params.message);
  const suspectedCauses = composeSection(
    "suspected_causes",
    params.suspectedCauses,
    params.message,
    params.guideSnippets,
    guideEvidence,
    params.caseMatches,
  );
  const checks = composeSection(
    "checks",
    params.checks,
    params.message,
    params.guideSnippets,
    guideEvidence,
    params.caseMatches,
  );
  const nextActions = composeSection(
    "next_actions",
    params.nextActions,
    params.message,
    params.guideSnippets,
    guideEvidence,
    params.caseMatches,
  );
  const fallbackUsed =
    guideEvidence.length === 0 ||
    checks.some((item) => item.includes("직접 대응되는 메뉴나 설정 항목이 제한적"));

  return {
    suspectedCauses,
    checks,
    nextActions,
    hospitalReply: buildHospitalReply(checks, nextActions, fallbackUsed),
    fallbackUsed,
  } satisfies GovernedIncidentResponse;
}

function buildGuideModeResponse(
  message: string,
  guideSnippets: RetrievedGuideSnippet[],
  highThreshold: number,
  lowThreshold: number,
): ChatApiResponse {
  const topGuideSimilarity = guideSnippets[0]?.similarity ?? null;
  const confidenceLevel = determineConfidenceLevel(topGuideSimilarity, highThreshold, lowThreshold);
  const guideEvidence = buildGuideEvidence(guideSnippets, message, 8);

  if (guideSnippets.length === 0 || guideEvidence.length === 0 || topGuideSimilarity === null || topGuideSimilarity < lowThreshold) {
    return {
      query_mode: "guide",
      suspected_causes: [],
      checks: [],
      next_actions: [],
      guide_overview: "현재 확보된 기준 문서에서 직접 대응되는 메뉴나 절차를 찾지 못했습니다.",
      guide_steps: [
        "- 현재 확인 가능한 기준 문서상 직접 대응되는 메뉴와 설정 항목이 제한적입니다.",
        "- 장비 종류, 화면명, 메뉴 경로, 현재 설정값을 추가 확인한 뒤 문서 기준으로 다시 안내합니다.",
      ],
      hospital_reply: "",
      show_related_guide: false,
      related_guide_title: "",
      related_guide_excerpt: "",
      related_guide_reason: "",
      confidence_level: confidenceLevel,
      confidence_note: buildGuideConfidenceNote(confidenceLevel, topGuideSimilarity, 0),
      similar_case_count: 0,
      top_similarity: topGuideSimilarity,
      most_similar_case_summary: "",
      similar_cases: [],
      fallback_used: true,
    };
  }

  const titles = [...new Set(guideSnippets.slice(0, 3).map((snippet) => snippet.sectionTitle).filter(Boolean))];

  return {
    query_mode: "guide",
    suspected_causes: [],
    checks: [],
    next_actions: [],
    guide_overview: `기준 문서 우선: ${titles.join(" / ")} 항목을 먼저 확인합니다. 문서에 나온 메뉴명, 설정명, 값 기준으로만 안내합니다.`,
    guide_steps: compactList(guideEvidence.slice(0, 6).map((item) => `- ${item.text}`), 8),
    hospital_reply: "",
    show_related_guide: false,
    related_guide_title: "",
    related_guide_excerpt: "",
    related_guide_reason: "",
    confidence_level: confidenceLevel,
    confidence_note: buildGuideConfidenceNote(confidenceLevel, topGuideSimilarity, guideSnippets.length),
    similar_case_count: 0,
    top_similarity: topGuideSimilarity,
    most_similar_case_summary: "",
    similar_cases: [],
    fallback_used: false,
  };
}

function splitRetrievedDocuments(input: Array<[Document, number]>) {
  const caseMatches: RetrievedMatch[] = [];
  const guideSnippets: RetrievedGuideSnippet[] = [];

  input.forEach(([document, similarity]) => {
    const metadata = document.metadata as Record<string, unknown>;
    const sourceType = String(metadata.source_type ?? "support_case");

    if (sourceType !== "support_case") {
      guideSnippets.push({
        sourceType: "guide",
        provenance: detectGuideProvenance(sourceType, String(metadata.source_file ?? "")),
        similarity,
        sectionTitle: String(metadata.guide_section_title ?? ""),
        sourceFile: String(metadata.source_file ?? ""),
        content: document.pageContent,
      });
      return;
    }

    caseMatches.push({
      sourceType: "case",
      provenance: "case_history",
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
  const searchTopK = Math.max(baseTopK * 4, queryMode === "guide" ? 20 : 24);
  const searchResults = await vectorStore.similaritySearchWithScore(message, searchTopK);
  const rerankedResults = rerankRetrievedDocuments(searchResults, message, queryMode);
  const { caseMatches, guideSnippets } = splitRetrievedDocuments(rerankedResults);
  const guideContext = guideSnippets.slice(0, queryMode === "incident" ? 6 : 8);
  const caseContext = queryMode === "incident" ? caseMatches.slice(0, 4) : [];
  const topSimilarity = guideContext[0]?.similarity ?? caseContext[0]?.similarity ?? null;
  const confidenceLevel = determineConfidenceLevel(
    topSimilarity,
    config.HIGH_CONFIDENCE_THRESHOLD,
    config.LOW_CONFIDENCE_THRESHOLD,
  );

  if (queryMode === "guide") {
    return buildGuideModeResponse(
      message,
      guideContext,
      config.HIGH_CONFIDENCE_THRESHOLD,
      config.LOW_CONFIDENCE_THRESHOLD,
    );
  }

  const paymentIncident = isPaymentIncident(message, caseContext, guideContext);

  const llm = createChatModel().withStructuredOutput(responseSchema, {
    name: "support_case_response",
    strict: true,
  });

  const structured = await llm.invoke([
    new SystemMessage(
      [
        "너는 유비케어 병원고객팀 상담사용 내부 지원 챗봇이다.",
        "모든 incident 답변은 반드시 기준 문서 우선 원칙을 따른다.",
        "1차 답변 상단에는 기준 문서에서 직접 찾은 설정값, 절차, 메뉴, 체크포인트만 배치하라.",
        "기준 문서보다 사례나 일반론을 앞세우지 마라.",
        "문서에 없는 운영체제 상식, 하드웨어 상식, 네트워크 추정, 업계 관성적 대응은 2차 보조안으로만 남겨라.",
        "메뉴명, 설정명, 옵션값, 절차 순서는 가능한 한 문서 표현을 유지하라.",
        "추정은 추정으로만 남기고, 문서 근거보다 먼저 쓰지 마라.",
        "답변이 막혀도 함부로 일반 IT 상식으로 빈칸을 메우지 마라.",
        "incident 모드에서는 의심 원인, 우선 확인사항, 권장 대응 방향을 작성한다.",
        "checks의 첫 번째 항목은 반드시 `기본 설정값 기준으로 보면 ... 항목을 우선 확인해야 합니다.` 형식으로 작성하라.",
        "guide_overview는 빈 문자열, guide_steps는 빈 배열로 채워라.",
        "짧고 건조하게 작성하라. 미사여구 금지.",
        ...(paymentIncident ? buildPaymentPolicyPrompt() : []),
      ].join("\n"),
    ),
    new HumanMessage(
      [
        `현재 문의:\n${message}`,
        `질의 모드: ${queryMode}`,
        "",
        `검색된 기준 문서:\n${formatRetrievedGuideSnippets(guideContext)}`,
        "",
        `검색된 유사 사례:\n${formatRetrievedCases(caseContext)}`,
        "",
        `검색 신뢰도 등급: ${confidenceLevel}`,
        "기준 문서에 직접 없는 일반론은 상단 bullet에 두지 마라.",
        "문서 근거가 있는 경우 메뉴명, 설정명, 옵션값, IP, 포트, 실행 파일명을 그대로 써라.",
        "유사 사례는 기준 문서를 보조하는 수준으로만 활용하라.",
        ...(paymentIncident
          ? [
              "이 문의는 결제/카드 실패 incident로 보고 현금 결제 대체안, 외부 VAN/카드사 망 추측을 배제하라.",
              "상담사가 통화 중 원격지원으로 바로 확인할 수 있는 항목을 우선 제시하고, 현장 조작 요청은 최소화하라.",
            ]
          : []),
      ].join("\n"),
    ),
  ]);

  const sanitizedPaymentResponse = paymentIncident
    ? sanitizePaymentIncidentResponse(
        structured.suspected_causes,
        structured.checks,
        structured.next_actions,
      )
    : null;
  const governed = applyBaselineFirstPolicy({
    message,
    suspectedCauses: paymentIncident ? sanitizedPaymentResponse!.suspectedCauses : structured.suspected_causes,
    checks: paymentIncident ? sanitizedPaymentResponse!.checks : structured.checks,
    nextActions: paymentIncident ? sanitizedPaymentResponse!.nextActions : structured.next_actions,
    guideSnippets: guideContext,
    caseMatches: caseContext,
  });
  const similarCases =
    queryMode === "incident"
      ? caseMatches
          .filter((match) => match.similarity >= config.LOW_CONFIDENCE_THRESHOLD)
          .slice(0, 3)
          .map((match) => match.summary)
      : [];
  const relatedGuidePreview = buildRelatedGuidePreview(
    message,
    governed.suspectedCauses,
    governed.checks,
    governed.nextActions,
    guideContext,
    paymentIncident,
    config.LOW_CONFIDENCE_THRESHOLD,
  );

  return {
    query_mode: queryMode,
    suspected_causes: governed.suspectedCauses,
    checks: governed.checks,
    next_actions: governed.nextActions,
    guide_overview: "",
    guide_steps: [],
    hospital_reply: governed.hospitalReply,
    ...relatedGuidePreview,
    confidence_level: confidenceLevel,
    confidence_note: buildConfidenceNote(confidenceLevel, topSimilarity, guideContext.length, similarCases.length),
    similar_case_count: similarCases.length,
    top_similarity: topSimilarity,
    most_similar_case_summary:
      similarCases.length > 0 ? buildMostSimilarCaseSummary(caseMatches[0], guideContext.length > 0) : "",
    similar_cases: similarCases,
    fallback_used: confidenceLevel === "low" || governed.fallbackUsed,
  };
}

export const __testing = {
  rerankRetrievedDocuments,
  splitRetrievedDocuments,
  buildGuideEvidence,
  applyBaselineFirstPolicy,
};
