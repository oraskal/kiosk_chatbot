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

type RelatedGuidePreview = {
  show_related_guide: boolean;
  related_guide_title: string;
  related_guide_excerpt: string;
  related_guide_reason: string;
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
    return "키오스크 내부 결제 설정과 에이전트 실행 상태가 배포 기본값과 다를 가능성";
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
  message: string,
  guideSnippets: RetrievedGuideSnippet[],
) {
  const defaultCauses = [
    "배포 기본 세팅과 다른 결제 설정(VAN, CATID, 포트, 결제 옵션값) 이 반영됐을 가능성",
    "키오스크 내부 카드장치 인식 또는 연결 상태가 불안정할 가능성",
    "결제 에이전트나 드라이버, 자동실행 옵션이 현장 장비와 맞지 않을 가능성",
  ];
  const defaultChecks = [
    "원격지원으로 서비스 사용, 기기 종류, VAN 선택, CATID, 메인화면 연결, 테스트 설정 원복 여부를 확인합니다.",
    "원격지원으로 결제 에이전트 실행 상태와 아이콘, 자동실행, 옵션값이 배포 기본값과 일치하는지 확인합니다.",
    "원격지원으로 장치 인식 상태와 결제 프로그램 오류 표시, 장치 관리자 경고 여부를 확인합니다.",
    "필요하면 통화 유지 상태에서 Windows 포함 키오스크 전체 재부팅 후 동일 증상 재현 여부만 다시 확인합니다.",
    "실물 카드 승인 재현은 현장에서만 가능한 경우에 한해 최소 범위로 요청합니다.",
  ];
  const defaultActions = [
    "상담사가 원격지원으로 결제 기본 설정과 에이전트 상태를 먼저 정리한 뒤 재시험 순서를 안내합니다.",
    "설정 이탈이나 에이전트 비정상이 확인되면 배포 기본값에 맞게 복구 후 카드 승인 재시도를 진행합니다.",
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
    checks: ensureBaselineFirstChecks(sanitizedChecks, message, guideSnippets),
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
  const answerParts = [...suspectedCauses, ...checks, ...nextActions];
  const hasBaselineSignal = answerParts.some((item) => /(배포 기본|기본 세팅|기본 설정|기본값|이탈)/.test(item));

  if (!hasBaselineSignal) {
    return createEmptyRelatedGuidePreview();
  }

  const selectedSnippet = selectRelatedGuideSnippet(guideSnippets, message, answerParts, paymentIncident);

  if (!selectedSnippet) {
    return createEmptyRelatedGuidePreview();
  }

  if (selectedSnippet.similarity < lowThreshold && !paymentIncident) {
    return createEmptyRelatedGuidePreview();
  }

  return {
    show_related_guide: true,
    related_guide_title: selectedSnippet.sectionTitle || "관련 기준 가이드",
    related_guide_excerpt: buildFocusedGuideExcerpt(selectedSnippet, message, answerParts, paymentIncident),
    related_guide_reason: paymentIncident
      ? "결제 기본 설정과 에이전트/옵션값이 배포 기준과 다른지 비교할 때 참고합니다."
      : "답변에서 언급한 기본 설정 기준을 확인할 때 참고합니다.",
  };
}

function inferBaselinePriorityItem(message: string, guideSnippets: RetrievedGuideSnippet[]) {
  const haystack = `${message}\n${guideSnippets
    .map((snippet) => `${snippet.sectionTitle}\n${snippet.content}`)
    .join("\n")}`.toLowerCase();

  const isHalfKiosk = haystack.includes("하프");
  const isFullKiosk = haystack.includes("풀");
  const hasPrinterSignal =
    haystack.includes("프린터") ||
    haystack.includes("출력") ||
    haystack.includes("영수증") ||
    haystack.includes("처방전");
  const hasPaymentSignal =
    haystack.includes("결제") ||
    haystack.includes("카드") ||
    haystack.includes("단말기") ||
    haystack.includes("ksnet") ||
    haystack.includes("smartro") ||
    haystack.includes("vcat");
  const hasReceptionSignal =
    haystack.includes("접수") || haystack.includes("신규환자") || haystack.includes("문진");
  const hasDoctorRoomSignal =
    haystack.includes("진료실") || haystack.includes("일시정지") || haystack.includes("접수시간");
  const hasAccessibilitySignal =
    haystack.includes("음성") || haystack.includes("장애인") || haystack.includes("언어팩");

  if (hasPrinterSignal && isHalfKiosk) {
    return "하프 키오스크 프린터 연결 방식(호스트 네임 포트, 기본 프린터)";
  }

  if (hasPrinterSignal && isFullKiosk) {
    return "풀 키오스크 지정프린터 설정(용지함2개, 172.25.123.99 포트, USB 제거)";
  }

  if (hasPrinterSignal) {
    return "프린터 기본 연결 및 포트 설정";
  }

  if (hasPaymentSignal && isHalfKiosk) {
    return "하프 키오스크 결제 옵션값과 SMARTRO/VCAT 기본 설정";
  }

  if (hasPaymentSignal && isFullKiosk) {
    return "풀 키오스크 KSNET 에이전트와 결제 포트 기본 설정";
  }

  if (hasPaymentSignal) {
    return "결제 에이전트, VAN, 포트 기본 설정";
  }

  if (hasReceptionSignal) {
    return "접수 기본 옵션(신규환자, 진료과, 사전문진, 임시접수)";
  }

  if (hasDoctorRoomSignal) {
    return "진료실 운영 설정(접수시간, 일시정지, 메인화면 연결)";
  }

  if (hasAccessibilitySignal) {
    return "장애인 메뉴와 음성안내 기본 설정";
  }

  return "서비스 사용 ON, 기기 종류/VAN 일치, 메인화면 연결, 테스트 설정 원복";
}

function buildBaselinePriorityCheck(message: string, guideSnippets: RetrievedGuideSnippet[]) {
  const priorityItem = inferBaselinePriorityItem(message, guideSnippets);
  return `배포 기본 세팅 기준으로 보면 ${priorityItem} 항목을 우선 확인해야 합니다.`;
}

function ensureBaselineFirstChecks(checks: string[], message: string, guideSnippets: RetrievedGuideSnippet[]) {
  const baselinePriorityCheck = buildBaselinePriorityCheck(message, guideSnippets);
  const remainingChecks = checks.filter((item) => !normalizeBulletText(item).startsWith("배포 기본 세팅 기준으로 보면"));
  return compactList([baselinePriorityCheck, ...remainingChecks], 6);
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
  const paymentIncident = queryMode === "incident" && isPaymentIncident(message, caseContext, guideContext);

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
      show_related_guide: false,
      related_guide_title: "",
      related_guide_excerpt: "",
      related_guide_reason: "",
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
        "incident 모드에서는 과거 지원내역보다 먼저 현재 현장이 배포 기본 세팅에서 이탈했는지 확인하라.",
        "incident 모드에서는 유사 사례를 바로 답으로 삼지 말고, 기본 설정 이탈 여부를 먼저 점검한 뒤 보조 근거로 활용하라.",
        "모드가 incident(장애/오류)라면 현재 문제상황과 가장 가까운 패턴을 찾아 의심 원인, 우선 확인사항, 권장 대응 방향을 정리하라.",
        "incident 모드에서는 checks의 첫 번째 항목을 반드시 `배포 기본 세팅 기준으로 보면 ... 항목을 우선 확인해야 합니다.` 형식으로 작성하라.",
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
        ...(paymentIncident ? buildPaymentPolicyPrompt() : []),
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
        "과거 지원내역보다 먼저 기본 설정 이탈 여부를 확인하는 흐름으로 답변하라.",
        "guide 모드에서는 장애 원인 단정 표현을 피하고, 기준 절차/설정 포인트를 명확하게 안내하라.",
        "incident 모드에서는 증상과 근거를 연결해 원인 가설과 확인 포인트를 우선 제시하라.",
        "guide_overview에는 한눈에 이해되는 핵심 가이드 요약을 2~4문장으로 작성하라.",
        "guide_steps에는 실제 클릭/설정 순서를 2~6개로 작성하라.",
        "guide_steps에는 가능한 경우 IP/포트/메뉴 경로 같은 구체값을 포함하라.",
        "긴 설명 대신 값/경로/단계만 남겨라.",
        "출력에는 사례/가이드를 그대로 복붙하지 말고, 현재 문의 맥락에 맞는 실무형 답변만 담아라.",
        ...(paymentIncident
          ? [
              "이 문의는 결제/카드 실패 incident로 보고 현금 결제 대체안, 카드리더기 단독 재부팅, 외부 VAN/카드사 망 추측을 배제하라.",
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
        message,
        guideContext,
      )
    : null;
  const compactedCauses = paymentIncident
    ? sanitizedPaymentResponse!.suspectedCauses
    : compactList(structured.suspected_causes, 4);
  const compactedChecks = paymentIncident
    ? sanitizedPaymentResponse!.checks
    : ensureBaselineFirstChecks(structured.checks, message, guideContext);
  const compactedActions = paymentIncident
    ? sanitizedPaymentResponse!.nextActions
    : compactList(structured.next_actions, 6);
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
  const relatedGuidePreview = buildRelatedGuidePreview(
    message,
    compactedCauses,
    compactedChecks,
    compactedActions,
    guideContext,
    paymentIncident,
    config.LOW_CONFIDENCE_THRESHOLD,
  );

  return {
    ...structured,
    query_mode: queryMode,
    suspected_causes: compactedCauses,
    checks: compactedChecks,
    next_actions: compactedActions,
    guide_overview: "",
    guide_steps: [],
    ...relatedGuidePreview,
    confidence_level: confidenceLevel,
    confidence_note: confidenceNote,
    similar_case_count: similarCases.length,
    top_similarity: topSimilarity,
    most_similar_case_summary: queryMode === "incident" ? buildMostSimilarCaseSummary(caseMatches[0]) : "",
    similar_cases: similarCases,
    fallback_used: confidenceLevel === "low",
  };
}
