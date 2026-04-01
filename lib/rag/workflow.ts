import { Document } from "@langchain/core/documents";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

import { getRequiredServerConfig } from "@/lib/config";
import { getSupabaseVectorStore } from "@/lib/rag/vector-store";
import type { BaselineReference, ChatApiResponse, ConfidenceLevel, QueryMode, SimilarCaseSummary } from "@/lib/types";

const QUERY_MODE_VALUES = ["incident", "guide"] as const;
const DOMAIN_VALUES = ["payment", "printer", "scanner", "launcher", "login", "network", "display", "etc"] as const;
const OBJECT_VALUES = [
  "card_payment",
  "payment_reader",
  "receipt_printer",
  "document_printer",
  "barcode_scanner",
  "kiosk_player",
  "login_account",
  "network_connection",
  "display_panel",
  "etc",
] as const;
const STAGE_VALUES = [
  "before_action",
  "in_progress",
  "after_action",
  "after_payment_or_after_approval",
  "cancel_flow",
  "startup",
  "output",
  "recognition",
  "login_flow",
  "network_flow",
  "etc",
] as const;
const POLARITY_VALUES = [
  "fails",
  "does_not_happen",
  "auto_happens",
  "auto_cancels",
  "cannot_cancel",
  "repeats",
  "closes_immediately",
  "outputs_twice",
  "scans_twice",
  "fails_to_start",
  "recognition_fail",
  "no_output",
  "unknown",
] as const;
const INTENT_VALUES = ["diagnosis", "baseline_check", "official_process", "recovery"] as const;
const SCOPE_VALUES = ["single_case", "all_cases", "intermittent", "persistent"] as const;
const PROVENANCE_VALUES = ["baseline_doc", "process_doc", "case_history", "inferred"] as const;

type Domain = (typeof DOMAIN_VALUES)[number];
type EvidenceObject = (typeof OBJECT_VALUES)[number];
type Stage = (typeof STAGE_VALUES)[number];
type Polarity = (typeof POLARITY_VALUES)[number];
type Intent = (typeof INTENT_VALUES)[number];
type Scope = (typeof SCOPE_VALUES)[number];
type EvidenceProvenance = (typeof PROVENANCE_VALUES)[number];
type LineFeature = "menu_path" | "test_step" | "config_step" | "install_step" | "process_step" | "cause_hint" | "guide_hint";
type IncidentSectionKey = "suspected_causes" | "checks" | "actions";
type RetrievalChannel = "baseline" | "process" | "case";

type SymptomSignature = {
  queryMode: QueryMode;
  domain: Domain;
  object: EvidenceObject;
  stage: Stage;
  polarity: Polarity;
  intent: Intent;
  scope: Scope;
  symptomSummary: string;
};

type RetrievedCase = {
  sourceType: "case";
  similarity: number;
  summary: SimilarCaseSummary;
  customerReplyReference: string;
  latestActionReference: string;
  qualityTier: string;
};

type RetrievedGuide = {
  sourceType: "guide";
  similarity: number;
  sectionTitle: string;
  content: string;
  sourceFile: string;
};

type RetrievedEvidence = {
  id: string;
  similarity: number;
  retrievalChannels: RetrievalChannel[];
  sourceType: "case" | "guide";
  provenance: EvidenceProvenance;
  title: string;
  content: string;
  sourceFile: string;
  signature: SymptomSignature;
  summary?: SimilarCaseSummary;
  customerReplyReference?: string;
  latestActionReference?: string;
  qualityTier?: string;
  alignmentScore: number;
  exclusionReason?: string;
};

type RetrievalCandidate = {
  document: Document;
  similarity: number;
  query: string;
  channel: RetrievalChannel;
};

type GroundedEvidenceBundle = {
  baselineDocs: RetrievedEvidence[];
  processDocs: RetrievedEvidence[];
  caseHistories: RetrievedEvidence[];
  excluded: RetrievedEvidence[];
  fallbackNeeded: boolean;
};

type SectionCandidate = {
  text: string;
  provenance: EvidenceProvenance;
  score: number;
  evidenceId?: string;
  features?: LineFeature[];
};

type IncidentDraft = {
  suspectedCauses: SectionCandidate[];
  checks: SectionCandidate[];
  actions: SectionCandidate[];
  fallbackUsed: boolean;
};

const interpretationSchema = z.object({
  query_mode: z.enum(QUERY_MODE_VALUES),
  domain: z.enum(DOMAIN_VALUES),
  object: z.enum(OBJECT_VALUES),
  stage: z.enum(STAGE_VALUES),
  polarity: z.enum(POLARITY_VALUES),
  intent: z.enum(INTENT_VALUES),
  scope: z.enum(SCOPE_VALUES),
  symptom_summary: z.string().max(120),
});

const groundingSchema = z.object({
  baseline_ids: z.array(z.string()).max(6),
  process_ids: z.array(z.string()).max(6),
  case_ids: z.array(z.string()).max(4),
  exclude_case_ids: z.array(z.string()).max(6),
  fallback_needed: z.boolean(),
});

const compositionSchema = z.object({
  suspected_causes: z.array(z.string()).min(0).max(4),
  checks: z.array(z.string()).min(1).max(6),
  actions: z.array(z.string()).min(1).max(6),
});

const PROVENANCE_LABEL: Record<EvidenceProvenance, string> = {
  baseline_doc: "[기준 문서]",
  process_doc: "[운영 절차]",
  case_history: "[유사 사례]",
  inferred: "[추정 보완]",
};

const PROVENANCE_PRIORITY: Record<EvidenceProvenance, number> = {
  baseline_doc: 0,
  process_doc: 1,
  case_history: 2,
  inferred: 3,
};

const SECTION_LIMITS: Record<IncidentSectionKey, number> = {
  suspected_causes: 4,
  checks: 6,
  actions: 6,
};

const GUIDE_PROCESS_PATTERNS = [
  /프로세스/i,
  /절차/i,
  /요청/i,
  /처리 기준/i,
  /운영 안내/i,
  /후속 처리/i,
  /승인 취소/i,
];

const GENERIC_GUESS_PATTERNS = [
  /장치\s*관리자/i,
  /usb\s*인식/i,
  /드라이버/i,
  /방화벽/i,
  /케이블/i,
  /외부망/i,
  /VAN사\s*통신망/i,
];

const TERM_STOP_WORDS = new Set([
  "기준",
  "문서",
  "설정",
  "확인",
  "항목",
  "경우",
  "관련",
  "정상",
  "진행",
  "필요",
  "방법",
  "안내",
  "절차",
  "프로세스",
  "요청",
  "환자",
  "병원",
  "합니다",
]);

function createChatModel() {
  const config = getRequiredServerConfig();

  return new ChatOpenAI({
    apiKey: config.OPENAI_API_KEY,
    model: config.OPENAI_CHAT_MODEL,
  });
}

function normalizeSpace(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

function normalizeLine(text: string) {
  return normalizeSpace(text.replace(/^[-*]\s+/, "").replace(/^\d+[\.\)]\s+/, ""));
}

function toComparableText(text: string) {
  return normalizeSpace(text.toLowerCase());
}

function extractTerms(text: string) {
  return toComparableText(text)
    .split(/[^0-9a-zA-Z가-힣_/.-]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 1 && !TERM_STOP_WORDS.has(token));
}

function countOverlap(left: string[], right: string[]) {
  if (left.length === 0 || right.length === 0) {
    return 0;
  }

  const rightSet = new Set(right);
  return [...new Set(left)].filter((token) => rightSet.has(token)).length;
}

function uniqueStrings(items: string[], maxItems: number) {
  const seen = new Set<string>();
  const output: string[] = [];

  for (const item of items) {
    const cleaned = normalizeSpace(item);
    const key = toComparableText(cleaned);

    if (!cleaned || seen.has(key)) {
      continue;
    }

    seen.add(key);
    output.push(cleaned);

    if (output.length >= maxItems) {
      break;
    }
  }

  return output;
}

function withProvenanceLabel(text: string, provenance: EvidenceProvenance) {
  return `${PROVENANCE_LABEL[provenance]} ${normalizeSpace(text)}`;
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

function buildConfidenceNote(
  level: ConfidenceLevel,
  topSimilarity: number | null,
  baselineCount: number,
  processCount: number,
  caseCount: number,
) {
  if (level === "low" || topSimilarity === null) {
    return "관련 기준문서 근거가 충분하지 않아 보수적으로 정리했습니다.";
  }

  return `기준문서 ${baselineCount}건, 운영 절차 ${processCount}건, 유사 사례 ${caseCount}건을 비교해 정리했습니다. 최고 유사도는 ${topSimilarity.toFixed(2)}입니다.`;
}

function buildGuideConfidenceNote(level: ConfidenceLevel, topSimilarity: number | null, count: number) {
  if (level === "low" || topSimilarity === null) {
    return "관련 기준문서가 제한적이어서 직접 확인 가능한 메뉴와 설정만 보수적으로 안내했습니다.";
  }

  return `관련 문서 ${count}건을 바탕으로 정리했습니다. 최고 유사도는 ${topSimilarity.toFixed(2)}입니다.`;
}

function hasAny(text: string, patterns: RegExp[]) {
  return patterns.some((pattern) => pattern.test(text));
}

function countMatches(text: string, patterns: RegExp[]) {
  return patterns.reduce((count, pattern) => count + (pattern.test(text) ? 1 : 0), 0);
}

const ACTION_FAMILY_PATTERNS = {
  cancel: [/취소/i, /cancel/i],
  startup: [/실행/i, /시작/i, /기동/i, /켜/i, /launch/i, /start/i, /boot/i],
  output: [/출력/i, /인쇄/i, /프린트/i, /print/i],
  recognition: [/인식/i, /스캔/i, /리더/i, /barcode/i, /scan/i, /read/i],
};

const MODIFIER_PATTERNS = {
  auto: [/자동/i, /저절로/i, /스스로/i, /혼자/i, /by itself/i, /automatically/i],
  negative: [/안\s?됨/i, /안\s?돼/i, /되지 않/i, /못/i, /불가/i, /실패/i, /오류/i, /fail/i, /cannot/i],
  repeated: [/계속/i, /반복/i, /자꾸/i, /두 번/i, /두번/i, /여러 번/i, /중복/i, /repeat/i, /twice/i],
  immediateClose: [/바로 꺼/i, /즉시 종료/i, /곧 꺼/i, /immediately closes/i, /closes right away/i],
  guide: [/어떻게/i, /방법/i, /어디/i, /메뉴/i, /경로/i, /설정/i, /기준/i, /값/i],
  process: [/절차/i, /프로세스/i, /처리/i, /요청/i, /안내/i, /후속/i],
  recovery: [/복구/i, /원복/i, /재실행/i, /재기동/i, /recover/i, /restore/i],
  intermittent: [/가끔/i, /간헐/i, /때때로/i, /occasionally/i, /intermittent/i],
  persistent: [/계속/i, /항상/i, /매번/i, /지속/i, /always/i, /persistent/i],
  allCases: [/전체/i, /모든/i, /전부/i, /모두/i, /all/i],
};

const EVENT_CONTEXT_PATTERNS = {
  completed: [
    /정상/i,
    /완료/i,
    /성공/i,
    /승인/i,
    /결제된/i,
    /수납된/i,
    /처리된/i,
    /잘\s?되/i,
    /approved/i,
    /completed/i,
    /successful/i,
  ],
  observedLater: [
    /나중/i,
    /이후/i,
    /뒤에/i,
    /조금 뒤/i,
    /살펴보면/i,
    /확인해보면/i,
    /처리되어 있/i,
    /남아 있/i,
    /later/i,
    /afterward/i,
  ],
  reflectionMissing: [
    /미반영/i,
    /누락/i,
    /안 된 것/i,
    /잡히지 않/i,
    /처리되지 않/i,
    /반영 안/i,
    /not reflected/i,
    /missing/i,
  ],
  attempt: [/시도/i, /진행/i, /하려/i, /해야/i, /필요/i, /요청/i, /trying/i, /attempt/i],
};

function hasObservedOutcomeContext(text: string) {
  return (
    hasAny(text, EVENT_CONTEXT_PATTERNS.observedLater) ||
    hasAny(text, EVENT_CONTEXT_PATTERNS.reflectionMissing) ||
    (hasAny(text, EVENT_CONTEXT_PATTERNS.completed) &&
      (hasAny(text, EVENT_CONTEXT_PATTERNS.observedLater) || hasAny(text, EVENT_CONTEXT_PATTERNS.reflectionMissing)))
  );
}

function hasAttemptContext(text: string) {
  return hasAny(text, EVENT_CONTEXT_PATTERNS.attempt);
}

function inferQueryModeFallback(message: string): QueryMode {
  const text = toComparableText(message);
  const incidentSignal =
    hasAny(text, [
      MODIFIER_PATTERNS.negative[0],
      MODIFIER_PATTERNS.negative[1],
      /오류/i,
      /실패/i,
      /멈춤/i,
      ...MODIFIER_PATTERNS.repeated,
      ...MODIFIER_PATTERNS.immediateClose,
    ]) || hasAny(text, [...ACTION_FAMILY_PATTERNS.cancel, ...ACTION_FAMILY_PATTERNS.output, ...ACTION_FAMILY_PATTERNS.recognition]);
  const guideSignal = hasAny(text, MODIFIER_PATTERNS.guide);

  if (guideSignal && !incidentSignal) {
    return "guide";
  }

  return "incident";
}

function inferDomain(text: string): Domain {
  const domainPatterns: Array<{ domain: Domain; patterns: RegExp[] }> = [
    { domain: "printer", patterns: [/영수증/i, /프린터/i, /출력/i, /인쇄/i, /printer/i, /print/i] },
    { domain: "scanner", patterns: [/스캐너/i, /바코드/i, /리더기/i, /qr/i, /scanner/i, /barcode/i] },
    { domain: "launcher", patterns: [/프로그램/i, /플레이어/i, /런처/i, /실행/i, /시작/i, /종료/i, /메인화면/i, /launcher/i, /player/i, /app/i] },
    { domain: "payment", patterns: [/결제/i, /카드/i, /승인/i, /정산/i, /수납/i, /payment/i, /card/i] },
    { domain: "login", patterns: [/로그인/i, /비밀번호/i, /계정/i, /아이디/i, /login/i, /password/i, /account/i] },
    { domain: "network", patterns: [/네트워크/i, /통신/i, /인터넷/i, /lan/i, /ip/i, /port/i, /network/i] },
    { domain: "display", patterns: [/화면/i, /모니터/i, /터치/i, /display/i, /screen/i, /monitor/i] },
  ];

  const top = domainPatterns
    .map((entry) => ({
      domain: entry.domain,
      score: countMatches(text, entry.patterns),
    }))
    .sort((left, right) => right.score - left.score)[0];

  if (!top || top.score === 0) {
    return "etc";
  }

  return top.domain;
}

function inferObject(text: string, domain: Domain): EvidenceObject {
  if (domain === "payment") {
    if (/(리더기|단말기|ic|마그네틱|긁|삽입|카드 인식|카드인식|reader|terminal)/i.test(text)) {
      return "payment_reader";
    }
    return "card_payment";
  }
  if (/(영수증|롤프린터|카드 영수증)/i.test(text)) {
    return "receipt_printer";
  }
  if (/(처방전|제증명)/i.test(text)) {
    return "document_printer";
  }
  if (domain === "scanner" || /(스캐너|바코드|리더기)/i.test(text)) {
    return "barcode_scanner";
  }
  if (domain === "launcher") {
    return "kiosk_player";
  }
  if (domain === "login") {
    return "login_account";
  }
  if (domain === "network") {
    return "network_connection";
  }
  if (domain === "display") {
    return "display_panel";
  }
  return "etc";
}

function inferStage(text: string, domain: Domain): Stage {
  const cancelObservedAfterCompletion =
    hasAny(text, ACTION_FAMILY_PATTERNS.cancel) &&
    domain === "payment" &&
    (hasAny(text, MODIFIER_PATTERNS.auto) || hasObservedOutcomeContext(text)) &&
    !hasAttemptContext(text);

  if (cancelObservedAfterCompletion) {
    return "after_payment_or_after_approval";
  }

  if (hasAny(text, ACTION_FAMILY_PATTERNS.cancel) && hasAny(text, MODIFIER_PATTERNS.auto)) {
    return "after_payment_or_after_approval";
  }
  if (hasAny(text, ACTION_FAMILY_PATTERNS.cancel)) {
    return "cancel_flow";
  }
  if (hasAny(text, ACTION_FAMILY_PATTERNS.startup) && hasAny(text, MODIFIER_PATTERNS.immediateClose)) {
    return "startup";
  }
  if (domain === "launcher" || hasAny(text, ACTION_FAMILY_PATTERNS.startup)) {
    return "startup";
  }
  if (hasAny(text, ACTION_FAMILY_PATTERNS.output)) {
    return "output";
  }
  if (hasAny(text, ACTION_FAMILY_PATTERNS.recognition)) {
    return "recognition";
  }
  if (domain === "login") {
    return "login_flow";
  }
  if (domain === "network") {
    return "network_flow";
  }
  return "in_progress";
}

function inferPolarity(text: string, domain: Domain, stage: Stage): Polarity {
  const hasCancel = hasAny(text, ACTION_FAMILY_PATTERNS.cancel);
  const hasStartup = hasAny(text, ACTION_FAMILY_PATTERNS.startup);
  const hasOutput = hasAny(text, ACTION_FAMILY_PATTERNS.output);
  const hasRecognition = hasAny(text, ACTION_FAMILY_PATTERNS.recognition);
  const hasAuto = hasAny(text, MODIFIER_PATTERNS.auto);
  const hasNegative = hasAny(text, MODIFIER_PATTERNS.negative);
  const hasRepeated = hasAny(text, MODIFIER_PATTERNS.repeated);
  const hasImmediateClose = hasAny(text, MODIFIER_PATTERNS.immediateClose);
  const observedOutcome = hasObservedOutcomeContext(text);
  const attemptedAction = hasAttemptContext(text);

  if (hasCancel && (hasAuto || (domain === "payment" && observedOutcome && !attemptedAction))) {
    return "auto_cancels";
  }
  if (hasCancel && hasNegative && !observedOutcome) {
    return "cannot_cancel";
  }
  if (hasStartup && hasImmediateClose) {
    return "closes_immediately";
  }
  if (hasOutput && hasRepeated) {
    return "outputs_twice";
  }
  if (hasRecognition && hasRepeated) {
    return "scans_twice";
  }
  if (hasOutput && hasNegative) {
    return "no_output";
  }
  if (hasRecognition && hasNegative) {
    return "recognition_fail";
  }
  if ((hasStartup || domain === "launcher") && hasNegative) {
    return "fails_to_start";
  }
  if (hasRepeated) {
    return "repeats";
  }
  if (hasNegative || /비정상|이상|문제/i.test(text)) {
    return stage === "output" ? "no_output" : stage === "recognition" ? "recognition_fail" : "fails";
  }
  return "unknown";
}

function inferScope(text: string): Scope {
  if (hasAny(text, MODIFIER_PATTERNS.intermittent)) {
    return "intermittent";
  }
  if (hasAny(text, MODIFIER_PATTERNS.persistent)) {
    return "persistent";
  }
  if (hasAny(text, MODIFIER_PATTERNS.allCases)) {
    return "all_cases";
  }
  return "single_case";
}

function inferIntent(message: string, queryMode: QueryMode, stage: Stage, polarity: Polarity): Intent {
  const text = toComparableText(message);

  if (queryMode === "guide") {
    if (hasAny(text, MODIFIER_PATTERNS.process)) {
      return "official_process";
    }

    return "baseline_check";
  }

  if (stage === "cancel_flow" && (hasAny(text, MODIFIER_PATTERNS.process) || polarity === "cannot_cancel")) {
    return "official_process";
  }

  if (hasAny(text, MODIFIER_PATTERNS.recovery)) {
    return "recovery";
  }

  return "diagnosis";
}

function normalizeSymptomSignatureFallback(message: string): SymptomSignature {
  const queryMode = inferQueryModeFallback(message);
  const text = normalizeSpace(message);
  const domain = inferDomain(text);
  const object = inferObject(text, domain);
  const stage = inferStage(text, domain);
  const polarity = inferPolarity(text, domain, stage);
  const scope = inferScope(text);
  const intent = inferIntent(text, queryMode, stage, polarity);

  return {
    queryMode,
    domain,
    object,
    stage,
    polarity,
    intent,
    scope,
    symptomSummary: text.slice(0, 120),
  };
}

function buildInterpretationPrompt(message: string) {
  return [
    new SystemMessage(
      [
        "당신은 키오스크 질의를 symptom signature로 구조화하는 분류기다.",
        "표면 단어보다 의미를 우선한다.",
        "같은 동사라도 auto-occurs 와 does-not-happen 을 구분한다.",
        "같은 행위라도 diagnosis 와 official_process 를 구분한다.",
        "반드시 query_mode, domain, object, stage, polarity, intent, scope, symptom_summary만 반환한다.",
      ].join("\n"),
    ),
    new HumanMessage(`질의: ${message}`),
  ];
}

async function interpretQuery(message: string): Promise<SymptomSignature> {
  const fallback = normalizeSymptomSignatureFallback(message);

  try {
    const model = createChatModel().withStructuredOutput(interpretationSchema, {
      name: "support_query_interpretation",
      strict: true,
    });

    const structured = await model.invoke(buildInterpretationPrompt(message));

    return {
      queryMode: structured.query_mode,
      domain: structured.domain,
      object: structured.object,
      stage: structured.stage,
      polarity: structured.polarity,
      intent: structured.intent,
      scope: structured.scope,
      symptomSummary: structured.symptom_summary || fallback.symptomSummary,
    };
  } catch {
    return fallback;
  }
}

function domainQueryTerms(domain: Domain) {
  switch (domain) {
    case "payment":
      return "결제 수납 승인";
    case "printer":
      return "프린터 출력";
    case "scanner":
      return "스캐너 인식";
    case "launcher":
      return "프로그램 실행 시작";
    case "login":
      return "로그인 계정";
    case "network":
      return "네트워크 통신";
    case "display":
      return "화면 표시";
    default:
      return "";
  }
}

function objectQueryTerms(object: EvidenceObject) {
  switch (object) {
    case "card_payment":
      return "카드 결제";
    case "payment_reader":
      return "카드 리더기 단말기";
    case "receipt_printer":
      return "영수증 프린터";
    case "document_printer":
      return "문서 프린터";
    case "barcode_scanner":
      return "바코드 스캐너";
    case "kiosk_player":
      return "키오스크 프로그램";
    case "login_account":
      return "로그인 계정";
    case "network_connection":
      return "네트워크 연결";
    case "display_panel":
      return "화면 표시";
    default:
      return "";
  }
}

function stageQueryTerms(stage: Stage) {
  switch (stage) {
    case "after_payment_or_after_approval":
      return "작업 이후 단계";
    case "cancel_flow":
      return "취소 단계";
    case "startup":
      return "시작 단계";
    case "output":
      return "출력 단계";
    case "recognition":
      return "인식 단계";
    case "login_flow":
      return "로그인 단계";
    case "network_flow":
      return "통신 단계";
    default:
      return "";
  }
}

function polarityQueryTerms(polarity: Polarity) {
  switch (polarity) {
    case "auto_cancels":
      return "자동 발생";
    case "cannot_cancel":
      return "실행되지 않음";
    case "closes_immediately":
      return "즉시 종료";
    case "outputs_twice":
      return "반복 발생";
    case "scans_twice":
      return "반복 인식";
    case "fails_to_start":
      return "시작 실패";
    case "recognition_fail":
      return "인식 실패";
    case "no_output":
      return "출력 실패";
    case "fails":
      return "실패";
    default:
      return "";
  }
}

function intentQueryTerms(intent: Intent) {
  switch (intent) {
    case "official_process":
      return "공식 절차 운영 처리";
    case "baseline_check":
      return "기준 설정 점검";
    case "recovery":
      return "복구 원복";
    default:
      return "장애 진단";
  }
}

function buildRetrievalQueries(message: string, signature: SymptomSignature) {
  const domainTerms = domainQueryTerms(signature.domain);
  const objectTerms = objectQueryTerms(signature.object);
  const stageTerms = stageQueryTerms(signature.stage);
  const polarityTerms = polarityQueryTerms(signature.polarity);
  const intentTerms = intentQueryTerms(signature.intent);
  const core = uniqueStrings(
    [
      normalizeSpace(message),
      normalizeSpace([domainTerms, objectTerms, stageTerms, polarityTerms, intentTerms].filter(Boolean).join(" ")),
    ],
    2,
  );

  const baselineQueries = uniqueStrings(
    [
      ...core,
      normalizeSpace(`${domainTerms} ${objectTerms} ${stageTerms} 기준 문서 정상 상태 설정 점검`),
    ],
    3,
  );

  const processQueries = uniqueStrings(
    [
      normalizeSpace(`${message} 공식 절차 운영 처리`),
      normalizeSpace(`${domainTerms} ${objectTerms} ${stageTerms} ${intentTerms} 운영 절차`),
    ],
    2,
  );

  const caseQueries = uniqueStrings(
    [
      normalizeSpace(`${message} 유사 사례`),
      normalizeSpace(`${domainTerms} ${objectTerms} ${stageTerms} ${polarityTerms} 사례`),
    ],
    2,
  );

  return {
    baselineQueries,
    processQueries,
    caseQueries,
  };
}

async function retrieveEvidence(message: string, signature: SymptomSignature, searchTopK: number) {
  const vectorStore = getSupabaseVectorStore();
  const queries = buildRetrievalQueries(message, signature);
  const requests: Array<{ query: string; channel: RetrievalChannel }> = [];

  for (const query of queries.baselineQueries) {
    requests.push({ query, channel: "baseline" });
  }
  for (const query of queries.processQueries) {
    requests.push({ query, channel: "process" });
  }
  if (signature.queryMode === "incident") {
    for (const query of queries.caseQueries) {
      requests.push({ query, channel: "case" });
    }
  }

  const results = await Promise.all(
    requests.map(async ({ query, channel }) => {
      const items = await vectorStore.similaritySearchWithScore(query, searchTopK);
      return items.map(
        ([document, similarity]) =>
          ({
            document,
            similarity,
            query,
            channel,
          }) satisfies RetrievalCandidate,
      );
    }),
  );

  return results.flat();
}

function getGuideBody(content: string) {
  const normalized = content.replace(/\r\n/g, "\n").trim();

  if (!normalized) {
    return "";
  }

  const paragraphs = normalized.split(/\n{2,}/);

  if (paragraphs.length > 1) {
    return paragraphs.slice(1).join("\n\n").trim();
  }

  return normalized
    .split("\n")
    .slice(2)
    .join("\n")
    .trim();
}

function classifyGuideProvenance(title: string, content: string, retrievalChannels: RetrievalChannel[], metadata: Record<string, unknown>) {
  const explicit = String(metadata.guide_kind ?? metadata.semantic_provenance ?? "");

  if (explicit === "process_doc" || explicit === "baseline_doc") {
    return explicit;
  }

  const haystack = `${title}\n${content}`;

  if (GUIDE_PROCESS_PATTERNS.some((pattern) => pattern.test(haystack))) {
    return "process_doc";
  }

  if (retrievalChannels.includes("process") && !retrievalChannels.includes("baseline")) {
    return "process_doc";
  }

  return "baseline_doc";
}

function readMetadataSignature(metadata: Record<string, unknown>): SymptomSignature | null {
  const domain = String(metadata.semantic_domain ?? "") as Domain;
  const object = String(metadata.semantic_object ?? "") as EvidenceObject;
  const stage = String(metadata.semantic_stage ?? "") as Stage;
  const polarity = String(metadata.semantic_polarity ?? "") as Polarity;
  const intent = String(metadata.semantic_intent ?? "") as Intent;
  const scope = String(metadata.semantic_scope ?? "") as Scope;
  const queryMode = String(metadata.semantic_query_mode ?? "incident") as QueryMode;

  if (
    !DOMAIN_VALUES.includes(domain) ||
    !OBJECT_VALUES.includes(object) ||
    !STAGE_VALUES.includes(stage) ||
    !POLARITY_VALUES.includes(polarity) ||
    !INTENT_VALUES.includes(intent) ||
    !SCOPE_VALUES.includes(scope) ||
    !QUERY_MODE_VALUES.includes(queryMode)
  ) {
    return null;
  }

  return {
    queryMode,
    domain,
    object,
    stage,
    polarity,
    intent,
    scope,
    symptomSummary: String(metadata.semantic_summary ?? "").slice(0, 120),
  };
}

function buildDocumentId(document: Document) {
  const metadata = document.metadata as Record<string, unknown>;
  const sourceType = String(metadata.source_type ?? "support_case");

  if (sourceType === "support_case") {
    return `case:${String(metadata.case_key ?? metadata.case_id ?? document.pageContent.slice(0, 30))}`;
  }

  return `guide:${String(metadata.source_file ?? "unknown")}:${String(metadata.guide_section_title ?? "section")}:${String(
    metadata.guide_chunk_index ?? 0,
  )}`;
}

function buildCaseSummary(metadata: Record<string, unknown>, similarity: number): SimilarCaseSummary {
  return {
    case_key: String(metadata.case_key ?? ""),
    clinic_name: String(metadata.clinic_name ?? ""),
    issue_subtype_label: String(metadata.issue_subtype_label ?? ""),
    problem_summary: String(metadata.problem_summary ?? ""),
    root_cause: String(metadata.root_cause ?? ""),
    resolution_action: String(metadata.resolution_action ?? ""),
    resolution_result: String(metadata.resolution_result ?? ""),
    similarity_score: similarity,
  };
}

function mergeRetrievedCandidates(candidates: RetrievalCandidate[]) {
  const merged = new Map<
    string,
    {
      document: Document;
      similarity: number;
      retrievalChannels: Set<RetrievalChannel>;
    }
  >();

  for (const candidate of candidates) {
    const id = buildDocumentId(candidate.document);
    const existing = merged.get(id);

    if (!existing) {
      merged.set(id, {
        document: candidate.document,
        similarity: candidate.similarity,
        retrievalChannels: new Set([candidate.channel]),
      });
      continue;
    }

    existing.similarity = Math.max(existing.similarity, candidate.similarity);
    existing.retrievalChannels.add(candidate.channel);
  }

  return [...merged.entries()].map(([id, item]) => ({
    id,
    document: item.document,
    similarity: item.similarity,
    retrievalChannels: [...item.retrievalChannels],
  }));
}

function classifyRetrievedEvidence(candidates: RetrievalCandidate[]): RetrievedEvidence[] {
  return mergeRetrievedCandidates(candidates).map(({ id, document, similarity, retrievalChannels }) => {
    const metadata = document.metadata as Record<string, unknown>;
    const sourceType = String(metadata.source_type ?? "support_case");

    if (sourceType === "support_case") {
      const summary = buildCaseSummary(metadata, similarity);
      const signature =
        readMetadataSignature(metadata) ??
        normalizeSymptomSignatureFallback(
        [
          summary.problem_summary,
          summary.root_cause,
          summary.resolution_action,
          summary.issue_subtype_label,
        ]
          .filter(Boolean)
          .join(" "),
        );

      return {
        id,
        similarity,
        retrievalChannels,
        sourceType: "case",
        provenance: "case_history",
        title: summary.problem_summary || summary.issue_subtype_label || "유사 사례",
        content: document.pageContent,
        sourceFile: String(metadata.source_file ?? ""),
        signature,
        summary,
        customerReplyReference: String(metadata.customer_reply_reference ?? ""),
        latestActionReference: String(metadata.latest_action_reference ?? ""),
        qualityTier: String(metadata.quality_tier ?? ""),
        alignmentScore: 0,
      } satisfies RetrievedEvidence;
    }

    const title = String(metadata.guide_section_title ?? "");
    const content = getGuideBody(document.pageContent);
    const provenance = classifyGuideProvenance(title, content, retrievalChannels, metadata);
    const signature = readMetadataSignature(metadata) ?? normalizeSymptomSignatureFallback(`${title}\n${content}`);

    return {
      id,
      similarity,
      retrievalChannels,
      sourceType: "guide",
      provenance,
      title,
      content,
      sourceFile: String(metadata.source_file ?? ""),
      signature,
      alignmentScore: 0,
    } satisfies RetrievedEvidence;
  });
}

function isStrongDomainMismatch(query: SymptomSignature, evidence: RetrievedEvidence) {
  return query.domain !== "etc" && evidence.signature.domain !== "etc" && query.domain !== evidence.signature.domain;
}

function isStrongObjectMismatch(query: SymptomSignature, evidence: RetrievedEvidence) {
  return query.object !== "etc" && evidence.signature.object !== "etc" && query.object !== evidence.signature.object;
}

function isStrongStageMismatch(query: SymptomSignature, evidence: RetrievedEvidence) {
  const left = query.stage;
  const right = evidence.signature.stage;

  if (left === "etc" || right === "etc" || left === right) {
    return false;
  }

  const mismatchPairs = new Set([
    "after_payment_or_after_approval|cancel_flow",
    "cancel_flow|after_payment_or_after_approval",
    "startup|output",
    "output|startup",
    "startup|recognition",
    "recognition|startup",
    "output|recognition",
    "recognition|output",
  ]);

  return mismatchPairs.has(`${left}|${right}`);
}

function isStrongPolarityMismatch(query: SymptomSignature, evidence: RetrievedEvidence) {
  const left = query.polarity;
  const right = evidence.signature.polarity;

  if (left === "unknown" || right === "unknown" || left === right) {
    return false;
  }

  const mismatchPairs = new Set([
    "auto_cancels|cannot_cancel",
    "cannot_cancel|auto_cancels",
    "outputs_twice|no_output",
    "no_output|outputs_twice",
    "scans_twice|recognition_fail",
    "recognition_fail|scans_twice",
    "closes_immediately|fails_to_start",
    "fails_to_start|closes_immediately",
  ]);

  return mismatchPairs.has(`${left}|${right}`);
}

function isGenericStage(stage: Stage) {
  return stage === "etc" || stage === "in_progress" || stage === "before_action" || stage === "after_action";
}

function stageFamily(stage: Stage) {
  switch (stage) {
    case "after_payment_or_after_approval":
      return "post_action";
    case "cancel_flow":
      return "cancel";
    case "startup":
      return "startup";
    case "output":
      return "output";
    case "recognition":
      return "recognition";
    case "login_flow":
      return "login";
    case "network_flow":
      return "network";
    case "before_action":
    case "in_progress":
      return "active";
    case "after_action":
      return "post_action_generic";
    default:
      return "generic";
  }
}

function isGenericPolarity(polarity: Polarity) {
  return polarity === "unknown" || polarity === "fails" || polarity === "does_not_happen" || polarity === "repeats";
}

function polarityFamily(polarity: Polarity) {
  switch (polarity) {
    case "auto_cancels":
    case "auto_happens":
    case "closes_immediately":
      return "unexpected_auto";
    case "cannot_cancel":
      return "blocked_process";
    case "outputs_twice":
    case "scans_twice":
    case "repeats":
      return "repeated";
    case "no_output":
    case "recognition_fail":
    case "fails_to_start":
    case "fails":
    case "does_not_happen":
      return "failure";
    default:
      return "generic";
  }
}

function computeStageAlignmentBonus(query: SymptomSignature, evidence: RetrievedEvidence) {
  if (query.stage === evidence.signature.stage) {
    return 14;
  }

  if (query.stage === "etc" || evidence.signature.stage === "etc") {
    return 0;
  }

  if (isGenericStage(evidence.signature.stage) && !isGenericStage(query.stage)) {
    return evidence.provenance === "case_history" ? -22 : -12;
  }

  if (stageFamily(query.stage) === stageFamily(evidence.signature.stage)) {
    return 6;
  }

  return -10;
}

function computePolarityAlignmentBonus(query: SymptomSignature, evidence: RetrievedEvidence) {
  if (query.polarity === evidence.signature.polarity) {
    return 14;
  }

  if (query.polarity === "unknown" || evidence.signature.polarity === "unknown") {
    return 0;
  }

  if (isGenericPolarity(evidence.signature.polarity) && !isGenericPolarity(query.polarity)) {
    return evidence.provenance === "case_history" ? -20 : -10;
  }

  if (polarityFamily(query.polarity) === polarityFamily(evidence.signature.polarity)) {
    return 6;
  }

  return -10;
}

function computeAlignmentScore(message: string, query: SymptomSignature, evidence: RetrievedEvidence) {
  if (isStrongDomainMismatch(query, evidence)) {
    return { included: false, score: -100, reason: "domain mismatch" };
  }

  if (isStrongObjectMismatch(query, evidence)) {
    return { included: false, score: -90, reason: "object mismatch" };
  }

  if (isStrongStageMismatch(query, evidence)) {
    return { included: false, score: -80, reason: "stage mismatch" };
  }

  if (isStrongPolarityMismatch(query, evidence)) {
    return { included: false, score: -80, reason: "polarity mismatch" };
  }

  let score = evidence.similarity * 100;
  const queryTerms = extractTerms(message);
  const evidenceTerms = extractTerms(`${evidence.title}\n${evidence.content}`);

  score += countOverlap(queryTerms, evidenceTerms) * 2;
  score += query.domain === evidence.signature.domain ? 20 : 0;
  score += query.object === evidence.signature.object ? 16 : 0;
  score += computeStageAlignmentBonus(query, evidence);
  score += computePolarityAlignmentBonus(query, evidence);
  score += evidence.retrievalChannels.includes("baseline") && evidence.provenance === "baseline_doc" ? 8 : 0;
  score += evidence.retrievalChannels.includes("process") && evidence.provenance === "process_doc" ? 8 : 0;
  score += evidence.retrievalChannels.includes("case") && evidence.provenance === "case_history" ? 8 : 0;

  if (query.intent === "official_process") {
    score += evidence.provenance === "process_doc" ? 24 : 0;
    score -= evidence.provenance === "case_history" ? 12 : 0;
  } else if (query.intent === "diagnosis") {
    score += evidence.provenance === "baseline_doc" ? 18 : 0;
    score -= evidence.provenance === "process_doc" ? 8 : 0;
  }

  if (
    query.intent === "diagnosis" &&
    evidence.provenance === "process_doc" &&
    evidence.signature.stage === "cancel_flow" &&
    query.stage !== "cancel_flow"
  ) {
    return { included: false, score: -70, reason: "diagnosis/process mismatch" };
  }

  return {
    included: score >= 35,
    score,
    reason: score >= 35 ? undefined : "low alignment",
  };
}

function applyRelevanceGate(message: string, signature: SymptomSignature, evidenceItems: RetrievedEvidence[]) {
  const included: RetrievedEvidence[] = [];
  const excluded: RetrievedEvidence[] = [];

  for (const item of evidenceItems) {
    const alignment = computeAlignmentScore(message, signature, item);
    const nextItem = {
      ...item,
      alignmentScore: alignment.score,
      exclusionReason: alignment.reason,
    };

    if (alignment.included) {
      included.push(nextItem);
    } else {
      excluded.push(nextItem);
    }
  }

  included.sort((left, right) => {
    if (PROVENANCE_PRIORITY[left.provenance] !== PROVENANCE_PRIORITY[right.provenance]) {
      return PROVENANCE_PRIORITY[left.provenance] - PROVENANCE_PRIORITY[right.provenance];
    }

    return right.alignmentScore - left.alignmentScore;
  });

  return { included, excluded };
}

function buildGroundingPrompt(signature: SymptomSignature, evidenceBundle: GroundedEvidenceBundle) {
  const lines = [
    ...evidenceBundle.baselineDocs.map((item) => `- ${item.id} | baseline_doc | ${item.title}`),
    ...evidenceBundle.processDocs.map((item) => `- ${item.id} | process_doc | ${item.title}`),
    ...evidenceBundle.caseHistories.map((item) => `- ${item.id} | case_history | ${item.title}`),
  ].join("\n");

  return [
    new SystemMessage(
      [
        "당신은 evidence grounding planner다.",
        "우선순위는 baseline_doc > process_doc > case_history 이다.",
        "domain/object/stage/polarity가 맞지 않는 사례는 제외한다.",
        "diagnosis 질의에 process_doc을 원인 근거처럼 올리지 않는다.",
        "official_process 질의에 case_history를 절차 근거처럼 올리지 않는다.",
        "선택 가능한 id만 반환한다.",
      ].join("\n"),
    ),
    new HumanMessage(
      [
        `query_mode=${signature.queryMode}`,
        `domain=${signature.domain}`,
        `object=${signature.object}`,
        `stage=${signature.stage}`,
        `polarity=${signature.polarity}`,
        `intent=${signature.intent}`,
        "",
        "후보 evidence:",
        lines || "- 없음",
      ].join("\n"),
    ),
  ];
}

async function refineGroundingWithPrompt(signature: SymptomSignature, bundle: GroundedEvidenceBundle) {
  if (bundle.baselineDocs.length + bundle.processDocs.length + bundle.caseHistories.length === 0) {
    return bundle;
  }

  try {
    const model = createChatModel().withStructuredOutput(groundingSchema, {
      name: "support_grounding_plan",
      strict: true,
    });
    const response = await model.invoke(buildGroundingPrompt(signature, bundle));
    const excludeCases = new Set(response.exclude_case_ids);

    const reorder = (items: RetrievedEvidence[], ids: string[]) => {
      const itemMap = new Map(items.map((item) => [item.id, item]));
      const ordered: RetrievedEvidence[] = [];
      const used = new Set<string>();

      for (const id of ids) {
        const item = itemMap.get(id);

        if (!item || used.has(id)) {
          continue;
        }

        ordered.push(item);
        used.add(id);
      }

      for (const item of items) {
        if (!used.has(item.id)) {
          ordered.push(item);
        }
      }

      return ordered;
    };

    return {
      baselineDocs: reorder(bundle.baselineDocs, response.baseline_ids),
      processDocs: reorder(bundle.processDocs, response.process_ids),
      caseHistories: reorder(
        bundle.caseHistories.filter((item) => !excludeCases.has(item.id)),
        response.case_ids,
      ),
      excluded: bundle.excluded,
      fallbackNeeded: bundle.fallbackNeeded || response.fallback_needed,
    } satisfies GroundedEvidenceBundle;
  } catch {
    return bundle;
  }
}

async function groundEvidence(message: string, signature: SymptomSignature, evidenceItems: RetrievedEvidence[]) {
  const gated = applyRelevanceGate(message, signature, evidenceItems);
  const baselineDocs = gated.included.filter((item) => item.provenance === "baseline_doc").slice(0, 6);
  const processDocs = gated.included.filter((item) => item.provenance === "process_doc").slice(0, 6);
  const caseHistories = gated.included.filter((item) => item.provenance === "case_history").slice(0, 4);
  const initial = {
    baselineDocs,
    processDocs,
    caseHistories,
    excluded: gated.excluded,
    fallbackNeeded: baselineDocs.length + processDocs.length === 0,
  } satisfies GroundedEvidenceBundle;

  return refineGroundingWithPrompt(signature, initial);
}

function extractEvidenceLines(evidence: RetrievedEvidence) {
  if (evidence.provenance === "case_history" && evidence.summary) {
    return [
      evidence.summary.problem_summary,
      evidence.summary.root_cause,
      evidence.summary.resolution_action,
      evidence.summary.issue_subtype_label,
    ]
      .map((line) => normalizeLine(line))
      .filter(Boolean);
  }

  return evidence.content
    .split("\n")
    .map((line) => normalizeLine(line))
    .filter(
      (line) =>
        Boolean(line) &&
        !/^#+/.test(line) &&
        !/^적용 대상/i.test(line) &&
        !/^경로$/i.test(line) &&
        !/^설치 절차$/i.test(line),
    );
}

function scoreEvidenceLine(message: string, evidence: RetrievedEvidence, line: string) {
  const queryTerms = extractTerms(message);
  const lineTerms = extractTerms(line);
  let score = evidence.alignmentScore + countOverlap(queryTerms, lineTerms) * 3;

  if (/`[^`]+`|\b\d{1,3}(?:\.\d{1,3}){3}\b|ON|OFF/i.test(line)) {
    score += 6;
  }

  if (/확인|조회|설정|복구|요청|처리|절차/i.test(line)) {
    score += 4;
  }

  if (/테스트|시험|점검|체크/i.test(line)) {
    score += 10;
  }

  if (/기본|default|드라이버|driver|설치|재설치|포트|속성|호스트/i.test(line)) {
    score += 8;
  }

  return score;
}

function classifyLineFeatures(line: string): LineFeature[] {
  const features = new Set<LineFeature>();

  if (/>|경로|메뉴|화면/i.test(line)) {
    features.add("menu_path");
  }
  if (/테스트|시험|점검|체크/i.test(line)) {
    features.add("test_step");
  }
  if (/설정|기본|ON|OFF|포트|속성|호스트|IP|유형|선택/i.test(line)) {
    features.add("config_step");
  }
  if (/설치|재설치|드라이버|driver|guide|가이드|pdf/i.test(line)) {
    features.add("install_step");
    features.add("guide_hint");
  }
  if (/절차|프로세스|요청|접수|전달|후속/i.test(line)) {
    features.add("process_step");
  }
  if (/원인|이탈|불일치|상이|누락|미반영|고장|오류|가능성/i.test(line)) {
    features.add("cause_hint");
  }
  if (/`[^`]+`|\.md\b|\.pdf\b/i.test(line)) {
    features.add("guide_hint");
  }

  return [...features];
}

function pickTopLines(message: string, evidence: RetrievedEvidence[], maxItems: number, maxPerEvidence = 2) {
  const scored = evidence.flatMap((item) =>
    extractEvidenceLines(item)
      .filter((line) => line.length >= 4)
      .map((line) => ({
        text: line,
        evidenceId: item.id,
        provenance: item.provenance,
        score: scoreEvidenceLine(message, item, line),
        features: classifyLineFeatures(line),
      })),
  );

  scored.sort((left, right) => right.score - left.score);

  const perEvidenceCount = new Map<string, number>();
  const selected: Array<{ text: string; provenance: EvidenceProvenance; score: number; evidenceId?: string; features: LineFeature[] }> = [];
  const seen = new Set<string>();

  for (const item of scored) {
    const key = toComparableText(item.text);
    const evidenceCount = perEvidenceCount.get(item.evidenceId ?? "") ?? 0;

    if (seen.has(key) || evidenceCount >= maxPerEvidence) {
      continue;
    }

    seen.add(key);
    perEvidenceCount.set(item.evidenceId ?? "", evidenceCount + 1);
    selected.push(item);

    if (selected.length >= maxItems) {
      break;
    }
  }

  return selected;
}

function lineToCause(line: string, title: string) {
  const cleaned = normalizeLine(
    line
      .replace(/(인가|인 가|여부)$/i, "")
      .replace(/확인(합니다)?$/i, "")
      .replace(/조회(합니다)?$/i, ""),
  );

  if (cleaned) {
    return `${cleaned} 기준값 이탈 가능성`;
  }

  return `${normalizeSpace(title)} 기준 설정 이탈 가능성`;
}

function buildSuspectedCauses(message: string, signature: SymptomSignature, bundle: GroundedEvidenceBundle) {
  const candidates: SectionCandidate[] = [];

  if (signature.intent === "official_process") {
    if (bundle.processDocs[0]) {
      candidates.push({
        text: "원인 추정보다 공식 운영 절차와 처리 기준 확인이 우선인 건으로 보입니다.",
        provenance: "process_doc",
        score: bundle.processDocs[0].alignmentScore,
        evidenceId: bundle.processDocs[0].id,
        features: ["process_step"],
      });
    }
  } else {
    for (const item of bundle.baselineDocs.slice(0, 2)) {
      const topLine = pickTopLines(message, [item], 1, 1)[0];
      candidates.push({
        text: lineToCause(topLine?.text ?? item.title, item.title),
        provenance: "baseline_doc",
        score: item.alignmentScore,
        evidenceId: item.id,
        features: ["cause_hint", ...(topLine?.features ?? [])],
      });
    }

    if (candidates.length === 0 && bundle.caseHistories[0]?.summary?.root_cause) {
      candidates.push({
        text: `${normalizeLine(bundle.caseHistories[0].summary.root_cause)} 가능성`,
        provenance: "case_history",
        score: bundle.caseHistories[0].alignmentScore,
        evidenceId: bundle.caseHistories[0].id,
        features: ["cause_hint"],
      });
    }
  }

  if (candidates.length === 0) {
    candidates.push({
      text: "현재 확보된 기준문서와 사례만으로는 특정 원인을 단정하기 어렵습니다.",
      provenance: "inferred",
      score: 1,
      features: [],
    });
  }

  return candidates;
}

function buildLineCandidates(message: string, evidence: RetrievedEvidence[], maxItems: number, maxPerEvidence = 2) {
  return pickTopLines(message, evidence, maxItems, maxPerEvidence).map((item) => ({
    text: item.text,
    provenance: item.provenance,
    score: item.score,
    evidenceId: item.evidenceId,
    features: item.features,
  }));
}

function findBestFeatureCandidate(message: string, evidence: RetrievedEvidence[], feature: LineFeature) {
  return evidence
    .flatMap((item) =>
      extractEvidenceLines(item)
        .map((line) => ({
          text: line,
          provenance: item.provenance,
          score: scoreEvidenceLine(message, item, line),
          evidenceId: item.id,
          features: classifyLineFeatures(line),
        }))
        .filter((item) => item.features.includes(feature)),
    )
    .sort((left, right) => right.score - left.score)[0];
}

function buildCheckCandidates(message: string, signature: SymptomSignature, bundle: GroundedEvidenceBundle) {
  const evidencePriority =
    signature.intent === "official_process"
      ? [...bundle.processDocs, ...bundle.baselineDocs]
      : [...bundle.baselineDocs, ...bundle.processDocs, ...bundle.caseHistories];
  const candidates = buildLineCandidates(message, evidencePriority, SECTION_LIMITS.checks + 4, 2);

  if (!candidates.some((item) => item.features?.includes("test_step"))) {
    const bestTestCandidate = findBestFeatureCandidate(message, evidencePriority, "test_step");

    if (bestTestCandidate) {
      candidates.unshift({
        ...bestTestCandidate,
        score: bestTestCandidate.score + 12,
      });
    }
  }

  if (candidates.length === 0) {
    candidates.push({
      text: "증상과 직접 맞닿은 기준문서가 부족하므로 실제 메뉴 경로와 설정값을 추가로 확인해야 합니다.",
      provenance: "inferred",
      score: 1,
      evidenceId: undefined,
      features: [],
    });
  }

  return candidates;
}

function buildActionCandidates(message: string, signature: SymptomSignature, bundle: GroundedEvidenceBundle) {
  const primaryEvidence =
    signature.intent === "official_process"
      ? [...bundle.processDocs, ...bundle.baselineDocs]
      : [...bundle.baselineDocs, ...bundle.processDocs, ...bundle.caseHistories];
  const candidates = buildLineCandidates(message, primaryEvidence, SECTION_LIMITS.actions + 4, 2);

  if (candidates.length === 0) {
    candidates.push({
      text: "문서 근거가 보강되기 전까지는 일반적인 OS 추정보다 실제 화면명과 설정값 확보를 우선합니다.",
      provenance: "inferred",
      score: 1,
      evidenceId: undefined,
      features: [],
    });
  }

  return candidates;
}

function candidateSpecificityScore(text: string) {
  let score = 0;

  if (/`[^`]+`|\b\d{1,3}(?:\.\d{1,3}){3}\b|ON|OFF/i.test(text)) {
    score += 10;
  }

  if (/테스트|시험|기본|default|경로|메뉴|설치|재설치|드라이버|driver|포트|속성|설정|가이드/i.test(text)) {
    score += 8;
  }

  if (/불일치|상이|누락|미반영|이탈|원인|가능성/i.test(text)) {
    score += 6;
  }

  return score;
}

function sectionFeatureBias(candidate: SectionCandidate, section: IncidentSectionKey) {
  const features = new Set(candidate.features ?? []);

  if (section === "suspected_causes") {
    return (
      (features.has("cause_hint") ? 14 : 0) +
      (features.has("config_step") ? 6 : 0) -
      (features.has("process_step") ? 10 : 0) -
      (features.has("install_step") ? 4 : 0)
    );
  }

  if (section === "checks") {
    return (
      (features.has("test_step") ? 40 : 0) +
      (features.has("config_step") ? 18 : 0) +
      (features.has("menu_path") ? 12 : 0) +
      (features.has("guide_hint") ? 4 : 0) -
      (features.has("process_step") ? 6 : 0)
    );
  }

  return (
    (features.has("config_step") ? 18 : 0) +
    (features.has("install_step") ? 16 : 0) +
    (features.has("guide_hint") ? 8 : 0) +
    (features.has("process_step") ? 10 : 0) +
    (features.has("test_step") ? 4 : 0)
  );
}

function sectionPriorityScore(candidate: SectionCandidate, section: IncidentSectionKey) {
  const specificity = candidateSpecificityScore(candidate.text);
  const provenanceBias =
    candidate.provenance === "baseline_doc"
      ? 18
      : candidate.provenance === "process_doc"
        ? 10
        : candidate.provenance === "case_history"
          ? 0
          : -24;
  const sectionBias = section === "suspected_causes" && candidate.provenance === "case_history" ? 4 : 0;
  const casePenalty = section !== "suspected_causes" && candidate.provenance === "case_history" ? -8 : 0;

  return candidate.score + provenanceBias + sectionBias + casePenalty + specificity + sectionFeatureBias(candidate, section);
}

function sanitizeCandidates(candidates: SectionCandidate[], section: IncidentSectionKey, hasDocumentGrounding: boolean) {
  const sorted = [...candidates].sort((left, right) => {
    const rightPriority = sectionPriorityScore(right, section);
    const leftPriority = sectionPriorityScore(left, section);

    if (rightPriority !== leftPriority) {
      return rightPriority - leftPriority;
    }

    if (PROVENANCE_PRIORITY[left.provenance] !== PROVENANCE_PRIORITY[right.provenance]) {
      return PROVENANCE_PRIORITY[left.provenance] - PROVENANCE_PRIORITY[right.provenance];
    }

    return right.score - left.score;
  });

  const output: SectionCandidate[] = [];
  const seen = new Set<string>();

  for (const candidate of sorted) {
    const normalized = toComparableText(candidate.text);

    if (!normalized || seen.has(normalized)) {
      continue;
    }

    if (
      hasDocumentGrounding &&
      candidate.provenance === "inferred" &&
      GENERIC_GUESS_PATTERNS.some((pattern) => pattern.test(candidate.text))
    ) {
      continue;
    }

    if (section === "actions" && output.length < 3 && candidate.provenance === "case_history") {
      continue;
    }

    seen.add(normalized);
    output.push(candidate);

    if (output.length >= SECTION_LIMITS[section]) {
      break;
    }
  }

  return output;
}

function buildIncidentDraft(message: string, signature: SymptomSignature, bundle: GroundedEvidenceBundle): IncidentDraft {
  const suspectedCauses = sanitizeCandidates(
    buildSuspectedCauses(message, signature, bundle),
    "suspected_causes",
    bundle.baselineDocs.length + bundle.processDocs.length > 0,
  );
  const checks = sanitizeCandidates(
    buildCheckCandidates(message, signature, bundle),
    "checks",
    bundle.baselineDocs.length + bundle.processDocs.length > 0,
  );
  const actions = sanitizeCandidates(
    buildActionCandidates(message, signature, bundle),
    "actions",
    bundle.baselineDocs.length + bundle.processDocs.length > 0,
  );
  const fallbackUsed = bundle.fallbackNeeded || (checks[0]?.provenance === "inferred" && actions[0]?.provenance === "inferred");

  return {
    suspectedCauses,
    checks,
    actions,
    fallbackUsed,
  };
}

function buildCompositionPrompt(signature: SymptomSignature, draft: IncidentDraft) {
  const serializeSection = (items: SectionCandidate[]) =>
    items.map((item, index) => `${index + 1}. (${item.provenance}) ${item.text}`).join("\n");

  return [
    new SystemMessage(
      [
        "당신은 grounded answer composer다.",
        "이미 relevance gate를 통과한 후보만 받는다.",
        "새 사실을 추가하지 말고, 의미를 바꾸지 않는 범위에서만 문장을 다듬는다.",
        "baseline_doc을 process_doc보다 우선하고, case_history는 보조적으로만 유지한다.",
      ].join("\n"),
    ),
    new HumanMessage(
      [
        `query_mode=${signature.queryMode}`,
        `domain=${signature.domain}`,
        `object=${signature.object}`,
        `stage=${signature.stage}`,
        `polarity=${signature.polarity}`,
        `intent=${signature.intent}`,
        "",
        "[suspected_causes]",
        serializeSection(draft.suspectedCauses),
        "",
        "[checks]",
        serializeSection(draft.checks),
        "",
        "[actions]",
        serializeSection(draft.actions),
      ].join("\n"),
    ),
  ];
}

async function composeIncidentDraft(signature: SymptomSignature, draft: IncidentDraft) {
  try {
    const model = createChatModel().withStructuredOutput(compositionSchema, {
      name: "support_answer_composition",
      strict: true,
    });
    const response = await model.invoke(buildCompositionPrompt(signature, draft));

    if (
      response.suspected_causes.length === draft.suspectedCauses.length &&
      response.checks.length === draft.checks.length &&
      response.actions.length === draft.actions.length
    ) {
      return {
        suspectedCauses: draft.suspectedCauses.map((item, index) => ({
          ...item,
          text: response.suspected_causes[index] ?? item.text,
        })),
        checks: draft.checks.map((item, index) => ({
          ...item,
          text: response.checks[index] ?? item.text,
        })),
        actions: draft.actions.map((item, index) => ({
          ...item,
          text: response.actions[index] ?? item.text,
        })),
        fallbackUsed: draft.fallbackUsed,
      } satisfies IncidentDraft;
    }
  } catch {
    return draft;
  }

  return draft;
}

function buildBaselineReference(message: string, bundle: GroundedEvidenceBundle, draft?: IncidentDraft): BaselineReference | null {
  const docs = bundle.baselineDocs.length > 0 ? bundle.baselineDocs.slice(0, 4) : bundle.processDocs.slice(0, 3);

  if (docs.length === 0) {
    return null;
  }

  const referencedEvidenceIds = new Set(
    [draft?.checks ?? [], draft?.actions ?? [], draft?.suspectedCauses ?? []]
      .flat()
      .map((item) => item.evidenceId)
      .filter((item): item is string => Boolean(item)),
  );

  const scoredLines = docs.flatMap((item) =>
    extractEvidenceLines(item)
      .filter((line) => line.length >= 4)
      .map((line) => {
        const features = classifyLineFeatures(line);
        const featureSet = new Set(features);
        const featureBoost =
          (featureSet.has("menu_path") ? 10 : 0) +
          (featureSet.has("test_step") ? 12 : 0) +
          (featureSet.has("config_step") ? 12 : 0) +
          (featureSet.has("install_step") ? 10 : 0) +
          (featureSet.has("guide_hint") ? 6 : 0);

        return {
          text: line,
          evidenceId: item.id,
          score:
            scoreEvidenceLine(message, item, line) +
            featureBoost +
            (referencedEvidenceIds.has(item.id) ? 12 : 0) +
            (item.provenance === "baseline_doc" ? 8 : 0),
          features,
        };
      }),
  );

  scoredLines.sort((left, right) => right.score - left.score);

  const selected: typeof scoredLines = [];
  const seenText = new Set<string>();
  const coveredFeatures = new Set<LineFeature>();
  const featureOrder: LineFeature[] = ["menu_path", "test_step", "config_step", "install_step", "guide_hint"];

  for (const feature of featureOrder) {
    const candidate = scoredLines.find((item) => item.features.includes(feature) && !seenText.has(toComparableText(item.text)));

    if (!candidate) {
      continue;
    }

    selected.push(candidate);
    seenText.add(toComparableText(candidate.text));
    candidate.features.forEach((item) => coveredFeatures.add(item));

    if (selected.length >= 6) {
      break;
    }
  }

  for (const item of scoredLines) {
    const key = toComparableText(item.text);

    if (seenText.has(key)) {
      continue;
    }

    selected.push(item);
    seenText.add(key);
    item.features.forEach((feature) => coveredFeatures.add(feature));

    if (selected.length >= 6) {
      break;
    }
  }

  if (!selected.some((item) => item.features.includes("test_step"))) {
    const bestTestLine = scoredLines.find((item) => item.features.includes("test_step"));

    if (bestTestLine && !seenText.has(toComparableText(bestTestLine.text))) {
      selected.unshift(bestTestLine);
      seenText.add(toComparableText(bestTestLine.text));
    }
  }

  return {
    source_titles: uniqueStrings(docs.map((item) => item.title), 3),
    source_files: uniqueStrings(docs.map((item) => item.sourceFile), 2),
    excerpts: selected.slice(0, 6).map((item) => item.text),
  };
}

function buildGuideResponse(
  signature: SymptomSignature,
  bundle: GroundedEvidenceBundle,
  highThreshold: number,
  lowThreshold: number,
): ChatApiResponse {
  const docs = signature.intent === "official_process" ? [...bundle.processDocs, ...bundle.baselineDocs] : [...bundle.baselineDocs, ...bundle.processDocs];
  const topSimilarity = docs[0]?.similarity ?? null;
  const confidenceLevel = determineConfidenceLevel(topSimilarity, highThreshold, lowThreshold);

  if (docs.length === 0) {
    return {
      query_mode: "guide",
      suspected_causes: [],
      checks: [],
      actions: ["[추정 보완] 실제 화면명, 메뉴 경로, 설정값을 추가로 확인해 주세요."],
      baseline_reference: null,
      confidence_level: confidenceLevel,
      confidence_note: buildGuideConfidenceNote(confidenceLevel, topSimilarity, 0),
      similar_case_count: 0,
      top_similarity: topSimilarity,
      similar_cases: [],
      fallback_used: true,
    };
  }

  const actions = uniqueStrings(
    buildLineCandidates(signature.symptomSummary, docs, SECTION_LIMITS.actions, 2).map((item) =>
      withProvenanceLabel(item.text, item.provenance),
    ),
    SECTION_LIMITS.actions,
  );

  return {
    query_mode: "guide",
    suspected_causes: [],
    checks: [],
    actions: actions.length > 0 ? actions : ["[추정 보완] 관련 문서의 직접 근거가 부족해 메뉴 경로 확인이 먼저 필요합니다."],
    baseline_reference: buildBaselineReference(signature.symptomSummary, {
      ...bundle,
      baselineDocs: docs,
      processDocs: [],
    }),
    confidence_level: confidenceLevel,
    confidence_note: buildGuideConfidenceNote(confidenceLevel, topSimilarity, docs.length),
    similar_case_count: 0,
    top_similarity: topSimilarity,
    similar_cases: [],
    fallback_used: false,
  };
}

function finalizeIncidentResponse(
  draft: IncidentDraft,
  signature: SymptomSignature,
  bundle: GroundedEvidenceBundle,
  highThreshold: number,
  lowThreshold: number,
): ChatApiResponse {
  const similarCaseEvidence =
    signature.intent === "official_process"
      ? []
      : bundle.caseHistories.filter(
          (item) => !item.exclusionReason && item.signature.stage === signature.stage && item.signature.polarity === signature.polarity,
        );
  const similarCases = similarCaseEvidence.slice(0, 3).flatMap((item) => (item.summary ? [item.summary] : []));
  const topSimilarity =
    bundle.baselineDocs[0]?.similarity ?? bundle.processDocs[0]?.similarity ?? similarCaseEvidence[0]?.similarity ?? null;
  const confidenceLevel = determineConfidenceLevel(topSimilarity, highThreshold, lowThreshold);
  const baselineReference = buildBaselineReference(signature.symptomSummary, bundle, draft);

  return {
    query_mode: "incident",
    suspected_causes: uniqueStrings(
      draft.suspectedCauses.map((item) => withProvenanceLabel(item.text, item.provenance)),
      SECTION_LIMITS.suspected_causes,
    ),
    checks: uniqueStrings(
      draft.checks.map((item) => withProvenanceLabel(item.text, item.provenance)),
      SECTION_LIMITS.checks,
    ),
    actions: uniqueStrings(
      draft.actions.map((item) => withProvenanceLabel(item.text, item.provenance)),
      SECTION_LIMITS.actions,
    ),
    baseline_reference: baselineReference,
    confidence_level: confidenceLevel,
    confidence_note: buildConfidenceNote(
      confidenceLevel,
      topSimilarity,
      bundle.baselineDocs.length,
      bundle.processDocs.length,
      similarCases.length,
    ),
    similar_case_count: similarCases.length,
    top_similarity: topSimilarity,
    similar_cases: similarCases,
    fallback_used: confidenceLevel === "low" || draft.fallbackUsed,
  };
}

function buildDeterministicGrounding(message: string, signature: SymptomSignature, evidenceItems: RetrievedEvidence[]) {
  const gated = applyRelevanceGate(message, signature, evidenceItems);

  return {
    baselineDocs: gated.included.filter((item) => item.provenance === "baseline_doc").slice(0, 6),
    processDocs: gated.included.filter((item) => item.provenance === "process_doc").slice(0, 6),
    caseHistories: gated.included.filter((item) => item.provenance === "case_history").slice(0, 4),
    excluded: gated.excluded,
    fallbackNeeded:
      gated.included.filter((item) => item.provenance === "baseline_doc" || item.provenance === "process_doc").length === 0,
  } satisfies GroundedEvidenceBundle;
}

function runDeterministicIncidentPipeline(
  message: string,
  candidates: Array<{ document: Document; similarity: number; channel?: RetrievalChannel; query?: string }>,
  thresholds = { high: 0.78, low: 0.58 },
  signatureOverride?: Partial<SymptomSignature>,
) {
  const baseSignature = normalizeSymptomSignatureFallback(message);
  const signature = {
    ...baseSignature,
    ...signatureOverride,
    symptomSummary: signatureOverride?.symptomSummary ?? baseSignature.symptomSummary,
  } satisfies SymptomSignature;
  const evidence = classifyRetrievedEvidence(
    candidates.map((item) => ({
      document: item.document,
      similarity: item.similarity,
      channel: item.channel ?? "baseline",
      query: item.query ?? message,
    })),
  );
  const grounded = buildDeterministicGrounding(message, signature, evidence);
  const draft = buildIncidentDraft(message, signature, grounded);
  const response = finalizeIncidentResponse(draft, signature, grounded, thresholds.high, thresholds.low);

  return {
    signature,
    evidence,
    grounded,
    draft,
    response,
  };
}

export async function analyzeSupportIssue(message: string, topK?: number): Promise<ChatApiResponse> {
  const config = getRequiredServerConfig();
  const baseTopK = topK ?? config.RAG_TOP_K;
  const searchTopK = Math.max(baseTopK * 3, 12);
  const signature = await interpretQuery(message);
  const retrieved = await retrieveEvidence(message, signature, searchTopK);
  const evidence = classifyRetrievedEvidence(retrieved);
  const grounded = await groundEvidence(message, signature, evidence);

  if (signature.queryMode === "guide") {
    return buildGuideResponse(signature, grounded, config.HIGH_CONFIDENCE_THRESHOLD, config.LOW_CONFIDENCE_THRESHOLD);
  }

  const draft = buildIncidentDraft(message, signature, grounded);
  const composed = await composeIncidentDraft(signature, draft);

  return finalizeIncidentResponse(
    composed,
    signature,
    grounded,
    config.HIGH_CONFIDENCE_THRESHOLD,
    config.LOW_CONFIDENCE_THRESHOLD,
  );
}

export const __testing = {
  normalizeSymptomSignatureFallback,
  classifyRetrievedEvidence,
  applyRelevanceGate,
  buildIncidentDraft,
  buildDeterministicGrounding,
  runDeterministicIncidentPipeline,
  buildGuideResponse,
  finalizeIncidentResponse,
};
