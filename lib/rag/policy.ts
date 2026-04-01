import type { QueryMode } from "@/lib/types";

export const QUERY_MODE_VALUES = ["incident", "guide"] as const;
export const DOMAIN_VALUES = ["payment", "printer", "scanner", "launcher", "login", "network", "display", "etc"] as const;
export const OBJECT_VALUES = [
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
export const STAGE_VALUES = [
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
export const POLARITY_VALUES = [
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
export const INTENT_VALUES = ["diagnosis", "baseline_check", "official_process", "recovery"] as const;
export const SCOPE_VALUES = ["single_case", "all_cases", "intermittent", "persistent"] as const;

export type Domain = (typeof DOMAIN_VALUES)[number];
export type EvidenceObject = (typeof OBJECT_VALUES)[number];
export type Stage = (typeof STAGE_VALUES)[number];
export type Polarity = (typeof POLARITY_VALUES)[number];
export type Intent = (typeof INTENT_VALUES)[number];
export type Scope = (typeof SCOPE_VALUES)[number];
export type GuideKind = "baseline_doc" | "process_doc" | "install_doc" | "reference_doc" | "meta_doc";
export type IncidentArchetype =
  | "configuration_sensitive"
  | "known_case_pattern"
  | "physical_device"
  | "process_execution"
  | "ambiguous_general";

export type SymptomSignature = {
  queryMode: QueryMode;
  domain: Domain;
  object: EvidenceObject;
  stage: Stage;
  polarity: Polarity;
  intent: Intent;
  scope: Scope;
  symptomSummary: string;
};

export type RoutingProfile = {
  archetype: IncidentArchetype;
  sourceWeights: {
    baseline_doc: number;
    process_doc: number;
    case_history: number;
  };
  guideKindWeights: Record<GuideKind, number>;
  preferCaseActions: boolean;
  preferCaseCauses: boolean;
  demoteConfigurationOnlyGuides: boolean;
  requireGuideEvidence: boolean;
  allowCaseOnlyAnswer: boolean;
  signalPhrases: string[];
};

const TERM_STOP_WORDS = new Set([
  "가이드",
  "기준",
  "문서",
  "설정",
  "확인",
  "필요",
  "안내",
  "절차",
  "프로세스",
  "요청",
  "방법",
  "증상",
  "문제",
  "경우",
  "관련",
  "기본",
  "동일",
  "이슈",
  "최신",
  "대응",
  "문의",
  "내용",
  "patient",
  "guide",
  "manual",
  "setup",
  "issue",
  "problem",
]);

const ACTION_FAMILY_PATTERNS = {
  cancel: [/취소/i, /cancel/i],
  startup: [/실행/i, /시작/i, /구동/i, /켜/i, /launch/i, /start/i, /boot/i],
  output: [/출력/i, /인쇄/i, /프린터/i, /print/i],
  recognition: [/인식/i, /스캔/i, /리더/i, /barcode/i, /scan/i, /read/i],
};

const MODIFIER_PATTERNS = {
  auto: [/자동/i, /저절로/i, /스스로/i, /by itself/i, /automatically/i],
  negative: [/안 됨/i, /안됨/i, /되지 않/i, /불가/i, /실패/i, /오류/i, /fail/i, /cannot/i],
  repeated: [/계속/i, /반복/i, /중복/i, /두 번/i, /여러 번/i, /repeat/i, /twice/i],
  immediateClose: [/바로 꺼/i, /즉시 종료/i, /곧 꺼/i, /immediately closes/i, /closes right away/i],
  guide: [/어떻게/i, /방법/i, /어디/i, /메뉴/i, /경로/i, /설정/i, /기준/i, /값/i],
  process: [/절차/i, /프로세스/i, /처리/i, /요청/i, /안내/i, /승인번호/i, /이관/i],
  recovery: [/복구/i, /복원/i, /재시작/i, /recover/i, /restore/i],
  intermittent: [/가끔/i, /간헐/i, /때때로/i, /occasionally/i, /intermittent/i],
  persistent: [/계속/i, /항상/i, /매번/i, /지속/i, /always/i, /persistent/i],
  allCases: [/전체/i, /모든/i, /전부/i, /모두/i, /all/i],
};

const EVENT_CONTEXT_PATTERNS = {
  completed: [/정상/i, /완료/i, /성공/i, /승인/i, /결제됨/i, /approved/i, /completed/i, /successful/i],
  observedLater: [/직후/i, /이후/i, /나중/i, /later/i, /afterward/i],
  reflectionMissing: [/미반영/i, /누락/i, /사라지/i, /반영 안/i, /not reflected/i, /missing/i],
  attempt: [/시도/i, /진행/i, /하려/i, /trying/i, /attempt/i],
};

const PHYSICAL_SIGNAL_PATTERNS = [
  /번져/i,
  /번짐/i,
  /흐리/i,
  /얼룩/i,
  /소음/i,
  /smeared?/i,
  /blur(red)?/i,
  /streak/i,
  /dirty/i,
  /noise/i,
  /이물/i,
  /찌꺼기/i,
  /debris/i,
  /걸림/i,
  /빨간불/i,
  /깜빡/i,
  /전원/i,
  /스위치/i,
  /패널/i,
  /토너/i,
  /잉크/i,
  /fuser/i,
  /롤러/i,
  /정착기/i,
  /부품/i,
  /케이블/i,
  /교체/i,
  /replacement/i,
  /불량/i,
  /고장/i,
  /찌그러/i,
  /소모품/i,
  /hardware/i,
  /physical/i,
];

const CONFIGURATION_SIGNAL_PATTERNS = [
  /설정/i,
  /기준/i,
  /기본값/i,
  /환경설정/i,
  /메뉴/i,
  /경로/i,
  /값/i,
  /on\/off/i,
  /고정\s*ip/i,
  /ip/i,
  /catid/i,
  /van/i,
  /옵션/i,
  /포트/i,
  /드라이버/i,
  /지정프린터/i,
  /프린터 유형/i,
  /proxy/i,
  /subnet/i,
  /설치/i,
  /baseline/i,
];

const PROCEDURE_SIGNAL_PATTERNS = [
  /프로세스/i,
  /절차/i,
  /처리/i,
  /요청/i,
  /이관/i,
  /전달/i,
  /승인번호/i,
  /취소 요청/i,
  /운영/i,
  /v[ -]?an/i,
];

const META_GUIDE_PATTERNS = [
  /문서.+방법/i,
  /답변/i,
  /응답/i,
  /템플릿/i,
  /예시/i,
  /운영 지침/i,
  /검색.+로직/i,
  /rag/i,
];

const INSTALL_GUIDE_PATTERNS = [
  /설치/i,
  /install/i,
  /driver/i,
  /드라이버/i,
  /포트/i,
  /hostname/i,
  /업데이트/i,
  /신규 프린터/i,
  /제어판/i,
  /\.exe\b/i,
];

const REFERENCE_GUIDE_PATTERNS = [
  /규격/i,
  /사양/i,
  /크기/i,
  /모델/i,
  /id\b/i,
  /pw\b/i,
  /https?:\/\//i,
  /\|\s*.+\s*\|/,
  /구성 요소/i,
  /문서 목적/i,
];

export function normalizeSpace(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

export function normalizeLine(text: string) {
  return normalizeSpace(text.replace(/^[-*]\s+/, "").replace(/^\d+[\.\)]\s+/, ""));
}

export function toComparableText(text: string) {
  return normalizeSpace(text.toLowerCase());
}

export function uniqueStrings(items: string[], maxItems = items.length) {
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

export function cleanTextArtifacts(text: string) {
  return normalizeSpace(
    text
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;|&#160;/gi, " ")
      .replace(/mso-[^:]+:[^;]+;?/gi, " ")
      .replace(/font-family:[^;]+;?/gi, " ")
      .replace(/font-size:[^;]+;?/gi, " ")
      .replace(/line-height:[^;]+;?/gi, " ")
      .replace(/color:\s*#[0-9a-f]{3,6};?/gi, " ")
      .replace(/\[PHONE_REDACTED\]|\[EMAIL_REDACTED\]/g, " ")
      .replace(/style\s*=\s*"[^"]*"/gi, " ")
      .replace(/style\s*=\s*'[^']*'/gi, " "),
  );
}

export function truncateText(text: string, maxChars = 220) {
  const normalized = normalizeSpace(text);

  if (normalized.length <= maxChars) {
    return normalized;
  }

  return `${normalized.slice(0, Math.max(maxChars - 1, 0)).trimEnd()}…`;
}

function hasAny(text: string, patterns: RegExp[]) {
  return patterns.some((pattern) => pattern.test(text));
}

function countMatches(text: string, patterns: RegExp[]) {
  return patterns.reduce((count, pattern) => count + (pattern.test(text) ? 1 : 0), 0);
}

export function extractTerms(text: string) {
  return toComparableText(cleanTextArtifacts(text))
    .split(/[^0-9a-zA-Z가-힣/._-]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 1 && !TERM_STOP_WORDS.has(token));
}

export function extractSignalPhrases(text: string, maxItems = 6) {
  const normalized = cleanTextArtifacts(text);
  const tokens = normalized
    .split(/\s+/)
    .map((token) => token.replace(/[^\p{L}\p{N}/._-]+/gu, ""))
    .filter((token) => token.length > 1);
  const phrases: string[] = [];

  for (let size = 4; size >= 2; size -= 1) {
    for (let index = 0; index <= tokens.length - size; index += 1) {
      const chunk = tokens.slice(index, index + size);
      const lowered = chunk.map((token) => token.toLowerCase());

      if (lowered.every((token) => TERM_STOP_WORDS.has(token))) {
        continue;
      }

      const phrase = normalizeSpace(chunk.join(" "));

      if (phrase.length < 5) {
        continue;
      }

      phrases.push(phrase);
    }
  }

  return uniqueStrings(phrases, maxItems);
}

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

function inferQueryMode(text: string): QueryMode {
  const comparable = toComparableText(text);
  const incidentSignal =
    hasAny(comparable, [
      MODIFIER_PATTERNS.negative[0],
      MODIFIER_PATTERNS.negative[1],
      /오류/i,
      /실패/i,
      /멈춤/i,
      ...MODIFIER_PATTERNS.repeated,
      ...MODIFIER_PATTERNS.immediateClose,
    ]) || hasAny(comparable, [...ACTION_FAMILY_PATTERNS.cancel, ...ACTION_FAMILY_PATTERNS.output, ...ACTION_FAMILY_PATTERNS.recognition]);
  const guideSignal = hasAny(comparable, MODIFIER_PATTERNS.guide);

  if (guideSignal && !incidentSignal) {
    return "guide";
  }

  return "incident";
}

function inferDomain(text: string): Domain {
  const domainPatterns: Array<{ domain: Domain; patterns: RegExp[] }> = [
    { domain: "printer", patterns: [/처방전/i, /영수증/i, /프린터/i, /출력/i, /인쇄/i, /printer/i, /print/i] },
    { domain: "scanner", patterns: [/스캐너/i, /바코드/i, /qr/i, /scanner/i, /barcode/i, /scan/i] },
    { domain: "launcher", patterns: [/프로그램/i, /플레이어/i, /런처/i, /실행/i, /시작/i, /종료/i, /launcher/i, /player/i, /app/i] },
    { domain: "payment", patterns: [/결제/i, /카드/i, /승인/i, /정산/i, /수납/i, /payment/i, /card/i] },
    { domain: "login", patterns: [/로그인/i, /비밀번호/i, /계정/i, /아이디/i, /login/i, /password/i, /account/i] },
    { domain: "network", patterns: [/네트워크/i, /통신/i, /인터넷/i, /lan/i, /ip/i, /network/i, /proxy/i] },
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
    if (/(리더기|단말기|ic|마그네틱|카드 인식|reader|terminal)/i.test(text)) {
      return "payment_reader";
    }

    return "card_payment";
  }

  if (/(영수증 롤프린터|카드 영수증)/i.test(text)) {
    return "receipt_printer";
  }

  if (/(처방전|제증명|문서 프린터)/i.test(text)) {
    return "document_printer";
  }

  if (domain === "scanner" || /(바코드 리더기|스캐너)/i.test(text)) {
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

function inferIntent(text: string, queryMode: QueryMode, stage: Stage, polarity: Polarity): Intent {
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

export function normalizeSymptomSignatureFallback(message: string): SymptomSignature {
  const text = cleanTextArtifacts(message);
  const queryMode = inferQueryMode(text);
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
    symptomSummary: truncateText(text, 120),
  };
}

export function classifyGuideKind(title: string, content: string): GuideKind {
  const haystack = cleanTextArtifacts(`${title}\n${content}`);

  if (META_GUIDE_PATTERNS.some((pattern) => pattern.test(haystack))) {
    return "meta_doc";
  }

  if (PROCEDURE_SIGNAL_PATTERNS.some((pattern) => pattern.test(haystack))) {
    return "process_doc";
  }

  if (INSTALL_GUIDE_PATTERNS.some((pattern) => pattern.test(haystack))) {
    return "install_doc";
  }

  if (REFERENCE_GUIDE_PATTERNS.some((pattern) => pattern.test(haystack))) {
    return "reference_doc";
  }

  return "baseline_doc";
}

export function buildRoutingProfile(message: string, signature: SymptomSignature): RoutingProfile {
  const text = cleanTextArtifacts(message);
  const physicalScore =
    countMatches(text, PHYSICAL_SIGNAL_PATTERNS) +
    (/(교체|재설치|재부팅|청소).*(동일|안 됨|안됨|불가|실패)/i.test(text) ? 2 : 0);
  const configurationScore =
    countMatches(text, CONFIGURATION_SIGNAL_PATTERNS) +
    (signature.queryMode === "guide" ? 2 : 0) +
    (signature.intent === "baseline_check" ? 2 : 0);
  const procedureScore =
    countMatches(text, PROCEDURE_SIGNAL_PATTERNS) +
    (signature.intent === "official_process" ? 3 : 0);
  const phraseCount = extractSignalPhrases(text, 6).length;
  const specificBehaviorScore = [
    "auto_cancels",
    "cannot_cancel",
    "outputs_twice",
    "scans_twice",
    "closes_immediately",
    "fails_to_start",
    "recognition_fail",
  ].includes(signature.polarity)
    ? 2
    : 0;
  const casePatternScore =
    (signature.queryMode === "incident" ? 1 : 0) +
    specificBehaviorScore +
    (signature.scope === "all_cases" || signature.scope === "persistent" ? 1 : 0) +
    (signature.stage === "after_payment_or_after_approval" || signature.stage === "cancel_flow" ? 1 : 0) +
    (phraseCount > 1 ? 1 : 0);

  let archetype: IncidentArchetype = "ambiguous_general";

  if (procedureScore >= Math.max(configurationScore, physicalScore, 2)) {
    archetype = "process_execution";
  } else if (physicalScore >= Math.max(configurationScore + 1, 2)) {
    archetype = "physical_device";
  } else if (configurationScore >= Math.max(physicalScore + 1, 2)) {
    archetype = "configuration_sensitive";
  } else if (casePatternScore >= 4) {
    archetype = "known_case_pattern";
  }

  switch (archetype) {
    case "process_execution":
      return {
        archetype,
        sourceWeights: { baseline_doc: 10, process_doc: 30, case_history: -18 },
        guideKindWeights: {
          baseline_doc: 4,
          process_doc: 18,
          install_doc: -4,
          reference_doc: -6,
          meta_doc: -30,
        },
        preferCaseActions: false,
        preferCaseCauses: false,
        demoteConfigurationOnlyGuides: false,
        requireGuideEvidence: true,
        allowCaseOnlyAnswer: false,
        signalPhrases: extractSignalPhrases(text, 6),
      };
    case "configuration_sensitive":
      return {
        archetype,
        sourceWeights: { baseline_doc: 22, process_doc: -4, case_history: 8 },
        guideKindWeights: {
          baseline_doc: 10,
          process_doc: -4,
          install_doc: 4,
          reference_doc: -6,
          meta_doc: -28,
        },
        preferCaseActions: false,
        preferCaseCauses: false,
        demoteConfigurationOnlyGuides: false,
        requireGuideEvidence: signature.queryMode === "guide",
        allowCaseOnlyAnswer: signature.queryMode !== "guide",
        signalPhrases: extractSignalPhrases(text, 6),
      };
    case "physical_device":
      return {
        archetype,
        sourceWeights: { baseline_doc: 4, process_doc: -12, case_history: 24 },
        guideKindWeights: {
          baseline_doc: -2,
          process_doc: -10,
          install_doc: -14,
          reference_doc: -18,
          meta_doc: -36,
        },
        preferCaseActions: true,
        preferCaseCauses: true,
        demoteConfigurationOnlyGuides: true,
        requireGuideEvidence: false,
        allowCaseOnlyAnswer: true,
        signalPhrases: extractSignalPhrases(text, 6),
      };
    case "known_case_pattern":
      return {
        archetype,
        sourceWeights: { baseline_doc: 8, process_doc: -8, case_history: 20 },
        guideKindWeights: {
          baseline_doc: 2,
          process_doc: -8,
          install_doc: -10,
          reference_doc: -10,
          meta_doc: -32,
        },
        preferCaseActions: true,
        preferCaseCauses: true,
        demoteConfigurationOnlyGuides: true,
        requireGuideEvidence: false,
        allowCaseOnlyAnswer: true,
        signalPhrases: extractSignalPhrases(text, 6),
      };
    default:
      return {
        archetype: "ambiguous_general",
        sourceWeights: { baseline_doc: 10, process_doc: -2, case_history: 10 },
        guideKindWeights: {
          baseline_doc: 4,
          process_doc: -2,
          install_doc: -6,
          reference_doc: -10,
          meta_doc: -30,
        },
        preferCaseActions: false,
        preferCaseCauses: false,
        demoteConfigurationOnlyGuides: false,
        requireGuideEvidence: signature.queryMode === "guide",
        allowCaseOnlyAnswer: true,
        signalPhrases: extractSignalPhrases(text, 6),
      };
  }
}
