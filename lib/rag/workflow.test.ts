import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

import { Document } from "@langchain/core/documents";
import { parse } from "csv-parse/sync";

import { __testing } from "@/lib/rag/workflow";

function createGuideDocument(
  sectionTitle: string,
  content: string,
  sourceType: "baseline_guide" | "official_guide" = "baseline_guide",
  semantic?: {
    guideKind?: "baseline_doc" | "process_doc";
    domain: string;
    object: string;
    stage: string;
    polarity: string;
    intent: string;
    scope?: string;
  },
) {
  return new Document({
    pageContent: `문서 유형: 키오스크 기준 가이드\n섹션: ${sectionTitle}\n\n${content}`,
    metadata: {
      source_type: sourceType,
      guide_section_title: sectionTitle,
      guide_chunk_index: 1,
      source_file: sourceType === "baseline_guide" ? "kiosk_baseline_guide.md" : "kiosk_operation_manual.md",
      guide_kind: semantic?.guideKind,
      semantic_domain: semantic?.domain,
      semantic_object: semantic?.object,
      semantic_stage: semantic?.stage,
      semantic_polarity: semantic?.polarity ?? "unknown",
      semantic_intent: semantic?.intent ?? "diagnosis",
      semantic_scope: semantic?.scope ?? "single_case",
      semantic_query_mode: "incident",
      semantic_summary: sectionTitle,
    },
  });
}

function createCaseDocument(
  problemSummary: string,
  rootCause: string,
  resolutionAction: string,
  semantic?: {
    domain: string;
    object: string;
    stage: string;
    polarity: string;
    intent?: string;
    scope?: string;
  },
) {
  return new Document({
    pageContent: `증상 요약: ${problemSummary}\n실제 원인: ${rootCause}\n실제 해결 조치: ${resolutionAction}`,
    metadata: {
      source_type: "support_case",
      case_key: `CASE-${problemSummary}`,
      clinic_name: "테스트의원",
      issue_subtype_label: "테스트",
      problem_summary: problemSummary,
      root_cause: rootCause,
      resolution_action: resolutionAction,
      resolution_result: "resolved",
      quality_tier: "ground_truth",
      source_file: "cases.csv",
      semantic_domain: semantic?.domain,
      semantic_object: semantic?.object,
      semantic_stage: semantic?.stage,
      semantic_polarity: semantic?.polarity,
      semantic_intent: semantic?.intent ?? "diagnosis",
      semantic_scope: semantic?.scope ?? "single_case",
      semantic_query_mode: "incident",
      semantic_summary: problemSummary,
    },
  });
}

type ActualCaseRow = {
  clinic_case_key: string;
  clinic_name: string;
  issue_subtype_label_auto: string;
  auto_issue_detail: string;
  auto_latest_action: string;
  gt_problem_summary: string;
  gt_root_cause: string;
  gt_resolution_action: string;
  gt_resolution_result: string;
};

const baselineGuideRaw = fs.readFileSync(new URL("../../data/input/kiosk_baseline_guide.md", import.meta.url), "utf8");
const actualCaseRows = parse(
  fs.readFileSync(new URL("../../data/input/kiosk_gt_annotation_master.csv", import.meta.url), "utf8"),
  {
    columns: true,
    skip_empty_lines: true,
  },
) as ActualCaseRow[];

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getGuideSection(heading: string) {
  const normalized = baselineGuideRaw.replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  const startIndex = lines.findIndex((line) => new RegExp(`^#{2,3}\\s+${escapeRegExp(heading)}$`).test(line.trim()));

  assert.notEqual(startIndex, -1, `guide section not found: ${heading}`);

  const startLevel = lines[startIndex]?.match(/^(#+)/)?.[1].length ?? 2;
  const collected = [lines[startIndex]];

  for (let index = startIndex + 1; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    const headingMatch = line.match(/^(#+)\s+/);

    if (headingMatch && headingMatch[1].length <= startLevel) {
      break;
    }

    collected.push(line);
  }

  return collected.join("\n").trim();
}

function createActualGuideDocument(
  heading: string,
  semantic: {
    guideKind?: "baseline_doc" | "process_doc";
    domain: string;
    object: string;
    stage: string;
    polarity?: string;
    intent: string;
    scope?: string;
  },
) {
  return new Document({
    pageContent: `문서 유형: 키오스크 기준 가이드\n섹션: ${heading}\n\n${getGuideSection(heading)}`,
    metadata: {
      source_type: "baseline_guide",
      guide_section_title: heading,
      guide_chunk_index: 1,
      source_file: "kiosk_baseline_guide.md",
      guide_kind: semantic.guideKind,
      semantic_domain: semantic.domain,
      semantic_object: semantic.object,
      semantic_stage: semantic.stage,
      semantic_polarity: semantic.polarity ?? "unknown",
      semantic_intent: semantic.intent,
      semantic_scope: semantic.scope ?? "single_case",
      semantic_query_mode: "incident",
      semantic_summary: heading,
    },
  });
}

function getActualCaseRow(caseKey: string) {
  const row = actualCaseRows.find((item) => item.clinic_case_key === caseKey);

  assert.ok(row, `case row not found: ${caseKey}`);

  return row;
}

function createActualCaseDocument(
  caseKey: string,
  semantic: {
    domain: string;
    object: string;
    stage: string;
    polarity: string;
    intent?: string;
    scope?: string;
  },
  contentMode: "gt" | "auto" = "gt",
) {
  const row = getActualCaseRow(caseKey);
  const problemSummary = contentMode === "auto" ? row.auto_issue_detail : row.gt_problem_summary;
  const rootCause = contentMode === "auto" ? row.auto_latest_action : row.gt_root_cause;
  const resolutionAction = contentMode === "auto" ? row.auto_latest_action : row.gt_resolution_action;

  return new Document({
    pageContent: `증상 요약: ${problemSummary}\n실제 원인: ${rootCause}\n실제 해결 조치: ${resolutionAction}`,
    metadata: {
      source_type: "support_case",
      case_key: row.clinic_case_key,
      clinic_name: row.clinic_name,
      issue_subtype_label: row.issue_subtype_label_auto,
      problem_summary: problemSummary,
      root_cause: rootCause,
      resolution_action: resolutionAction,
      resolution_result: row.gt_resolution_result,
      quality_tier: "ground_truth",
      source_file: "kiosk_gt_annotation_master.csv",
      semantic_domain: semantic.domain,
      semantic_object: semantic.object,
      semantic_stage: semantic.stage,
      semantic_polarity: semantic.polarity,
      semantic_intent: semantic.intent ?? "diagnosis",
      semantic_scope: semantic.scope ?? "single_case",
      semantic_query_mode: "incident",
      semantic_summary: problemSummary,
    },
  });
}

function runPipeline(
  message: string,
  inputs: Array<[Document, number, "baseline" | "process" | "case" | undefined]>,
  signatureOverride?: {
    queryMode?: "incident" | "guide";
    domain?: "payment" | "printer" | "scanner" | "launcher" | "login" | "network" | "display" | "etc";
    object?:
      | "card_payment"
      | "payment_reader"
      | "receipt_printer"
      | "document_printer"
      | "barcode_scanner"
      | "kiosk_player"
      | "login_account"
      | "network_connection"
      | "display_panel"
      | "etc";
    stage?:
      | "before_action"
      | "in_progress"
      | "after_action"
      | "after_payment_or_after_approval"
      | "cancel_flow"
      | "startup"
      | "output"
      | "recognition"
      | "login_flow"
      | "network_flow"
      | "etc";
    polarity?:
      | "fails"
      | "does_not_happen"
      | "auto_happens"
      | "auto_cancels"
      | "cannot_cancel"
      | "repeats"
      | "closes_immediately"
      | "outputs_twice"
      | "scans_twice"
      | "fails_to_start"
      | "recognition_fail"
      | "no_output"
      | "unknown";
    intent?: "diagnosis" | "baseline_check" | "official_process" | "recovery";
    scope?: "single_case" | "all_cases" | "intermittent" | "persistent";
  },
) {
  return __testing.runDeterministicIncidentPipeline(
    message,
    inputs.map(([document, similarity, channel]) => ({
      document,
      similarity,
      channel: channel ?? "baseline",
      query: message,
    })),
    undefined,
    signatureOverride,
  );
}

test("generic symptom normalization separates auto action from blocked action", () => {
  const autoAction = __testing.normalizeSymptomSignatureFallback("작업이 자동으로 취소됩니다");
  const blockedAction = __testing.normalizeSymptomSignatureFallback("작업 취소가 되지 않습니다");

  assert.equal(autoAction.stage, "after_payment_or_after_approval");
  assert.equal(autoAction.polarity, "auto_cancels");
  assert.equal(blockedAction.stage, "cancel_flow");
  assert.equal(blockedAction.polarity, "cannot_cancel");
});

test("auto-cancel query stays baseline-first and excludes cannot-cancel process/case mixing", () => {
  const paymentBaseline = createGuideDocument(
    "20.2 결제 이슈 1차 체크",
    [
      "- 풀: KSNET 아이콘이 초록색인가",
      "- 하프: VCAT 옵션값이 기본 기준과 일치하는가",
      "- 포트 자동검색 / 워킹키수신 / 무결성체크가 정상인가",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "payment",
      object: "card_payment",
      stage: "after_payment_or_after_approval",
      polarity: "unknown",
      intent: "diagnosis",
    },
  );
  const cancelProcess = createGuideDocument(
    "26.1 키오스크 중복결제 및 의도하지 않은 결제건 취소 요청 프로세스",
    [
      "- 승인번호를 우선 확보합니다.",
      "- 키오스크 플레이어 결제내역 조회 또는 VAN 결제내역 조회 사이트에서 승인내역을 확인합니다.",
      "- 운영기획팀 전달 후 VAN사 취소 요청을 진행합니다.",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "process_doc",
      domain: "payment",
      object: "card_payment",
      stage: "cancel_flow",
      polarity: "cannot_cancel",
      intent: "official_process",
    },
  );
  const cannotCancelCase = createCaseDocument(
    "민생지원금 결제건 취소 안됨",
    "운영 취소 절차 확인이 늦어 승인 취소 진행이 지연됨",
    "승인번호 확보 후 운영기획팀 전달",
    {
      domain: "payment",
      object: "card_payment",
      stage: "cancel_flow",
      polarity: "cannot_cancel",
    },
  );

  const result = runPipeline("수납시 카드결제가 자동으로 취소됨", [
    [paymentBaseline, 0.86, "baseline"],
    [cancelProcess, 0.91, "process"],
    [cannotCancelCase, 0.95, "case"],
  ], {
    domain: "payment",
    object: "card_payment",
    stage: "after_payment_or_after_approval",
    polarity: "auto_cancels",
    intent: "diagnosis",
    scope: "single_case",
  });

  assert.match(result.response.suspected_causes[0] ?? "", /^\[기준 문서\]/);
  assert.match(result.response.checks[0] ?? "", /^\[기준 문서\]/);
  assert.ok(result.response.suspected_causes.every((item) => !/운영 절차/.test(item)));
  assert.ok(result.response.similar_cases.every((item) => !/취소 안됨/.test(item.problem_summary)));
});

test("cannot-cancel query prioritizes official process and excludes auto-cancel diagnosis mixing", () => {
  const paymentBaseline = createGuideDocument(
    "20.2 결제 이슈 1차 체크",
    [
      "- 풀: KSNET 아이콘이 초록색인가",
      "- 하프: VCAT 옵션값이 기본 기준과 일치하는가",
      "- 포트 자동검색 / 워킹키수신 / 무결성체크가 정상인가",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "payment",
      object: "card_payment",
      stage: "after_payment_or_after_approval",
      polarity: "unknown",
      intent: "diagnosis",
    },
  );
  const cancelProcess = createGuideDocument(
    "26.1 키오스크 중복결제 및 의도하지 않은 결제건 취소 요청 프로세스",
    [
      "- 승인번호를 우선 확보합니다.",
      "- 키오스크 플레이어 결제내역 조회 또는 VAN 결제내역 조회 사이트에서 승인내역을 확인합니다.",
      "- 운영기획팀 전달 후 VAN사 취소 요청을 진행합니다.",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "process_doc",
      domain: "payment",
      object: "card_payment",
      stage: "cancel_flow",
      polarity: "cannot_cancel",
      intent: "official_process",
    },
  );
  const autoCancelCase = createCaseDocument(
    "승인 직후 자동 취소됨",
    "결제 기준 설정값 이탈",
    "결제 이슈 1차 체크 기준값 복구",
    {
      domain: "payment",
      object: "card_payment",
      stage: "after_payment_or_after_approval",
      polarity: "auto_cancels",
    },
  );

  const result = runPipeline("민생지원금 결제건 취소안됨", [
    [paymentBaseline, 0.78, "baseline"],
    [cancelProcess, 0.92, "process"],
    [autoCancelCase, 0.93, "case"],
  ], {
    domain: "payment",
    object: "card_payment",
    stage: "cancel_flow",
    polarity: "cannot_cancel",
    intent: "official_process",
    scope: "single_case",
  });

  assert.match(result.response.checks[0] ?? "", /^\[운영 절차\]/);
  assert.match(result.response.actions[0] ?? "", /^\[운영 절차\]/);
  assert.ok(result.response.suspected_causes[0]?.includes("공식 운영 절차") || result.response.suspected_causes[0]?.includes("처리 기준"));
  assert.ok(result.response.similar_cases.every((item) => !/자동 취소/.test(item.problem_summary)));
});

test("printer repeated-output query excludes no-output case history", () => {
  const printerBaseline = createGuideDocument(
    "9.5 카드 영수증 프린터 설정",
    [
      "- 실물카드 / 삼성페이 / 애플페이 영수증 출력 롤프린터 지정",
      "- 카드 영수증 프린터 기본 지정 상태를 확인합니다.",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "printer",
      object: "receipt_printer",
      stage: "output",
      polarity: "unknown",
      intent: "diagnosis",
    },
  );
  const noOutputCase = createCaseDocument(
    "영수증 출력 안됨",
    "영수증 프린터 지정 누락",
    "카드 영수증 프린터를 다시 지정",
    {
      domain: "printer",
      object: "receipt_printer",
      stage: "output",
      polarity: "no_output",
    },
  );

  const result = runPipeline("영수증이 계속 출력됨", [
    [printerBaseline, 0.87, "baseline"],
    [noOutputCase, 0.94, "case"],
  ], {
    domain: "printer",
    object: "receipt_printer",
    stage: "output",
    polarity: "outputs_twice",
    intent: "diagnosis",
    scope: "persistent",
  });

  assert.match(result.response.checks[0] ?? "", /^\[기준 문서\]/);
  assert.ok(result.response.similar_cases.every((item) => !/출력 안됨/.test(item.problem_summary)));
});

test("scanner double-scan query excludes recognition-fail case history", () => {
  const scannerBaseline = createGuideDocument(
    "주변기기 연동",
    [
      "- 자동설정 실행",
      "- 리더기 입력 동작을 기준값과 비교합니다.",
      "- 포트값을 확인합니다.",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "scanner",
      object: "barcode_scanner",
      stage: "recognition",
      polarity: "unknown",
      intent: "diagnosis",
    },
  );
  const recognitionFailCase = createCaseDocument(
    "바코드 스캐너 인식 안됨",
    "포트값 누락",
    "자동설정 재실행 후 포트 복구",
    {
      domain: "scanner",
      object: "barcode_scanner",
      stage: "recognition",
      polarity: "recognition_fail",
    },
  );

  const result = runPipeline("스캐너가 두 번 찍힘", [
    [scannerBaseline, 0.82, "baseline"],
    [recognitionFailCase, 0.96, "case"],
  ], {
    domain: "scanner",
    object: "barcode_scanner",
    stage: "recognition",
    polarity: "scans_twice",
    intent: "diagnosis",
    scope: "persistent",
  });

  assert.match(result.response.suspected_causes[0] ?? "", /^\[기준 문서\]/);
  assert.ok(result.response.similar_cases.every((item) => !/인식 안됨/.test(item.problem_summary)));
});

test("launcher closes-immediately query excludes fails-to-start case history", () => {
  const launcherBaseline = createGuideDocument(
    "기본 운영 점검",
    [
      "- 서비스 사용 ON 상태를 확인합니다.",
      "- 메인화면 연결 상태를 확인합니다.",
      "- 테스트 설정이 전체화면 / 운영으로 복구되어 있는지 확인합니다.",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "launcher",
      object: "kiosk_player",
      stage: "startup",
      polarity: "unknown",
      intent: "diagnosis",
    },
  );
  const failsToStartCase = createCaseDocument(
    "프로그램이 안 켜짐",
    "서비스 사용 OFF",
    "서비스 사용 ON으로 복구",
    {
      domain: "launcher",
      object: "kiosk_player",
      stage: "startup",
      polarity: "fails_to_start",
    },
  );

  const result = runPipeline("프로그램이 켜졌다가 바로 꺼짐", [
    [launcherBaseline, 0.84, "baseline"],
    [failsToStartCase, 0.95, "case"],
  ], {
    domain: "launcher",
    object: "kiosk_player",
    stage: "startup",
    polarity: "closes_immediately",
    intent: "diagnosis",
    scope: "single_case",
  });

  assert.match(result.response.checks[0] ?? "", /^\[기준 문서\]/);
  assert.ok(result.response.similar_cases.every((item) => !/안 켜짐/.test(item.problem_summary)));
});

test("payment settlement-loss incident surfaces fixed IP check from aligned evidence instead of cancel process", () => {
  const paymentBaseline = createActualGuideDocument("20.2 결제 이슈 1차 체크", {
    guideKind: "baseline_doc",
    domain: "payment",
    object: "card_payment",
    stage: "after_payment_or_after_approval",
    intent: "diagnosis",
  });
  const networkBaseline = createActualGuideDocument("12.2 PC 네트워크 설정", {
    guideKind: "baseline_doc",
    domain: "payment",
    object: "card_payment",
    stage: "after_payment_or_after_approval",
    polarity: "auto_cancels",
    intent: "diagnosis",
  });
  const cancelProcess = createActualGuideDocument("26.1 키오스크 중복결제 및 의도하지 않은 결제건 취소 요청 프로세스", {
    guideKind: "process_doc",
    domain: "payment",
    object: "card_payment",
    stage: "cancel_flow",
    polarity: "cannot_cancel",
    intent: "official_process",
  });
  const fixedIpCase = createActualCaseDocument("130793-001", {
    domain: "payment",
    object: "card_payment",
    stage: "after_payment_or_after_approval",
    polarity: "auto_cancels",
    scope: "all_cases",
  });

  const result = runPipeline(
    "키오스크 수납시 카드결제가 잘 되는데 조금 뒤에 살펴보면 결제된 것이 전부 취소처리 되어 있음. 수납도 안 된 것으로 확인이 됨",
    [
      [paymentBaseline, 0.87, "baseline"],
      [networkBaseline, 0.93, "baseline"],
      [cancelProcess, 0.91, "process"],
      [fixedIpCase, 0.97, "case"],
    ],
  );

  assert.equal(result.signature.domain, "payment");
  assert.equal(result.signature.stage, "after_payment_or_after_approval");
  assert.equal(result.signature.polarity, "auto_cancels");
  assert.ok(
    [...result.response.checks, ...result.response.actions, ...result.response.suspected_causes].some((item) =>
      /고정\s*IP|수동\s*IP/i.test(item),
    ),
  );
  assert.ok(result.response.checks.some((item) => /고정\s*IP|수동\s*IP/i.test(item)));
  assert.ok((result.response.suspected_causes[0] ?? "").startsWith("[기준 문서]"));
  assert.ok(result.response.checks.every((item) => !/승인번호|운영기획팀|취소 요청 프로세스/i.test(item)));
});

test("post-payment anomaly excludes payment-reader failure cases through object and stage alignment", () => {
  const settlementBaseline = createGuideDocument(
    "결제 승인 후 반영 점검",
    [
      "- 승인 이후 수납 반영 여부 확인",
      "- 네트워크 및 고정 IP 기준값 확인",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "payment",
      object: "card_payment",
      stage: "after_payment_or_after_approval",
      polarity: "auto_cancels",
      intent: "diagnosis",
    },
  );
  const paymentReaderCase = createCaseDocument(
    "카드 리더기가 IC 카드를 인식하지 못함",
    "카드 단말기 또는 리더기 불량",
    "단말기 교체 후 재확인",
    {
      domain: "payment",
      object: "payment_reader",
      stage: "recognition",
      polarity: "recognition_fail",
    },
  );

  const result = runPipeline(
    "승인까지는 된 것 같은데 조금 뒤에 보면 결제가 취소로 바뀌고 수납 반영도 안 됩니다",
    [
      [settlementBaseline, 0.88, "baseline"],
      [paymentReaderCase, 0.97, "case"],
    ],
    {
      domain: "payment",
      object: "card_payment",
      stage: "after_payment_or_after_approval",
      polarity: "auto_cancels",
      intent: "diagnosis",
      scope: "single_case",
    },
  );

  assert.ok(result.response.suspected_causes.every((item) => !/리더기|단말기 교체/i.test(item)));
  assert.ok(result.response.checks.every((item) => !/리더기|단말기 교체/i.test(item)));
  assert.equal(result.response.similar_cases.length, 0);
});

test("printer output incident prioritizes test print baseline checks and links driver guide section", () => {
  const printerBaseline = createActualGuideDocument("20.3 프린터 이슈 1차 체크", {
    guideKind: "baseline_doc",
    domain: "printer",
    object: "receipt_printer",
    stage: "output",
    polarity: "no_output",
    intent: "diagnosis",
  });
  const printerMenuGuide = createActualGuideDocument("9.1 메뉴 개요", {
    guideKind: "baseline_doc",
    domain: "printer",
    object: "receipt_printer",
    stage: "output",
    polarity: "no_output",
    intent: "diagnosis",
  });
  const driverGuide = createActualGuideDocument("11.2 풀 키오스크 및 하프 키오스크(삼성프린터) 드라이버 설치", {
    guideKind: "baseline_doc",
    domain: "printer",
    object: "receipt_printer",
    stage: "output",
    polarity: "no_output",
    intent: "diagnosis",
  });
  const printerPortGuide = createActualGuideDocument("11.3 포트 및 기본 프린터 설정", {
    guideKind: "baseline_doc",
    domain: "printer",
    object: "receipt_printer",
    stage: "output",
    polarity: "no_output",
    intent: "diagnosis",
  });
  const printerCase = createActualCaseDocument("129702-001", {
    domain: "printer",
    object: "receipt_printer",
    stage: "output",
    polarity: "no_output",
  }, "auto");
  const printerTestCase = createActualCaseDocument("37901-002", {
    domain: "printer",
    object: "receipt_printer",
    stage: "output",
    polarity: "no_output",
  }, "auto");

  const result = runPipeline("키오스크 수납 후 프린터 인쇄가 안됨", [
    [printerBaseline, 0.95, "baseline"],
    [printerMenuGuide, 0.96, "baseline"],
    [driverGuide, 0.93, "baseline"],
    [printerPortGuide, 0.94, "baseline"],
    [printerCase, 0.96, "case"],
    [printerTestCase, 0.95, "case"],
  ]);
  const combined = [
    ...result.response.checks,
    ...result.response.actions,
    ...(result.response.baseline_reference?.source_titles ?? []),
    ...(result.response.baseline_reference?.source_files ?? []),
    ...(result.response.baseline_reference?.excerpts ?? []),
  ].join("\n");

  assert.equal(result.signature.domain, "printer");
  assert.equal(result.signature.stage, "output");
  assert.equal(result.signature.polarity, "no_output");
  assert.match(combined, /테스트 페이지|출력 테스트/i);
  assert.match(combined, /프린터 설정|기본 프린터/i);
  assert.match(combined, /드라이버/i);
  assert.match(combined, /기본 프린터|기본 세팅|불일치|상이|호스트 네임|포트 구성/i);
  assert.ok(Boolean(result.response.baseline_reference));
  assert.match(result.response.baseline_reference?.source_files.join(" ") ?? "", /kiosk_baseline_guide\.md/i);
  assert.match(
    `${result.response.baseline_reference?.source_titles.join("\n")}\n${result.response.baseline_reference?.excerpts.join("\n")}`,
    /드라이버|기본 프린터/i,
  );
});

test("weak retrieval falls back conservatively without generic IT guesses", () => {
  const result = runPipeline("무언가 이상함", []);

  assert.equal(result.response.fallback_used, true);
  assert.match(result.response.checks[0] ?? "", /^\[추정 보완\]/);
  assert.ok(result.response.checks.every((item) => !/장치관리자|USB|드라이버|방화벽/.test(item)));
  assert.ok(result.response.actions.every((item) => !/장치관리자|USB|드라이버|방화벽/.test(item)));
});
