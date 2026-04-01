import test from "node:test";
import assert from "node:assert/strict";

import { Document } from "@langchain/core/documents";

import { buildSupportCaseDocuments } from "@/lib/ingest/case-loader";
import type { SupportCaseRow } from "@/lib/types";
import { __testing } from "@/lib/rag/workflow";

function createGuideDocument(
  sectionTitle: string,
  content: string,
  sourceType: "baseline_guide" | "official_guide" = "baseline_guide",
  semantic?: {
    guideKind?: "baseline_doc" | "process_doc" | "install_doc" | "reference_doc" | "meta_doc";
    domain: string;
    object: string;
    stage: string;
    polarity?: string;
    intent: string;
    scope?: string;
  },
) {
  return new Document({
    pageContent: [
      "Document Type: kiosk guide",
      `Guide Kind: ${semantic?.guideKind ?? "baseline_doc"}`,
      `Section: ${sectionTitle}`,
      "",
      content,
    ].join("\n"),
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
  semantic: {
    domain: string;
    object: string;
    stage: string;
    polarity: string;
    intent?: string;
    scope?: string;
  },
) {
  return new Document({
    pageContent: [
      "Document Type: support case",
      `Symptom Summary: ${problemSummary}`,
      `Root Cause: ${rootCause}`,
      `Resolution Action: ${resolutionAction}`,
    ].join("\n"),
    metadata: {
      source_type: "support_case",
      case_key: `CASE-${problemSummary}`,
      clinic_name: "Test Clinic",
      issue_subtype_label: "Test",
      problem_summary: problemSummary,
      root_cause: rootCause,
      resolution_action: resolutionAction,
      resolution_result: "resolved",
      quality_tier: "ground_truth",
      source_file: "cases.csv",
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

type SignatureOverride = Parameters<typeof __testing.runDeterministicIncidentPipeline>[3];

function runPipeline(
  message: string,
  inputs: Array<[Document, number, "baseline" | "process" | "case" | undefined]>,
  signatureOverride?: SignatureOverride,
) {
  return __testing.runDeterministicIncidentPipeline(
    message,
    inputs.map(([document, similarity, channel]) => ({ document, similarity, channel })),
    { high: 0.78, low: 0.58 },
    signatureOverride,
  );
}

test("configuration-sensitive network incident keeps guide checks ahead of similar-case extras", () => {
  const networkGuide = createGuideDocument(
    "12.2 PC network setting",
    [
      "Setting location: kiosk PC",
      "- Menu path: Ethernet > Properties > TCP/IPv4",
      "- Check IPv4 is enabled",
      "- Verify manual subnet mask value",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "network",
      object: "network_connection",
      stage: "network_flow",
      polarity: "fails",
      intent: "diagnosis",
    },
  );
  const proxyGuide = createGuideDocument(
    "Network default checks",
    [
      "- Confirm Ethernet adapter is enabled",
      "- Confirm proxy is disabled",
      "- If subnet wording is unclear, cross-check the live screen value once",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "network",
      object: "network_connection",
      stage: "network_flow",
      polarity: "fails",
      intent: "diagnosis",
    },
  );
  const proxyCase = createCaseDocument(
    "LAN cable was connected but internet still failed",
    "Proxy remained enabled and blocked outbound traffic",
    "Disabled proxy and connection recovered",
    {
      domain: "network",
      object: "network_connection",
      stage: "network_flow",
      polarity: "fails",
    },
  );

  const result = runPipeline(
    "internet is down even after LAN cable check",
    [
      [networkGuide, 0.94, "baseline"],
      [proxyGuide, 0.92, "baseline"],
      [proxyCase, 0.97, "case"],
    ],
    {
      domain: "network",
      object: "network_connection",
      stage: "network_flow",
      polarity: "fails",
      intent: "diagnosis",
      scope: "single_case",
    },
  );

  assert.ok(result.response.checks.every((item) => !item.startsWith("[")));
  assert.ok(result.response.section_groups.checks[0]?.items.some((item) => /IPv4|Ethernet|proxy|subnet/i.test(item)));
  assert.equal(result.response.similar_cases.length, 1);
});

test("official process query promotes ordered procedure guidance", () => {
  const paymentBaseline = createGuideDocument(
    "20.2 payment first checks",
    [
      "- Confirm device and network status",
      "- Confirm approval number and payment history",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "payment",
      object: "card_payment",
      stage: "cancel_flow",
      polarity: "cannot_cancel",
      intent: "diagnosis",
    },
  );
  const cancelProcess = createGuideDocument(
    "26.1 cancellation request process",
    [
      "1. Check approval number",
      "2. Compare kiosk payment log and VAN log",
      "3. Escalate to operations and request VAN cancellation",
    ].join("\n"),
    "official_guide",
    {
      guideKind: "process_doc",
      domain: "payment",
      object: "card_payment",
      stage: "cancel_flow",
      polarity: "cannot_cancel",
      intent: "official_process",
    },
  );

  const result = runPipeline(
    "cancel is not possible for a pending card payment",
    [
      [paymentBaseline, 0.88, "baseline"],
      [cancelProcess, 0.95, "process"],
    ],
    {
      domain: "payment",
      object: "card_payment",
      stage: "cancel_flow",
      polarity: "cannot_cancel",
      intent: "official_process",
      scope: "single_case",
    },
  );

  assert.equal(result.response.section_groups.checks[0]?.list_style, "ordered");
  assert.equal(result.response.section_groups.actions[0]?.list_style, "ordered");
  assert.ok(result.response.actions.some((item) => /approval|van|cancel/i.test(item)));
});

test("configuration-shaped printer incident keeps baseline reference and guide checks", () => {
  const printerSettingGuide = createGuideDocument(
    "9.4 document printer setting",
    [
      "Setting location: kiosk HW setting > printer setting",
      "- Check target printer for patient submission documents",
      "- Keep the assigned document printer aligned with kiosk HW setting",
      "- Run print test after reassignment",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "no_output",
      intent: "diagnosis",
    },
  );
  const printerCheckGuide = createGuideDocument(
    "20.3 printer first checks",
    [
      "- Confirm kiosk HW setting printer type",
      "- Confirm test page and kiosk print test both run",
      "- Confirm output target is still assigned",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "no_output",
      intent: "diagnosis",
    },
  );
  const printerCase = createCaseDocument(
    "Test page worked after paper shortage but kiosk print did not",
    "Patient document printer assignment drifted in kiosk HW setting",
    "Reassigned document printer in kiosk HW setting and output recovered",
    {
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "no_output",
    },
  );

  const result = runPipeline(
    "after paper shortage the test page works but kiosk printer test does not",
    [
      [printerSettingGuide, 0.96, "baseline"],
      [printerCheckGuide, 0.95, "baseline"],
      [printerCase, 0.97, "case"],
    ],
    {
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "no_output",
      intent: "diagnosis",
      scope: "single_case",
    },
  );

  assert.ok(result.response.section_groups.checks[0]?.items.some((item) => /printer|print test|assigned/i.test(item)));
  assert.ok(result.response.baseline_reference?.markdown_excerpt?.includes("`kiosk_baseline_guide.md`"));
  assert.ok(result.response.baseline_reference?.markdown_excerpt?.includes("9.4 document printer setting"));
});

test("weak retrieval keeps fallback grouped as additional confirmation without generic IT guesses", () => {
  const result = runPipeline("something feels wrong", []);

  assert.equal(result.response.fallback_used, true);
  assert.ok(result.response.checks.every((item) => !/device manager|USB|driver|firewall/i.test(item)));
  assert.ok(result.response.actions.every((item) => !/device manager|USB|driver|firewall/i.test(item)));
});

test("repeated-output query excludes no-output case history from similar cases", () => {
  const printerBaseline = createGuideDocument(
    "9.5 receipt printer setting",
    [
      "- Confirm receipt printer base status",
      "- Run output test",
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
    "Receipt output failed completely",
    "Receipt printer assignment was removed",
    "Reassigned receipt printer",
    {
      domain: "printer",
      object: "receipt_printer",
      stage: "output",
      polarity: "no_output",
    },
  );

  const result = runPipeline(
    "receipt keeps printing twice",
    [
      [printerBaseline, 0.87, "baseline"],
      [noOutputCase, 0.95, "case"],
    ],
    {
      domain: "printer",
      object: "receipt_printer",
      stage: "output",
      polarity: "outputs_twice",
      intent: "diagnosis",
      scope: "persistent",
    },
  );

  assert.equal(result.response.similar_cases.length, 0);
});

test("strong case hit suppresses fallback for post-payment auto-cancel incident", () => {
  const paymentBaseline = createGuideDocument(
    "20.2 payment first checks",
    [
      "- Confirm device and network status",
      "- Confirm CAT ID and fixed IP value",
      "- Confirm clinic admin fields are reflected",
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
  const strongCase = createCaseDocument(
    "All payments were auto-cancelled right after approval",
    "Transactions not fully reflected were treated as no-card cancels",
    "Checked admin fields and fixed IP reflection, then recovered",
    {
      domain: "payment",
      object: "card_payment",
      stage: "after_payment_or_after_approval",
      polarity: "auto_cancels",
      scope: "all_cases",
    },
  );

  const result = runPipeline(
    "all kiosk payments are auto-cancelled right after approval",
    [
      [paymentBaseline, 0.89, "baseline"],
      [strongCase, 0.98, "case"],
    ],
    {
      domain: "payment",
      object: "card_payment",
      stage: "after_payment_or_after_approval",
      polarity: "auto_cancels",
      intent: "diagnosis",
      scope: "all_cases",
    },
  );

  assert.equal(result.response.fallback_used, false);
  assert.equal(result.response.similar_cases.length, 1);
  assert.ok(result.response.similar_cases[0]?.problem_summary.includes("auto-cancelled"));
});

test("physical-quality incident demotes configuration-only guide chunks", () => {
  const configGuide = createGuideDocument(
    "9.4 document printer setting",
    [
      "- Menu path: kiosk HW setting > printer setting",
      "- Select assigned printer",
      "- Keep the printer type aligned",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "no_output",
      intent: "diagnosis",
    },
  );
  const printerCase = createCaseDocument(
    "Prescription and test-page output were smeared even after toner replacement",
    "Paper debris remained inside the printer and lowered print quality",
    "Cleaned debris and restored normal output",
    {
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "fails",
    },
  );

  const result = runPipeline(
    "prescription and test-page output are smeared even after toner replacement",
    [
      [configGuide, 0.97, "baseline"],
      [printerCase, 0.98, "case"],
    ],
    {
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "fails",
      intent: "diagnosis",
      scope: "single_case",
    },
  );

  const combined = [...result.response.checks, ...result.response.actions].join(" ");

  assert.equal(result.response.fallback_used, false);
  assert.equal(result.response.similar_cases.length, 1);
  assert.ok(!/menu path|printer setting|assigned printer/i.test(combined));
  assert.ok(/debris|cleaned|print quality/i.test(combined));
});

test("meta guide chunks do not leak into grounded checks or references", () => {
  const metaGuide = createGuideDocument(
    "22. response logic",
    [
      "- Split printer and payment by domain",
      "- Prefer baseline before cases",
      "- Use answer templates",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "meta_doc",
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "fails",
      intent: "diagnosis",
    },
  );
  const printerGuide = createGuideDocument(
    "20.3 printer first checks",
    [
      "- Run printer self-test",
      "- Check whether output is physically smeared or torn",
      "- If quality stays poor after consumable replacement, inspect the printer body",
    ].join("\n"),
    "baseline_guide",
    {
      guideKind: "baseline_doc",
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "fails",
      intent: "diagnosis",
    },
  );
  const printerCase = createCaseDocument(
    "Printer output stayed smeared after toner replacement",
    "Fuser-side contamination remained inside the printer",
    "Replaced the faulty printer unit",
    {
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "fails",
    },
  );

  const result = runPipeline(
    "printer output remains smeared after replacement",
    [
      [metaGuide, 0.99, "baseline"],
      [printerGuide, 0.91, "baseline"],
      [printerCase, 0.95, "case"],
    ],
    {
      domain: "printer",
      object: "document_printer",
      stage: "output",
      polarity: "fails",
      intent: "diagnosis",
      scope: "single_case",
    },
  );

  const combined = [...result.response.checks, ...result.response.actions, result.response.baseline_reference?.markdown_excerpt ?? ""].join(" ");

  assert.ok(!/answer templates|split printer and payment|response logic/i.test(combined));
  assert.ok(/self-test|smeared|printer body/i.test(combined));
});

test("case ingest serialization front-loads symptom text and semantic metadata", () => {
  const row: SupportCaseRow = {
    sourceFile: "cases.csv",
    rowNumber: 2,
    caseId: "row-2",
    caseKey: "CASE-2",
    clinicName: "Test Clinic",
    issueCategory: "hardware",
    issueSubtype: "printer_output",
    issueSubtypeLabel: "프린터/출력",
    issueKeyRaw: "printer output quality issue",
    autoIssueSummary: "[프린터/출력] 처방전 출력물이 번져 보인다고 합니다 최신 대응: 토너 교체 후에도 동일",
    autoIssueDetail:
      "<p style=\"font-size:12pt\">문의 내용 및 요청사항 - 이슈: 키오스크 처방전 출력 시 잉크가 번져서 출력됩니다. 토너 교체 후에도 동일합니다.</p>",
    autoLatestAction: "토너 교체 후에도 동일하여 추가 점검 필요",
    gtProblemSummary: "처방전 출력물이 번져서 출력됨",
    gtRootCause: "프린터 내부 오염 또는 정착기 이상",
    gtResolutionAction: "내부 청소 및 부품 점검",
    gtResolutionResult: "resolved",
    gtCustomerReply:
      "안녕하세요. 문의 내용 및 요청사항 - 이슈: 출력물이 번져 보입니다. 삼성 점검 전 확인 부탁드립니다.",
    qualityTier: "ground_truth",
  };

  const [document] = buildSupportCaseDocuments([row]);

  assert.ok(document.pageContent.startsWith("Document Type: support case\nSymptom Summary:"));
  assert.ok(document.pageContent.includes("Symptom Keywords:"));
  assert.ok(document.pageContent.includes("User Observation:"));
  assert.ok(!String(document.metadata.customer_reply_reference).includes("<p"));
  assert.equal(document.metadata.semantic_domain, "printer");
  assert.equal(document.metadata.semantic_object, "document_printer");
});
