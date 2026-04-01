import test from "node:test";
import assert from "node:assert/strict";

import { Document } from "@langchain/core/documents";

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
    polarity?: string;
    intent: string;
    scope?: string;
  },
) {
  return new Document({
    pageContent: `Document Type: kiosk guide\nSection: ${sectionTitle}\n\n${content}`,
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
    pageContent: `Problem Summary: ${problemSummary}\nRoot Cause: ${rootCause}\nResolution Action: ${resolutionAction}`,
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

test("network incident separates baseline checks from similar-case extras and removes repeated source labels", () => {
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

  assert.ok(result.response.suspected_causes.every((item) => !item.startsWith("[")));
  assert.ok(result.response.checks.every((item) => !item.startsWith("[")));
  assert.equal(result.response.section_groups.checks[0]?.title, "설치가이드 기준");
  assert.equal(result.response.section_groups.checks[0]?.list_style, "bullet");
  assert.equal(result.response.section_groups.checks[1]?.title, "추가로 볼 사항(유사 사례)");
  assert.ok(result.response.section_groups.checks[0]?.items.some((item) => /IPv4|Ethernet|proxy|subnet/i.test(item)));
});

test("official process query promotes process guidance first and uses ordered groups only for procedure-shaped steps", () => {
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

  assert.equal(result.response.section_groups.checks[0]?.title, "운영 절차 기준");
  assert.equal(result.response.section_groups.checks[0]?.list_style, "ordered");
  assert.equal(result.response.section_groups.actions[0]?.list_style, "ordered");
  assert.ok(result.response.checks.every((item) => !item.startsWith("[")));
});

test("printer output incident returns guide excerpt as markdown-style baseline reference", () => {
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

  assert.equal(result.response.section_groups.checks[0]?.title, "설치가이드 기준");
  assert.ok(result.response.baseline_reference?.markdown_excerpt?.includes("`kiosk_baseline_guide.md`"));
  assert.ok(result.response.baseline_reference?.markdown_excerpt?.includes("9.4 document printer setting"));
  assert.match(
    result.response.baseline_reference?.markdown_excerpt ?? "",
    /patient submission documents|kiosk HW setting > printer setting/i,
  );
});

test("weak retrieval keeps fallback grouped as additional confirmation without generic IT guesses", () => {
  const result = runPipeline("something feels wrong", []);

  assert.equal(result.response.fallback_used, true);
  assert.equal(result.response.section_groups.checks[0]?.title, "추가 확인");
  assert.equal(result.response.section_groups.checks[0]?.list_style, "bullet");
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
