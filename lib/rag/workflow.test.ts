import test from "node:test";
import assert from "node:assert/strict";

import { Document } from "@langchain/core/documents";

import { __testing } from "@/lib/rag/workflow";

function createGuideDocument(
  sectionTitle: string,
  content: string,
  sourceType: "baseline_guide" | "official_guide" = "baseline_guide",
) {
  return new Document({
    pageContent: `문서 유형: 키오스크 기준 가이드\n섹션: ${sectionTitle}\n\n${content}`,
    metadata: {
      source_type: sourceType,
      guide_section_title: sectionTitle,
      source_file: sourceType === "baseline_guide" ? "kiosk_baseline_guide.md" : "kiosk_operation_manual.md",
    },
  });
}

function createCaseDocument(problemSummary: string, rootCause: string, resolutionAction: string) {
  return new Document({
    pageContent: `증상 요약: ${problemSummary}\n실제 원인: ${rootCause}\n실제 해결 조치: ${resolutionAction}`,
    metadata: {
      source_type: "support_case",
      case_key: "CASE-1",
      clinic_name: "테스트의원",
      issue_subtype_label: "테스트",
      problem_summary: problemSummary,
      root_cause: rootCause,
      resolution_action: resolutionAction,
      resolution_result: "resolved",
      quality_tier: "ground_truth",
    },
  });
}

test("retrieval reranking puts baseline guide ahead of case history for payment incidents", () => {
  const reranked = __testing.rerankRetrievedDocuments(
    [
      [
        createCaseDocument("결제 안됨", "장치 관리자에서 리더기를 재설치해 해결", "드라이버 재설치"),
        0.95,
      ],
      [
        createGuideDocument(
          "결제 이슈 1차 체크",
          "- KSNET 에이전트 아이콘 상태 확인\n- VAN 주소 `210.181.28.137` 확인\n- CATID 확인",
        ),
        0.72,
      ],
    ],
    "결제 안됨",
    "incident",
  );

  const firstMetadata = reranked[0]?.[0].metadata as Record<string, unknown>;
  assert.equal(firstMetadata.source_type, "baseline_guide");
});

test("payment response keeps grounded baseline bullets above generic device-manager guesses", () => {
  const result = __testing.applyBaselineFirstPolicy({
    message: "결제가 안됩니다",
    suspectedCauses: [
      "장치 관리자에서 카드리더기 인식 상태를 확인해야 합니다.",
      "VAN, CATID, 포트 설정이 맞지 않을 수 있습니다.",
    ],
    checks: [
      "장치 관리자 경고 여부를 확인합니다.",
      "결제 에이전트 실행 상태와 CATID를 확인합니다.",
    ],
    nextActions: [
      "드라이버를 재설치합니다.",
      "기준값에 맞게 VAN과 포트를 다시 설정합니다.",
    ],
    guideSnippets: [
      {
        sourceType: "guide",
        provenance: "baseline_doc",
        similarity: 0.86,
        sectionTitle: "결제 이슈 1차 체크",
        sourceFile: "kiosk_baseline_guide.md",
        content: "문서 유형: 키오스크 기준 가이드\n섹션: 결제 이슈 1차 체크\n\n- KSNET 에이전트 아이콘 상태 확인\n- VAN 주소 `210.181.28.137` 확인\n- CATID 확인",
      },
    ],
    caseMatches: [],
  });

  assert.match(result.checks[0] ?? "", /^\[기준 문서\]/);
  assert.match(result.checks[0] ?? "", /기본 설정값 기준으로 보면/);
  assert.ok(result.checks.slice(0, 3).every((item) => !/장치 관리자|드라이버/.test(item)));
  assert.match(result.suspectedCauses[0] ?? "", /^\[기준 문서\]/);
  assert.ok(result.hospitalReply.length > 0);
});

test("printer issue prefers documented printer baseline before generic windows troubleshooting", () => {
  const result = __testing.applyBaselineFirstPolicy({
    message: "프린터 출력이 안됨",
    suspectedCauses: ["윈도우 프린터 문제일 수 있습니다."],
    checks: [
      "윈도우에서 기본 프린터를 확인합니다.",
      "포트가 `172.25.123.99` 인지 확인합니다.",
    ],
    nextActions: [
      "프린터 드라이버를 재설치합니다.",
      "지정 프린터와 포트를 기준값으로 원복합니다.",
    ],
    guideSnippets: [
      {
        sourceType: "guide",
        provenance: "baseline_doc",
        similarity: 0.88,
        sectionTitle: "풀 키오스크 지정 프린터",
        sourceFile: "kiosk_baseline_guide.md",
        content:
          "문서 유형: 키오스크 기준 가이드\n섹션: 풀 키오스크 지정 프린터\n\n- 지정 프린터 선택\n- 포트 `172.25.123.99` 확인\n- USB 제거 상태 확인",
      },
    ],
    caseMatches: [],
  });

  assert.match(result.checks[0] ?? "", /^\[기준 문서\]/);
  assert.ok(result.checks.slice(0, 2).some((item) => item.includes("172.25.123.99")));
  assert.ok(result.checks.slice(0, 3).every((item) => !/윈도우에서/.test(item)));
});

test("peripheral and runtime issues surface documented baseline options before OS guesses", () => {
  const result = __testing.applyBaselineFirstPolicy({
    message: "리더기 이상으로 프로그램 실행이 안됨",
    suspectedCauses: ["운영체제 문제일 수 있습니다."],
    checks: [
      "장치 관리자에서 USB 인식을 확인합니다.",
      "서비스 사용 ON, 메인화면 연결, 자동설정, 포트값을 확인합니다.",
    ],
    nextActions: [
      "OS 재부팅 후 다시 시도합니다.",
      "서비스 사용 ON과 메인화면 연결, 자동설정, 포트값을 기준값으로 복구합니다.",
    ],
    guideSnippets: [
      {
        sourceType: "guide",
        provenance: "baseline_doc",
        similarity: 0.85,
        sectionTitle: "주변기기 연동",
        sourceFile: "kiosk_baseline_guide.md",
        content:
          "문서 유형: 키오스크 기준 가이드\n섹션: 주변기기 연동\n\n- 서비스 사용 ON\n- 메인화면 연결 확인\n- 자동설정 실행\n- 포트값 확인",
      },
    ],
    caseMatches: [],
  });

  assert.match(result.checks[0] ?? "", /^\[기준 문서\]/);
  assert.ok(result.checks.slice(0, 3).every((item) => !/운영체제|OS|장치 관리자|USB 인식/.test(item)));
  assert.match(result.nextActions[0] ?? "", /^\[기준 문서\]/);
});

test("weak retrieval falls back conservatively instead of inventing generic fixes", () => {
  const result = __testing.applyBaselineFirstPolicy({
    message: "뭔가 이상함",
    suspectedCauses: ["네트워크 문제일 수 있습니다."],
    checks: ["핑 테스트를 해봅니다."],
    nextActions: ["드라이버를 재설치합니다."],
    guideSnippets: [],
    caseMatches: [],
  });

  assert.equal(result.fallbackUsed, true);
  assert.match(result.checks[0] ?? "", /^\[추가 확인\]/);
  assert.match(result.checks[0] ?? "", /직접 대응되는 메뉴나 설정 항목이 제한적/);
  assert.ok(result.nextActions.every((item) => !/드라이버 재설치/.test(item)));
});
