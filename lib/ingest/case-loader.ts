import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

import { Document } from "@langchain/core/documents";
import { parse } from "csv-parse/sync";

import { getAppConfig } from "@/lib/config";
import type { SupportCaseRow } from "@/lib/types";

type RawCsvRow = Record<string, string | undefined>;

function clean(value: string | undefined) {
  return (value ?? "").trim();
}

function buildProblemSummary(row: SupportCaseRow) {
  return row.gtProblemSummary || row.autoIssueSummary || row.autoIssueDetail || row.issueKeyRaw;
}

function buildSearchableText(row: SupportCaseRow) {
  return [
    `증상 요약: ${buildProblemSummary(row)}`,
    `자동 요약: ${row.autoIssueSummary || "기록 없음"}`,
    `상세 증상: ${row.autoIssueDetail || "기록 없음"}`,
    `실제 원인: ${row.gtRootCause || "기록 없음"}`,
    `실제 해결 조치: ${row.gtResolutionAction || "기록 없음"}`,
    `처리 결과: ${row.gtResolutionResult || "기록 없음"}`,
    `병원 답변 참고문안: ${row.gtCustomerReply || "기록 없음"}`,
    `최신 조치 메모: ${row.autoLatestAction || "기록 없음"}`,
    `이슈 카테고리: ${row.issueCategory || "기록 없음"}`,
    `이슈 세부유형: ${row.issueSubtypeLabel || row.issueSubtype || "기록 없음"}`,
  ].join("\n");
}

function toSupportCaseRow(raw: RawCsvRow, rowNumber: number, sourceFile: string): SupportCaseRow {
  const gtProblemSummary = clean(raw.gt_problem_summary);
  const gtRootCause = clean(raw.gt_root_cause);
  const gtResolutionAction = clean(raw.gt_resolution_action);

  return {
    sourceFile,
    rowNumber,
    caseId: clean(raw.case_id) || `row-${rowNumber}`,
    caseKey: clean(raw.clinic_case_key) || clean(raw.case_id) || `${sourceFile}-${rowNumber}`,
    clinicName: clean(raw.clinic_name),
    issueCategory: clean(raw.issue_category),
    issueSubtype: clean(raw.issue_subtype_auto),
    issueSubtypeLabel: clean(raw.issue_subtype_label_auto),
    issueKeyRaw: clean(raw.issue_key_raw),
    autoIssueSummary: clean(raw.auto_issue_summary),
    autoIssueDetail: clean(raw.auto_issue_detail),
    autoLatestAction: clean(raw.auto_latest_action),
    gtProblemSummary,
    gtRootCause,
    gtResolutionAction,
    gtResolutionResult: clean(raw.gt_resolution_result) as SupportCaseRow["gtResolutionResult"],
    gtCustomerReply: clean(raw.gt_customer_reply),
    qualityTier:
      gtProblemSummary || gtRootCause || gtResolutionAction ? "ground_truth" : "auto_only",
  };
}

export async function findCsvFiles() {
  const config = getAppConfig();
  const candidates = [path.resolve(process.cwd(), config.CSV_SOURCE_DIR), process.cwd()];
  const seen = new Set<string>();
  const files: string[] = [];

  for (const directory of candidates) {
    if (seen.has(directory)) {
      continue;
    }

    seen.add(directory);

    try {
      const entries = await readdir(directory, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isFile() && entry.name.toLowerCase().endsWith(".csv")) {
          files.push(path.join(directory, entry.name));
        }
      }
    } catch {
      continue;
    }
  }

  return [...new Set(files)].sort((left, right) => left.localeCompare(right));
}

export async function loadSupportCaseRows() {
  const files = await findCsvFiles();

  if (files.length === 0) {
    throw new Error("CSV 파일을 찾지 못했습니다. data/input 폴더 또는 프로젝트 루트에 CSV를 넣어주세요.");
  }

  const rows: SupportCaseRow[] = [];

  for (const file of files) {
    const rawText = await readFile(file, "utf-8");
    const parsed = parse(rawText, {
      columns: true,
      skip_empty_lines: true,
    }) as RawCsvRow[];

    parsed.forEach((raw, index) => {
      const row = toSupportCaseRow(raw, index + 2, path.basename(file));
      const hasMeaningfulContent =
        row.gtProblemSummary ||
        row.autoIssueSummary ||
        row.autoIssueDetail ||
        row.gtRootCause ||
        row.gtResolutionAction;

      if (hasMeaningfulContent) {
        rows.push(row);
      }
    });
  }

  return rows;
}

export function buildSupportCaseDocuments(rows: SupportCaseRow[]) {
  return rows.map(
    (row) =>
      new Document({
        pageContent: buildSearchableText(row),
        metadata: {
          case_id: row.caseId,
          case_key: row.caseKey,
          clinic_name: row.clinicName,
          issue_category: row.issueCategory,
          issue_subtype: row.issueSubtype,
          issue_subtype_label: row.issueSubtypeLabel,
          problem_summary: buildProblemSummary(row),
          root_cause: row.gtRootCause,
          resolution_action: row.gtResolutionAction,
          resolution_result: row.gtResolutionResult,
          customer_reply_reference: row.gtCustomerReply,
          latest_action_reference: row.autoLatestAction,
          quality_tier: row.qualityTier,
          source_file: row.sourceFile,
          row_number: row.rowNumber,
        },
      }),
  );
}

export async function getIngestPreview() {
  const rows = await loadSupportCaseRows();

  return {
    totalRows: rows.length,
    groundTruthRows: rows.filter((row) => row.qualityTier === "ground_truth").length,
    autoOnlyRows: rows.filter((row) => row.qualityTier === "auto_only").length,
    rows,
    documents: buildSupportCaseDocuments(rows),
  };
}
