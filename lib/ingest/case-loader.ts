import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

import { Document } from "@langchain/core/documents";
import { parse } from "csv-parse/sync";

import { getAppConfig } from "@/lib/config";
import type { SupportCaseRow } from "@/lib/types";

type RawCsvRow = Record<string, string | undefined>;
type GuideSection = {
  title: string;
  body: string;
};
type GuideChunk = GuideSection & {
  chunkIndex: number;
};

const BASELINE_GUIDE_FILE_NAME = "kiosk_baseline_guide.md";

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

function splitGuideIntoSections(markdown: string) {
  const normalized = markdown.replace(/\r\n/g, "\n").trim();

  if (!normalized) {
    return [] as GuideSection[];
  }

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
    });
  }

  return sections;
}

function chunkGuideSection(section: GuideSection, chunkCharLimit = 2200) {
  const paragraphs = section.body
    .split(/\n{2,}/)
    .map((item) => item.trim())
    .filter(Boolean);

  if (paragraphs.length === 0) {
    return [] as GuideChunk[];
  }

  const chunks: GuideChunk[] = [];
  let buffer: string[] = [];
  let currentLength = 0;

  for (const paragraph of paragraphs) {
    const nextLength = currentLength + paragraph.length + (buffer.length > 0 ? 2 : 0);

    if (nextLength > chunkCharLimit && buffer.length > 0) {
      chunks.push({
        title: section.title,
        body: buffer.join("\n\n"),
        chunkIndex: chunks.length + 1,
      });
      buffer = [paragraph];
      currentLength = paragraph.length;
      continue;
    }

    buffer.push(paragraph);
    currentLength = nextLength;
  }

  if (buffer.length > 0) {
    chunks.push({
      title: section.title,
      body: buffer.join("\n\n"),
      chunkIndex: chunks.length + 1,
    });
  }

  return chunks;
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

export async function findBaselineGuideFiles() {
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
        if (!entry.isFile()) {
          continue;
        }

        if (entry.name.toLowerCase() === BASELINE_GUIDE_FILE_NAME) {
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
          source_type: "support_case",
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

export async function buildBaselineGuideDocuments() {
  const files = await findBaselineGuideFiles();

  if (files.length === 0) {
    return [] as Document[];
  }

  const documents: Document[] = [];

  for (const file of files) {
    const markdown = await readFile(file, "utf-8");
    const sections = splitGuideIntoSections(markdown);

    for (const section of sections) {
      const sectionChunks = chunkGuideSection(section);

      for (const chunk of sectionChunks) {
        documents.push(
          new Document({
            pageContent: [
              "문서 유형: 키오스크 기준 가이드",
              `섹션: ${chunk.title}`,
              "",
              chunk.body,
            ].join("\n"),
            metadata: {
              source_type: "baseline_guide",
              guide_section_title: chunk.title,
              guide_chunk_index: chunk.chunkIndex,
              source_file: path.basename(file),
              quality_tier: "guide_baseline",
            },
          }),
        );
      }
    }
  }

  return documents;
}

export async function getIngestPreview() {
  const rows = await loadSupportCaseRows();
  const caseDocuments = buildSupportCaseDocuments(rows);
  const guideDocuments = await buildBaselineGuideDocuments();
  const documents = [...caseDocuments, ...guideDocuments];

  return {
    totalRows: rows.length,
    groundTruthRows: rows.filter((row) => row.qualityTier === "ground_truth").length,
    autoOnlyRows: rows.filter((row) => row.qualityTier === "auto_only").length,
    guideDocuments: guideDocuments.length,
    totalDocuments: documents.length,
    rows,
    documents,
  };
}
