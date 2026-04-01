export type ResolutionResult = "resolved" | "monitoring" | "workaround" | "unresolved" | "";

export type ConfidenceLevel = "high" | "medium" | "low";
export type QueryMode = "incident" | "guide";

export interface SupportCaseRow {
  sourceFile: string;
  rowNumber: number;
  caseId: string;
  caseKey: string;
  clinicName: string;
  issueCategory: string;
  issueSubtype: string;
  issueSubtypeLabel: string;
  issueKeyRaw: string;
  autoIssueSummary: string;
  autoIssueDetail: string;
  autoLatestAction: string;
  gtProblemSummary: string;
  gtRootCause: string;
  gtResolutionAction: string;
  gtResolutionResult: ResolutionResult;
  gtCustomerReply: string;
  qualityTier: "ground_truth" | "auto_only";
}

export interface SimilarCaseSummary {
  case_key: string;
  clinic_name: string;
  issue_subtype_label: string;
  problem_summary: string;
  root_cause: string;
  resolution_action: string;
  resolution_result: string;
  similarity_score: number;
}

export type ListStyle = "bullet" | "ordered";

export interface AnswerSectionGroup {
  title?: string;
  list_style: ListStyle;
  items: string[];
}

export interface StructuredAnswerSections {
  suspected_causes: AnswerSectionGroup[];
  checks: AnswerSectionGroup[];
  actions: AnswerSectionGroup[];
}

export interface BaselineReference {
  source_titles: string[];
  source_files: string[];
  excerpts: string[];
  section_title?: string;
  markdown_excerpt?: string;
}

export interface ChatApiResponse {
  query_mode: QueryMode;
  suspected_causes: string[];
  checks: string[];
  actions: string[];
  section_groups: StructuredAnswerSections;
  baseline_reference?: BaselineReference | null;
  confidence_level: ConfidenceLevel;
  confidence_note: string;
  similar_case_count: number;
  top_similarity: number | null;
  similar_cases: SimilarCaseSummary[];
  fallback_used: boolean;
}
