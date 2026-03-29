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

export interface ChatApiResponse {
  query_mode: QueryMode;
  suspected_causes: string[];
  checks: string[];
  next_actions: string[];
  guide_overview: string;
  guide_steps: string[];
  confidence_level: ConfidenceLevel;
  confidence_note: string;
  similar_case_count: number;
  top_similarity: number | null;
  most_similar_case_summary: string;
  similar_cases: SimilarCaseSummary[];
  fallback_used: boolean;
}
