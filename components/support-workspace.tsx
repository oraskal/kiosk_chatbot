"use client";

import { useMemo, useState, useTransition } from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { AnswerSectionGroup, ChatApiResponse } from "@/lib/types";

type SectionKey = "suspected_causes" | "checks" | "actions";

function getSectionGroups(result: ChatApiResponse, key: SectionKey): AnswerSectionGroup[] {
  const groups = result.section_groups?.[key];

  if (groups?.length) {
    return groups;
  }

  const fallbackItems =
    key === "suspected_causes" ? result.suspected_causes : key === "checks" ? result.checks : result.actions;

  if (fallbackItems.length === 0) {
    return [];
  }

  return [
    {
      list_style: key === "actions" && result.query_mode === "guide" ? "ordered" : "bullet",
      items: fallbackItems,
    },
  ];
}

function buildSectionCopyLines(title: string, groups: AnswerSectionGroup[]) {
  if (groups.length === 0) {
    return [];
  }

  const lines = [title];

  groups.forEach((group, groupIndex) => {
    if (group.title) {
      lines.push(group.title);
    }

    lines.push(
      ...group.items.map((item, index) => (group.list_style === "ordered" ? `${index + 1}. ${item}` : `- ${item}`)),
    );

    if (groupIndex < groups.length - 1) {
      lines.push("");
    }
  });

  return lines;
}

function buildCopyText(result: ChatApiResponse) {
  const lines: string[] = [];
  const suspectedCauseGroups = getSectionGroups(result, "suspected_causes");
  const checkGroups = getSectionGroups(result, "checks");
  const actionGroups = getSectionGroups(result, "actions");

  lines.push(...buildSectionCopyLines("의심되는 원인", suspectedCauseGroups));

  if (lines.length > 0 && checkGroups.length > 0) {
    lines.push("");
  }

  lines.push(...buildSectionCopyLines("우선 확인사항", checkGroups));

  if (lines.length > 0 && actionGroups.length > 0) {
    lines.push("");
  }

  lines.push(...buildSectionCopyLines("권장 대응 방향", actionGroups));

  if (result.baseline_reference?.markdown_excerpt) {
    lines.push("", "설치가이드 기준 보기", result.baseline_reference.markdown_excerpt);
  } else if (result.baseline_reference?.excerpts.length) {
    lines.push(
      "",
      "설치가이드 기준 보기",
      `참조 문서: ${[...result.baseline_reference.source_files, ...result.baseline_reference.source_titles].filter(Boolean).join(" / ")}`,
      ...result.baseline_reference.excerpts.map((item) => `- ${item}`),
    );
  }

  if (result.similar_cases.length > 0) {
    lines.push("", "유사 사례 요약");
    result.similar_cases.forEach((item) => {
      lines.push(
        `- 증상: ${item.problem_summary} | 원인: ${item.root_cause || "기록 없음"} | 조치: ${item.resolution_action || "기록 없음"} | 결과: ${
          item.resolution_result || "기록 없음"
        }`,
      );
    });
  }

  return lines.join("\n");
}

function SectionGroups({ groups }: { groups: AnswerSectionGroup[] }) {
  return (
    <div className="section-groups">
      {groups.map((group, groupIndex) => {
        const ListTag = group.list_style === "ordered" ? "ol" : "ul";

        return (
          <div key={`${group.title ?? "group"}-${groupIndex}`} className="section-group">
            {group.title ? <h4 className="section-group-title">{group.title}</h4> : null}
            <ListTag className="result-list">
              {group.items.map((item, itemIndex) => (
                <li key={`${item}-${itemIndex}`}>{item}</li>
              ))}
            </ListTag>
          </div>
        );
      })}
    </div>
  );
}

export function SupportWorkspace() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<ChatApiResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [isPending, startTransition] = useTransition();

  const promptLength = prompt.trim().length;
  const canSubmit = promptLength > 0 && !isPending;

  const confidenceClass = useMemo(() => {
    if (!result) return "";
    if (result.confidence_level === "low") return "warning";
    return "";
  }, [result]);

  async function handleSubmit() {
    if (!prompt.trim()) {
      setErrorMessage("문제상황을 먼저 입력해 주세요.");
      return;
    }

    setErrorMessage("");
    setStatusMessage("분석 중입니다...");

    startTransition(async () => {
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: prompt.trim() }),
        });

        const payload = (await response.json()) as ChatApiResponse | { error?: string };

        if (!response.ok || !("suspected_causes" in payload)) {
          const nextError =
            "error" in payload && typeof payload.error === "string" ? payload.error : "요청을 처리하지 못했습니다.";
          throw new Error(nextError);
        }

        setResult(payload);
        setStatusMessage("");
      } catch (error) {
        const message = error instanceof Error ? error.message : "알 수 없는 오류로 응답을 만들지 못했습니다.";
        setResult(null);
        setErrorMessage(message);
        setStatusMessage("");
      }
    });
  }

  async function handleCopy() {
    if (!result) return;
    await navigator.clipboard.writeText(buildCopyText(result));
    setStatusMessage("클립보드에 복사했습니다.");
  }

  const suspectedCauseGroups = result ? getSectionGroups(result, "suspected_causes") : [];
  const checkGroups = result ? getSectionGroups(result, "checks") : [];
  const actionGroups = result ? getSectionGroups(result, "actions") : [];

  return (
    <main className="page-shell">
      <div className="page-grid">
        <section className="hero-card">
          <span className="eyebrow">Ubcare Internal RAG Assistant</span>
          <h1 className="hero-title">유비케어 병원고객팀 상담지원 챗봇</h1>
          <p className="hero-subtitle">
            상담 중 접수된 문제상황을 입력하면 설치가이드 기준과 과거 유사 사례를 분리해 읽기 쉬운 현장 대응 답변으로 정리합니다.
          </p>
        </section>

        <section className="workspace-grid">
          <div className="panel-card">
            <h2 className="panel-title">문제상황 입력</h2>

            <textarea
              className="prompt-box"
              placeholder="예: 랜선은 점검했는데도 키오스크가 인터넷에 연결되지 않고, 결제와 접수가 모두 안 됩니다."
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />

            <div className="field-row">
              <span className="char-count">{promptLength}자</span>
            </div>

            <button className="primary-button" type="button" onClick={handleSubmit} disabled={!canSubmit}>
              {isPending ? "분석 중..." : "유사 사례 찾고 답변 만들기"}
            </button>

            {(errorMessage || statusMessage) && (
              <div className={`status-line ${errorMessage ? "error" : confidenceClass}`.trim()}>
                {errorMessage || statusMessage}
              </div>
            )}
          </div>

          <div className="panel-card">
            {!result ? (
              <div className="empty-state">
                <span>문제상황을 입력하면 결과가 여기에 표시됩니다.</span>
              </div>
            ) : (
              <>
                <div className="results-header">
                  <h2 className="panel-title">추천 결과</h2>
                  <button className="secondary-button" type="button" onClick={handleCopy}>
                    전체 복사
                  </button>
                </div>

                <div className="results-stack">
                  {suspectedCauseGroups.length > 0 && (
                    <section className="result-card">
                      <h3>의심되는 원인</h3>
                      <SectionGroups groups={suspectedCauseGroups} />
                    </section>
                  )}

                  {checkGroups.length > 0 && (
                    <section className="result-card">
                      <h3>우선 확인사항</h3>
                      <SectionGroups groups={checkGroups} />
                    </section>
                  )}

                  {actionGroups.length > 0 && (
                    <section className="result-card">
                      <h3>권장 대응 방향</h3>
                      <SectionGroups groups={actionGroups} />
                    </section>
                  )}

                  {result.baseline_reference?.markdown_excerpt ? (
                    <section className="result-card guide-support-card">
                      <h3>설치가이드 기준 보기</h3>
                      <div className="guide-markdown-shell">
                        <div className="guide-markdown markdown-body">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.baseline_reference.markdown_excerpt}</ReactMarkdown>
                        </div>
                      </div>
                    </section>
                  ) : result.baseline_reference?.excerpts.length ? (
                    <section className="result-card">
                      <h3>설치가이드 기준 보기</h3>
                      <div className="baseline-meta">
                        {[...result.baseline_reference.source_files, ...result.baseline_reference.source_titles].join(" / ")}
                      </div>
                      <ul className="result-list">
                        {result.baseline_reference.excerpts.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </section>
                  ) : null}

                  {result.similar_cases.length > 0 && (
                    <section className="result-card">
                      <h3>유사 사례 요약</h3>
                      <div className="case-list">
                        {result.similar_cases.map((item) => (
                          <article key={item.case_key} className="case-card">
                            <h4>{item.problem_summary}</h4>
                            <p>
                              <strong>증상:</strong> {item.problem_summary}
                            </p>
                            <p>
                              <strong>원인:</strong> {item.root_cause || "기록 없음"}
                            </p>
                            <p>
                              <strong>조치:</strong> {item.resolution_action || "기록 없음"}
                            </p>
                            <p>
                              <strong>결과:</strong> {item.resolution_result || "기록 없음"}
                            </p>
                            <div className="meta-row">
                              <span className="meta-pill">사례 {item.case_key}</span>
                              <span className="meta-pill">유사도 {item.similarity_score.toFixed(2)}</span>
                              {item.issue_subtype_label && <span className="meta-pill">{item.issue_subtype_label}</span>}
                            </div>
                          </article>
                        ))}
                      </div>
                    </section>
                  )}

                  <section className="result-card">
                    <h3>응답 메타</h3>
                    <div className="meta-row">
                      <span className="meta-pill">신뢰도 {result.confidence_level}</span>
                      <span className="meta-pill">유사 사례 {result.similar_case_count}건</span>
                      {typeof result.top_similarity === "number" && (
                        <span className="meta-pill">최고 유사도 {result.top_similarity.toFixed(2)}</span>
                      )}
                    </div>
                  </section>
                </div>
              </>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
