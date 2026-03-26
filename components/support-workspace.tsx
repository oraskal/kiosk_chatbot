"use client";

import { useMemo, useState, useTransition } from "react";

import type { ChatApiResponse } from "@/lib/types";

function buildCopyText(result: ChatApiResponse) {
  const lines = [
    "의심되는 원인",
    ...result.suspected_causes.map((item, index) => `${index + 1}. ${item}`),
    "",
    "우선 확인사항",
    ...result.checks.map((item, index) => `${index + 1}. ${item}`),
    "",
    "권장 대응 방향",
    ...result.next_actions.map((item, index) => `${index + 1}. ${item}`),
    "",
    "병원 안내용 답변 초안",
    result.customer_reply_draft,
  ];

  if (result.most_similar_case_summary) {
    lines.push("", `가장 유사한 사례 요약: ${result.most_similar_case_summary}`);
  }

  return lines.join("\n");
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
      setErrorMessage("문제상황을 먼저 입력해주세요.");
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
          const errorMessage =
            "error" in payload && typeof payload.error === "string"
              ? payload.error
              : "요청을 처리하지 못했습니다.";
          throw new Error(errorMessage);
        }

        setResult(payload);
        setStatusMessage("");
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "알 수 없는 오류로 응답을 만들지 못했습니다.";
        setResult(null);
        setErrorMessage(message);
        setStatusMessage("");
      }
    });
  }

  async function handleCopy() {
    if (!result) return;
    await navigator.clipboard.writeText(buildCopyText(result));
    setStatusMessage("클립보드에 복사되었습니다.");
  }

  return (
    <main className="page-shell">
      <div className="page-grid">
        <section className="hero-card">
          <span className="eyebrow">Ubcare Internal RAG Assistant</span>
          <h1 className="hero-title">유비케어 병원고객팀 상담지원 챗봇</h1>
          <p className="hero-subtitle">
            상담 중 접수된 문제상황을 입력하면 과거 유사 사례를 검색하여 원인·확인사항·대응 방향·답변 초안을 정리합니다.
          </p>
        </section>

        <section className="workspace-grid">
          <div className="panel-card">
            <h2 className="panel-title">문제상황 입력</h2>

            <textarea
              className="prompt-box"
              placeholder="예) 키오스크에서 접수는 되는데 바코드 출력이 안 되고, 재부팅해도 동일하다고 합니다."
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

                <div className="results-grid">
                  <section className="result-card">
                    <h3>의심되는 원인</h3>
                    <ol className="result-list">
                      {result.suspected_causes.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ol>
                  </section>

                  <section className="result-card">
                    <h3>우선 확인사항</h3>
                    <ol className="result-list">
                      {result.checks.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ol>
                  </section>

                  <section className="result-card">
                    <h3>권장 대응 방향</h3>
                    <ol className="result-list">
                      {result.next_actions.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ol>
                  </section>

                  <section className="result-card">
                    <h3>병원 안내용 답변 초안</h3>
                    <div className="reply-box">{result.customer_reply_draft}</div>
                    <div className="meta-row">
                      <span className="meta-pill">신뢰도: {result.confidence_level}</span>
                      <span className="meta-pill">유사 사례 {result.similar_case_count}건</span>
                      {typeof result.top_similarity === "number" && (
                        <span className="meta-pill">
                          최고 유사도 {result.top_similarity.toFixed(2)}
                        </span>
                      )}
                    </div>
                  </section>

                  {result.most_similar_case_summary && (
                    <section className="result-card">
                      <h3>가장 유사한 사례 요약</h3>
                      <div className="reply-box">{result.most_similar_case_summary}</div>
                    </section>
                  )}

                  {result.similar_cases.length > 0 && (
                    <section className="result-card">
                      <h3>유사 사례 참고</h3>
                      <div className="results-grid">
                        {result.similar_cases.map((item) => (
                          <article key={item.case_key} className="case-card">
                            <h4>{item.problem_summary}</h4>
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
                              {item.issue_subtype_label && (
                                <span className="meta-pill">{item.issue_subtype_label}</span>
                              )}
                            </div>
                          </article>
                        ))}
                      </div>
                    </section>
                  )}
                </div>
              </>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
