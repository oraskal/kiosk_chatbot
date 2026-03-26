"use client";

import { useMemo, useState, useTransition } from "react";

import type { ChatApiResponse } from "@/lib/types";

const samplePrompts = [
  "카드결제는 되는데 영수증이 2장 출력되지 않는다고 합니다.",
  "키오스크에서 접수는 되는데 바코드 출력이 안 된다고 합니다.",
  "의사랑 CRM에서 알림톡 발송 실패가 반복된다고 합니다.",
];

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
    "",
    `신뢰도 메모: ${result.confidence_note}`,
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
  const [statusMessage, setStatusMessage] = useState(
    "문제상황을 입력하면 유사 사례를 찾고 실무형 응답 초안을 생성합니다.",
  );
  const [isPending, startTransition] = useTransition();

  const promptLength = prompt.trim().length;
  const canSubmit = promptLength > 0 && !isPending;

  const confidenceClass = useMemo(() => {
    if (!result) {
      return "";
    }

    if (result.confidence_level === "low") {
      return "warning";
    }

    return "";
  }, [result]);

  async function handleSubmit() {
    if (!prompt.trim()) {
      setErrorMessage("문제상황을 먼저 입력해주세요.");
      return;
    }

    setErrorMessage("");
    setStatusMessage("유사 사례를 검색하고 답변 초안을 만드는 중입니다...");

    startTransition(async () => {
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: prompt.trim(),
          }),
        });

        const payload = (await response.json()) as ChatApiResponse | { error?: string };

        if (!response.ok || !("suspected_causes" in payload)) {
          throw new Error(payload.error || "요청을 처리하지 못했습니다.");
        }

        setResult(payload);
        setStatusMessage("답변 초안이 준비되었습니다. 필요한 문장만 바로 복사해서 활용하시면 됩니다.");
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "알 수 없는 오류로 응답을 만들지 못했습니다.";

        setResult(null);
        setErrorMessage(message);
        setStatusMessage("설정값 또는 인덱싱 상태를 확인해주세요.");
      }
    });
  }

  async function handleCopy() {
    if (!result) {
      return;
    }

    await navigator.clipboard.writeText(buildCopyText(result));
    setStatusMessage("결과 전체를 클립보드에 복사했습니다.");
  }

  return (
    <main className="page-shell">
      <div className="page-grid">
        <section className="hero-card">
          <span className="eyebrow">Ubcare Internal RAG Assistant</span>
          <h1 className="hero-title">유비케어 병원고객팀 상담지원 챗봇</h1>
          <p className="hero-subtitle">
            상담 중 들은 문제상황을 넣으면, 과거 사례를 검색해 의심 원인과 확인 포인트,
            권장 대응 방향, 병원 안내용 답변 초안을 실무형으로 정리합니다.
          </p>
          <ul className="hero-points">
            <li>Next.js App Router + Route Handlers 기반 풀스택 구조</li>
            <li>Supabase pgvector 검색 + OpenAI 기반 구조화 응답 생성</li>
            <li>상담사가 바로 읽거나 복붙할 수 있는 실무형 출력</li>
          </ul>
        </section>

        <section className="workspace-grid">
          <div className="panel-card">
            <h2 className="panel-title">문제상황 입력</h2>
            <p className="panel-copy">
              병원에서 들은 증상과 상황을 자연어로 적어주세요. 짧게 적어도 되지만, 가능한
              한 증상과 시도해본 내용을 같이 적으면 더 정확합니다.
            </p>

            <div className="chip-row">
              {samplePrompts.map((item) => (
                <button
                  key={item}
                  className="chip"
                  type="button"
                  onClick={() => setPrompt(item)}
                >
                  {item}
                </button>
              ))}
            </div>

            <textarea
              className="prompt-box"
              placeholder="예) 키오스크에서 접수는 되는데 바코드 출력이 안 되고, 재부팅해도 동일하다고 합니다."
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />

            <div className="field-row">
              <span>입력 길이: {promptLength}자</span>
              <span>빈 입력은 전송되지 않습니다.</span>
            </div>

            <button className="primary-button" type="button" onClick={handleSubmit} disabled={!canSubmit}>
              {isPending ? "분석 중..." : "유사 사례 찾고 답변 만들기"}
            </button>

            <div className={`status-line ${errorMessage ? "error" : confidenceClass}`.trim()}>
              {errorMessage || statusMessage}
            </div>

            <p className="footer-note">
              검색된 사례는 참고자료입니다. 유사 사례가 부족하면 챗봇이 단정하지 않고 추가
              확인 포인트 중심으로 안내합니다.
            </p>
          </div>

          <div className="panel-card">
            {!result ? (
              <div className="empty-state">
                <div>
                  <strong>아직 생성된 답변이 없습니다.</strong>
                  문제상황을 입력한 뒤 왼쪽 버튼을 누르면 결과 카드가 여기에 표시됩니다.
                </div>
              </div>
            ) : (
              <>
                <div className="results-header">
                  <div>
                    <h2 className="panel-title">추천 결과</h2>
                    <p className="panel-copy">{result.confidence_note}</p>
                  </div>
                  <button className="secondary-button" type="button" onClick={handleCopy}>
                    결과 전체 복사
                  </button>
                </div>

                <div className="results-grid">
                  <section className="result-card">
                    <h3>1. 의심되는 원인</h3>
                    <ol className="result-list">
                      {result.suspected_causes.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ol>
                  </section>

                  <section className="result-card">
                    <h3>2. 우선 확인사항</h3>
                    <ol className="result-list">
                      {result.checks.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ol>
                  </section>

                  <section className="result-card">
                    <h3>3. 권장 대응 방향</h3>
                    <ol className="result-list">
                      {result.next_actions.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ol>
                  </section>

                  <section className="result-card">
                    <h3>4. 병원 안내용 답변 초안</h3>
                    <div className="reply-box">{result.customer_reply_draft}</div>
                    <div className="meta-row">
                      <span className="meta-pill">신뢰도: {result.confidence_level}</span>
                      <span className="meta-pill">유사 사례 {result.similar_case_count}건 참고</span>
                      {typeof result.top_similarity === "number" ? (
                        <span className="meta-pill">
                          최고 유사도 {result.top_similarity.toFixed(2)}
                        </span>
                      ) : null}
                    </div>
                  </section>

                  {result.most_similar_case_summary ? (
                    <section className="result-card">
                      <h3>가장 유사한 사례 요약</h3>
                      <div className="reply-box">{result.most_similar_case_summary}</div>
                    </section>
                  ) : null}

                  <section className="result-card">
                    <h3>유사 사례 참고</h3>
                    <div className="results-grid">
                      {result.similar_cases.length > 0 ? (
                        result.similar_cases.map((item) => (
                          <article key={item.case_key} className="case-card">
                            <h4>{item.problem_summary}</h4>
                            <p>
                              <strong>원인 후보:</strong> {item.root_cause || "기록 없음"}
                            </p>
                            <p>
                              <strong>실제 조치:</strong> {item.resolution_action || "기록 없음"}
                            </p>
                            <p>
                              <strong>처리 결과:</strong> {item.resolution_result || "기록 없음"}
                            </p>
                            <div className="meta-row">
                              <span className="meta-pill">사례키 {item.case_key}</span>
                              <span className="meta-pill">
                                유사도 {item.similarity_score.toFixed(2)}
                              </span>
                              {item.issue_subtype_label ? (
                                <span className="meta-pill">{item.issue_subtype_label}</span>
                              ) : null}
                            </div>
                          </article>
                        ))
                      ) : (
                        <p className="panel-copy">
                          충분히 유사한 사례를 찾지 못했습니다. 확인 포인트 중심으로 안내해드리는
                          fallback 응답입니다.
                        </p>
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
