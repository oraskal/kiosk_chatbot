import { config as loadEnv } from "dotenv";

import { buildSupportCaseDocuments, getIngestPreview } from "@/lib/ingest/case-loader";
import { clearVectorTable, getSupabaseVectorStore } from "@/lib/rag/vector-store";

async function main() {
  loadEnv({ path: ".env.local" });
  loadEnv();

  const shouldReset = process.argv.includes("--reset");
  const preview = await getIngestPreview();

  console.log("");
  console.log("=== 유비케어 상담 사례 인덱싱 시작 ===");
  console.log(`총 CSV 행 수: ${preview.totalRows}`);
  console.log(`Ground truth 포함 행 수: ${preview.groundTruthRows}`);
  console.log(`Auto only 행 수: ${preview.autoOnlyRows}`);

  const documents = buildSupportCaseDocuments(preview.rows);

  if (shouldReset) {
    console.log("기존 벡터 문서를 비우는 중...");
    await clearVectorTable();
  }

  const vectorStore = getSupabaseVectorStore();

  console.log(`임베딩 및 적재 중... (${documents.length}건)`);
  await vectorStore.addDocuments(documents);

  console.log("인덱싱이 완료되었습니다.");
  console.log("이제 npm run dev 로 앱을 실행한 뒤 http://localhost:3000 에서 확인하세요.");
  console.log("");
}

main().catch((error) => {
  console.error("인덱싱 실패:", error);
  process.exit(1);
});
