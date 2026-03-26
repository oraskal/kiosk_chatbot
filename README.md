# 유비케어 상담지원 챗봇

유비케어 병원고객팀 상담사가 고객 전화 응대 중 문제상황을 입력하면, 과거 상담 사례 CSV를 검색해서 아래 4가지를 바로 제안하는 내부용 RAG 웹앱입니다.

- 의심되는 원인
- 우선 확인사항
- 권장 대응 방향
- 병원 안내용 답변 초안

이 프로젝트는 말로만 설명하는 문서가 아니라, 실제로 로컬 실행과 Vercel 배포까지 가능한 Next.js 풀스택 앱입니다.

## 1. 기술 스택

- Next.js App Router
- React
- Route Handlers
- OpenAI API
- Supabase Postgres + pgvector
- Vercel

중요한 점은 아래입니다.

- 프론트와 백엔드를 같은 저장소에서 함께 관리합니다.
- CSV는 서비스 실행 중 매번 읽지 않습니다.
- CSV는 별도 인덱싱 명령어로 Supabase pgvector에 넣습니다.
- 앱 실행 중에는 Supabase에서 유사 사례를 검색합니다.

## 2. 현재 구현된 기능

- 상담사가 문제상황을 입력할 수 있는 간단한 내부 웹 UI
- `/api/chat` Route Handler 기반 RAG 응답 생성
- CSV 파일 여러 개를 읽어 Supabase pgvector에 적재하는 ingest 스크립트
- `/api/health` 헬스체크 API
- 유사 사례 개수, 최고 유사도 참고값, 가장 유사한 사례 요약 제공
- 유사 사례가 약할 때 단정하지 않고 확인 포인트 중심으로 답하는 fallback 로직

## 3. 폴더 구조

```text
ubcare-support-rag/
├─ app/
│  ├─ api/
│  │  ├─ chat/route.ts
│  │  └─ health/route.ts
│  ├─ globals.css
│  ├─ layout.tsx
│  └─ page.tsx
├─ components/
│  └─ support-workspace.tsx
├─ data/
│  └─ input/
│     └─ .gitkeep
├─ lib/
│  ├─ config.ts
│  ├─ ingest/
│  │  └─ case-loader.ts
│  ├─ rag/
│  │  ├─ vector-store.ts
│  │  └─ workflow.ts
│  ├─ supabase/
│  │  └─ admin.ts
│  └─ types.ts
├─ scripts/
│  └─ ingest.ts
├─ supabase/
│  └─ schema.sql
├─ .env.example
├─ package.json
└─ README.md
```

## 4. 먼저 준비해야 하는 것

아래 6가지를 먼저 준비해주세요.

1. Node.js
2. Git
3. GitHub 계정
4. Vercel 계정
5. Supabase 계정
6. OpenAI API Key

### 4-1. Node.js 설치 방법

1. 브라우저에서 `https://nodejs.org` 로 이동합니다.
2. `LTS` 버전을 설치합니다.
3. 설치 중 특별한 설정이 없다면 기본값으로 `Next`를 눌러도 됩니다.
4. 설치가 끝나면 PowerShell을 완전히 닫았다가 다시 엽니다.
5. 아래 명령어로 설치 확인을 합니다.

```powershell
node -v
npm -v
```

정상이라면 버전 번호가 나옵니다.

### 4-2. Git 설치 방법

1. 브라우저에서 `https://git-scm.com/download/win` 으로 이동합니다.
2. Windows 설치 파일을 다운로드합니다.
3. 설치 중 기본값으로 진행해도 됩니다.
4. 설치가 끝나면 PowerShell을 다시 열고 아래 명령어를 실행합니다.

```powershell
git --version
```

### 4-3. GitHub 계정 준비

1. `https://github.com` 에 접속합니다.
2. 회원가입을 합니다.
3. 이메일 인증을 완료합니다.

### 4-4. Vercel 계정 준비

1. `https://vercel.com` 에 접속합니다.
2. GitHub 계정으로 가입하거나 로그인합니다.
3. 나중에 GitHub 저장소를 가져와서 배포할 때 사용합니다.

### 4-5. Supabase 계정 준비

1. `https://supabase.com` 에 접속합니다.
2. 로그인합니다.
3. `New project` 를 눌러 새 프로젝트를 만듭니다.
4. 프로젝트 이름과 데이터베이스 비밀번호를 입력합니다.
5. 프로젝트가 완전히 준비될 때까지 기다립니다.

### 4-6. OpenAI API Key 준비

1. `https://platform.openai.com` 에 로그인합니다.
2. API Key를 발급합니다.
3. 발급된 키를 안전한 곳에 복사해 둡니다.
4. 이 키는 `.env.local` 에 넣습니다.

## 5. 이 프로젝트 받기

이미 폴더를 받았다면 이 단계는 건너뛰어도 됩니다.

새로 GitHub에서 내려받을 때는 아래처럼 합니다.

```powershell
git clone <여기에-저장소-주소>
cd <저장소-폴더명>
```

현재처럼 로컬 폴더에서 바로 시작하는 경우에는 아래 경로로 이동합니다.

```powershell
cd C:\kiosk_chatbot
```

## 6. Supabase 설정하기

### 6-1. SQL 실행하기

1. Supabase 프로젝트 화면으로 들어갑니다.
2. 왼쪽 메뉴에서 `SQL Editor` 를 클릭합니다.
3. `New query` 를 누릅니다.
4. 이 저장소의 `supabase/schema.sql` 파일 내용을 그대로 복사합니다.
5. Supabase SQL Editor에 붙여넣습니다.
6. `Run` 버튼을 눌러 실행합니다.

이 작업이 끝나면 벡터 검색용 테이블과 함수가 생성됩니다.

### 6-2. Supabase 값 확인 위치

Supabase 프로젝트에서 아래 값을 확인해야 합니다.

1. 왼쪽 아래 `Project Settings` 클릭
2. `API` 메뉴 클릭

여기서 아래 값을 찾습니다.

- `Project URL` -> `.env.local` 의 `SUPABASE_URL`
- `anon public` -> 이 프로젝트는 직접 사용하지 않지만, 같은 화면에 있어 위치를 기억해두면 좋습니다
- `service_role secret` -> `.env.local` 의 `SUPABASE_SERVICE_ROLE_KEY`

중요:

- `service_role key` 는 절대로 브라우저 코드에 넣으면 안 됩니다.
- 이 프로젝트는 Next.js 서버(Route Handler)에서만 그 키를 사용합니다.

## 7. 환경변수 파일 만들기

### 7-1. `.env.local` 파일 만들기

PowerShell에서 아래 명령어를 실행합니다.

```powershell
Copy-Item .env.example .env.local
```

이제 `.env.local` 파일을 메모장이나 VS Code로 열어서 값을 채웁니다.

예시:

```env
OPENAI_API_KEY=sk-xxxx
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

VECTOR_TABLE_NAME=support_case_documents
VECTOR_QUERY_NAME=match_support_case_documents

CSV_SOURCE_DIR=data/input

RAG_TOP_K=6
HIGH_CONFIDENCE_THRESHOLD=0.78
LOW_CONFIDENCE_THRESHOLD=0.58

NEXT_PUBLIC_APP_NAME=유비케어 상담지원 챗봇
```

### 7-2. 각 값 설명

- `OPENAI_API_KEY`: OpenAI API 키
- `OPENAI_CHAT_MODEL`: 답변 생성용 모델명
- `OPENAI_EMBEDDING_MODEL`: 임베딩 모델명
- `SUPABASE_URL`: Supabase Project URL
- `SUPABASE_SERVICE_ROLE_KEY`: Supabase service role key
- `VECTOR_TABLE_NAME`: 기본값 그대로 사용 권장
- `VECTOR_QUERY_NAME`: 기본값 그대로 사용 권장
- `CSV_SOURCE_DIR`: CSV를 넣어둘 폴더
- `RAG_TOP_K`: 한 번에 참고할 유사 사례 개수
- `HIGH_CONFIDENCE_THRESHOLD`: 높은 신뢰도로 볼 유사도 기준
- `LOW_CONFIDENCE_THRESHOLD`: fallback 여부를 판단할 유사도 기준

## 8. CSV 파일 넣기

권장 위치는 아래 폴더입니다.

```text
data/input
```

예를 들어 다음처럼 두면 됩니다.

```text
data/input/kiosk_gt_annotation_master.csv
```

현재 코드에서는 아래 두 위치를 모두 찾습니다.

- `data/input`
- 프로젝트 루트

즉, 이미 루트에 CSV가 있다면 바로 인덱싱이 가능합니다. 다만 운영상 헷갈리지 않게 하려면 앞으로는 `data/input` 에 넣는 것을 권장합니다.

## 9. 패키지 설치

PowerShell에서 아래 명령어를 실행합니다.

```powershell
npm install
```

처음 설치에는 시간이 조금 걸릴 수 있습니다.

## 10. 인덱싱하기

중요:

- 앱 실행 전에 먼저 인덱싱을 해야 합니다.
- 인덱싱은 CSV를 읽어서 Supabase pgvector에 넣는 작업입니다.
- 서비스 실행 중에 CSV를 직접 읽지 않기 때문에 이 단계가 반드시 필요합니다.

### 10-1. 기존 데이터 비우고 다시 넣기

처음 시작하거나 CSV를 수정했다면 아래 명령어를 권장합니다.

```powershell
npm run reindex
```

### 10-2. 설명

- `npm run ingest`: 기존 데이터에 추가 적재
- `npm run reindex`: 기존 벡터 데이터를 비우고 다시 적재

실무에서는 CSV가 바뀌었으면 `npm run reindex` 를 쓰는 편이 안전합니다.

## 11. 로컬에서 실행하기

인덱싱이 끝났으면 아래 명령어를 실행합니다.

```powershell
npm run dev
```

브라우저에서 아래 주소를 엽니다.

```text
http://localhost:3000
```

## 12. 로컬 테스트 순서

아래 순서대로 확인하면 됩니다.

1. `http://localhost:3000/api/health` 접속
2. JSON에 `ok: true` 가 보이는지 확인
3. `indexed_documents` 값이 0보다 큰지 확인
4. `http://localhost:3000` 접속
5. 예시 문장을 하나 입력
6. 결과 카드가 나오는지 확인

예시 입력:

- 카드결제는 되는데 영수증이 2장 출력 안 된다고 함
- 키오스크에서 접수는 되는데 바코드 출력이 안 됨
- 의사랑 CRM에서 알림톡 발송 실패가 반복됨

## 13. 화면이 안 뜰 때 확인할 곳

아래 순서대로 확인하세요.

1. PowerShell 창에 오류가 있는지 확인합니다.
2. `npm run dev` 가 실제로 켜져 있는지 확인합니다.
3. `.env.local` 값이 비어 있지 않은지 확인합니다.
4. Supabase SQL을 실행했는지 확인합니다.
5. `npm run reindex` 를 먼저 했는지 확인합니다.
6. `http://localhost:3000/api/health` 에서 오류 메시지를 확인합니다.

## 14. 자주 나는 오류와 해결법

### 오류 1. `node` 명령어를 찾을 수 없다고 나옴

원인:

- Node.js가 설치되지 않았거나
- 설치 후 PowerShell을 다시 열지 않았습니다

해결:

1. Node.js LTS를 설치합니다.
2. PowerShell을 완전히 닫고 다시 엽니다.
3. `node -v` 로 확인합니다.

### 오류 2. `필수 환경변수가 비어 있습니다` 라고 나옴

원인:

- `.env.local` 파일 값이 비어 있거나 오타가 있습니다

해결:

1. `.env.local` 을 열어 확인합니다.
2. 특히 아래 3개를 다시 확인합니다.
   - `OPENAI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`

### 오류 3. `relation does not exist` 또는 테이블이 없다고 나옴

원인:

- Supabase SQL을 아직 실행하지 않았습니다

해결:

1. `supabase/schema.sql` 내용을 SQL Editor에서 실행합니다.
2. 다시 `npm run reindex` 를 실행합니다.

### 오류 4. `indexed_documents` 가 0으로 나옴

원인:

- 인덱싱이 아직 안 되었거나
- CSV 파일을 못 찾고 있습니다

해결:

1. CSV가 `data/input` 또는 프로젝트 루트에 있는지 확인합니다.
2. `npm run reindex` 를 다시 실행합니다.
3. `CSV_SOURCE_DIR` 값이 맞는지 확인합니다.

### 오류 5. 답변은 오는데 유사 사례가 너무 적음

원인:

- 현재 threshold 값이 높을 수 있습니다

해결:

1. `.env.local` 의 `LOW_CONFIDENCE_THRESHOLD` 를 조금 낮춰봅니다.
2. `RAG_TOP_K` 를 늘려봅니다.
3. CSV 데이터 품질을 보강한 뒤 다시 `npm run reindex` 합니다.

## 15. CSV를 수정하면 언제 재인덱싱해야 하나요?

아래 중 하나라도 바뀌면 재인덱싱이 필요합니다.

- 새 CSV 파일을 추가했을 때
- 기존 CSV의 행이 늘었을 때
- `gt_problem_summary` 가 바뀌었을 때
- `gt_root_cause` 가 바뀌었을 때
- `gt_resolution_action` 가 바뀌었을 때
- `gt_customer_reply` 가 바뀌었을 때

추천 명령어:

```powershell
npm run reindex
```

이유:

- 검색과 생성 품질은 Supabase에 들어간 임베딩 데이터 기준으로 동작하기 때문입니다.
- CSV만 바꾸고 재인덱싱하지 않으면 앱은 예전 벡터 데이터를 계속 사용합니다.

## 16. GitHub에 올리는 방법

아래는 완전 처음부터 하는 순서입니다.

### 16-1. GitHub에서 새 저장소 만들기

1. GitHub 로그인
2. 오른쪽 위 `+` 버튼 클릭
3. `New repository` 클릭
4. 저장소 이름 입력
5. `Create repository` 클릭

### 16-2. 로컬 폴더를 Git 저장소로 만들기

PowerShell에서 아래 명령어를 순서대로 실행합니다.

```powershell
git init
git add .
git commit -m "Initial internal RAG app"
```

### 16-3. GitHub 저장소와 연결하기

GitHub 새 저장소 화면에 보이는 주소를 복사한 뒤 아래처럼 실행합니다.

```powershell
git remote add origin <깃허브-저장소-주소>
git branch -M main
git push -u origin main
```

## 17. Vercel 배포 방법

중요:

- Vercel은 웹앱 배포용입니다.
- CSV 인덱싱은 Supabase에 데이터를 넣는 작업이라 보통 로컬에서 먼저 실행합니다.
- 한 번 Supabase에 데이터가 들어가면 Vercel 앱은 그 데이터를 조회합니다.

### 17-1. GitHub에 코드가 올라간 상태여야 합니다

먼저 16번 절차를 완료하세요.

### 17-2. Vercel에서 프로젝트 가져오기

1. Vercel 로그인
2. `Add New...` 클릭
3. `Project` 클릭
4. GitHub 저장소 목록에서 이 프로젝트를 선택
5. `Import` 클릭

### 17-3. 환경변수 넣기

배포 전에 `Environment Variables` 영역에 아래 값을 넣습니다.

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `VECTOR_TABLE_NAME`
- `VECTOR_QUERY_NAME`
- `RAG_TOP_K`
- `HIGH_CONFIDENCE_THRESHOLD`
- `LOW_CONFIDENCE_THRESHOLD`
- `NEXT_PUBLIC_APP_NAME`

참고:

- `CSV_SOURCE_DIR` 는 Vercel 런타임에서는 사실상 필요하지 않습니다.
- 이유는 배포된 앱이 CSV를 직접 읽지 않고, 이미 Supabase에 적재된 벡터 데이터를 조회하기 때문입니다.

### 17-4. Deploy 클릭

설정이 끝나면 `Deploy` 를 클릭합니다.

### 17-5. 배포 후 테스트

1. 배포 URL 접속
2. `/api/health` 확인
3. `ok: true` 와 `indexed_documents > 0` 인지 확인
4. 메인 화면에서 예시 문장 2~3개 테스트

## 18. 운영 시 권장 작업 순서

처음 세팅할 때:

1. Supabase 프로젝트 생성
2. `supabase/schema.sql` 실행
3. `.env.local` 설정
4. `npm install`
5. `npm run reindex`
6. `npm run dev`
7. 로컬 확인
8. GitHub 업로드
9. Vercel 배포

CSV가 나중에 바뀌었을 때:

1. CSV 파일 교체 또는 추가
2. `npm run reindex`
3. `http://localhost:3000/api/health` 확인
4. 필요하면 다시 배포

중요:

- 데이터는 Supabase에 있으므로, 단순 UI 코드 변경만 있으면 항상 재인덱싱이 필요한 것은 아닙니다.
- CSV 내용이 바뀐 경우에만 재인덱싱이 필요합니다.

## 19. 설계 요약

이 프로젝트는 아래 흐름으로 작동합니다.

1. CSV를 로컬에서 읽습니다.
2. 각 상담 row를 검색용 문서로 정리합니다.
3. OpenAI 임베딩으로 벡터를 만듭니다.
4. Supabase pgvector 테이블에 저장합니다.
5. 상담사가 웹에서 문제상황을 입력합니다.
6. Next.js Route Handler가 유사 사례를 Supabase에서 검색합니다.
7. 검색된 사례를 바탕으로 OpenAI가 구조화 응답을 생성합니다.
8. 프론트 화면에 실무형 카드 형태로 보여줍니다.

## 20. 앞으로 추가하면 좋은 것

이번 버전은 빠르게 작동하는 내부 도구에 집중했습니다. 다음 단계로는 아래를 고려할 수 있습니다.

- 로그인 또는 사내 SSO
- 상담 카테고리별 필터
- CSV 외에 티켓 시스템 연동
- 정답셋 200건 기반 자동 평가 스크립트
- 자주 쓰는 답변 복사 버튼 분리
- 상담 이력 저장

## 21. 한 줄 요약

이 프로젝트는 `Next.js + Supabase pgvector + OpenAI` 조합으로 만든, 실제 배포 가능한 내부 상담용 RAG 웹앱입니다.  
CSV를 `npm run reindex` 로 먼저 적재한 뒤, `npm run dev` 또는 Vercel 배포로 바로 사용할 수 있습니다.
