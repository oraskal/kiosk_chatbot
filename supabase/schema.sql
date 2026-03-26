-- Embedding model default: text-embedding-3-small (1536 dimensions)
-- If you change to a different embedding model with another dimension,
-- update vector(1536) below to the correct size before indexing.

create extension if not exists vector;

create table if not exists support_case_documents (
  id bigserial primary key,
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(1536)
);

create index if not exists support_case_documents_embedding_idx
on support_case_documents
using hnsw (embedding vector_cosine_ops);

create index if not exists support_case_documents_metadata_idx
on support_case_documents
using gin (metadata);

alter table support_case_documents enable row level security;

create or replace function match_support_case_documents (
  query_embedding vector(1536),
  match_count int default 5,
  filter jsonb default '{}'::jsonb
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    support_case_documents.id,
    support_case_documents.content,
    support_case_documents.metadata,
    1 - (support_case_documents.embedding <=> query_embedding) as similarity
  from support_case_documents
  where support_case_documents.metadata @> filter
  order by support_case_documents.embedding <=> query_embedding
  limit least(greatest(match_count, 1), 50);
end;
$$;
