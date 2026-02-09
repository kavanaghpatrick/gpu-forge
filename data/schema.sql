-- GPU Computing Knowledge Database Schema
-- Structured, queryable store for all research findings

-- The 11 expertise areas (skills)
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,          -- e.g. 'gpu-silicon'
    layer INTEGER NOT NULL,             -- 0-4
    title TEXT NOT NULL,                -- e.g. 'Apple GPU Microarchitecture'
    github_issue INTEGER,               -- issue number in gpucomputer repo
    description TEXT
);

-- Core findings table — every piece of knowledge goes here
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id INTEGER NOT NULL REFERENCES skills(id),
    topic TEXT NOT NULL,                 -- subcategory within the skill
    claim TEXT NOT NULL,                 -- the actual finding/assertion
    evidence TEXT,                       -- supporting evidence or explanation
    source_url TEXT,                     -- primary source URL
    source_title TEXT,                   -- human-readable source name
    source_type TEXT CHECK(source_type IN (
        'academic_paper', 'apple_docs', 'wwdc_session', 'github_repo',
        'blog_post', 'reverse_engineering', 'benchmark', 'empirical_test',
        'patent', 'forum_post', 'book', 'other'
    )) DEFAULT 'other',
    confidence TEXT CHECK(confidence IN (
        'verified',      -- tested/reproduced or from authoritative source
        'high',          -- strong evidence, multiple sources
        'medium',        -- single credible source, not independently verified
        'low',           -- speculative, inferred, or single informal source
        'unverified'     -- needs investigation
    )) DEFAULT 'unverified',
    date_found TEXT DEFAULT (date('now')),
    date_published TEXT,                 -- when the source was published
    tags TEXT,                           -- comma-separated tags for cross-cutting concerns
    notes TEXT,                          -- investigator notes
    investigation_session TEXT           -- links back to which /investigate run produced this
);

-- Citations table — for tracking academic references
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id INTEGER REFERENCES findings(id) ON DELETE CASCADE,
    authors TEXT,
    title TEXT NOT NULL,
    venue TEXT,                          -- e.g. 'ASPLOS 2023', 'WWDC25', 'arXiv'
    year INTEGER,
    doi TEXT,
    url TEXT,
    bibtex TEXT,
    author_source TEXT CHECK(author_source IN (
        'extracted_from_page',  -- authors confirmed by reading the actual source page
        'from_metadata',        -- authors from DOI/API metadata lookup
        'unverified'            -- authors recalled from memory or guessed — needs verification
    )) DEFAULT 'unverified'
);

-- Investigation log — tracks each /investigate session
CREATE TABLE IF NOT EXISTS investigations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id INTEGER REFERENCES skills(id),
    topic TEXT NOT NULL,
    github_issue INTEGER,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    queries_run INTEGER DEFAULT 0,       -- how many web searches performed
    findings_added INTEGER DEFAULT 0,    -- how many findings stored
    status TEXT CHECK(status IN ('running', 'completed', 'failed')) DEFAULT 'running',
    summary TEXT                         -- brief summary of what was found
);

-- Full-text search index on findings
CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(
    claim, evidence, tags, notes,
    content=findings,
    content_rowid=id
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
    INSERT INTO findings_fts(rowid, claim, evidence, tags, notes)
    VALUES (new.id, new.claim, new.evidence, new.tags, new.notes);
END;

CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, claim, evidence, tags, notes)
    VALUES ('delete', old.id, old.claim, old.evidence, old.tags, old.notes);
END;

CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, claim, evidence, tags, notes)
    VALUES ('delete', old.id, old.claim, old.evidence, old.tags, old.notes);
    INSERT INTO findings_fts(rowid, claim, evidence, tags, notes)
    VALUES (new.id, new.claim, new.evidence, new.tags, new.notes);
END;

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_findings_skill ON findings(skill_id);
CREATE INDEX IF NOT EXISTS idx_findings_confidence ON findings(confidence);
CREATE INDEX IF NOT EXISTS idx_findings_topic ON findings(topic);
CREATE INDEX IF NOT EXISTS idx_findings_source_type ON findings(source_type);
CREATE INDEX IF NOT EXISTS idx_citations_finding ON citations(finding_id);

-- Prevent duplicate citations (same paper inserted by parallel agents)
CREATE UNIQUE INDEX IF NOT EXISTS idx_citations_title_year ON citations(title, year);
CREATE UNIQUE INDEX IF NOT EXISTS idx_citations_doi ON citations(doi) WHERE doi IS NOT NULL AND doi != '';

-- Confidence ceiling: blog_post/forum_post/other sources cannot be "verified"
-- These are not authoritative sources; max confidence should be "high"
CREATE TRIGGER IF NOT EXISTS findings_confidence_check_insert
BEFORE INSERT ON findings
WHEN NEW.source_type IN ('forum_post','blog_post','other') AND NEW.confidence = 'verified'
BEGIN
    SELECT RAISE(ABORT, 'blog_post/forum_post/other sources cannot have verified confidence — use high instead');
END;

CREATE TRIGGER IF NOT EXISTS findings_confidence_check_update
BEFORE UPDATE ON findings
WHEN NEW.source_type IN ('forum_post','blog_post','other') AND NEW.confidence = 'verified'
BEGIN
    SELECT RAISE(ABORT, 'blog_post/forum_post/other sources cannot have verified confidence — use high instead');
END;

-- Seed the 11 skills
INSERT OR IGNORE INTO skills (id, name, layer, title, github_issue, description) VALUES
(1,  'gpu-silicon',      0, 'Apple GPU Microarchitecture',          1,  'Hardware internals: cores, SIMD groups, ALUs, registers, ISA, TBDR, M4/M5'),
(2,  'unified-memory',   0, 'Unified Memory Architecture',          2,  'Zero-copy, storage modes, SLC, virtual addressing, buffer management'),
(3,  'metal-compute',    1, 'Metal Compute Pipeline',               3,  'Device/Queue/Buffer/Encoder chain, dispatch, sync, language bindings'),
(4,  'msl-kernels',      1, 'Metal Shading Language for Compute',   4,  'Kernel syntax, address spaces, atomics, SIMD intrinsics, function constants'),
(5,  'gpu-io',           2, 'GPU Storage & I/O Patterns',           5,  'Metal Fast Resource Loading, mmap+GPU, async streaming, SSD path'),
(6,  'gpu-perf',         2, 'GPU Performance Engineering',          6,  'Profiling, occupancy, coalescing, divergence, kernel fusion'),
(7,  'simd-wave',        2, 'SIMD & Threadgroup Programming',       7,  'Reductions, scans, simdgroup_matrix, cooperative operations, sorting'),
(8,  'mlx-compute',      3, 'MLX Framework for GPU Computing',      8,  'Custom Metal kernels, lazy eval, streams, distributed, quantization'),
(9,  'metal4-api',       3, 'Metal 4 Next-Generation API',          9,  'Unified encoders, MTLTensor, cooperative tensors, explicit memory'),
(10, 'gpu-distributed',  3, 'Distributed GPU Computing',            10, 'RDMA/TB5, multi-Mac clusters, MLX distributed, topology'),
(11, 'gpu-centric-arch', 4, 'GPU-Centric Application Design',      11, 'Reverse offloading, persistent kernels, GPU databases/networking/FS');
