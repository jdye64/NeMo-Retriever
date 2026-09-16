# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained HTML gallery for a local Retriever index."""

from __future__ import annotations

import html
import json

from nemo_retriever.inspect.summary import IndexSummary

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #0b0f14;
      --panel: #151b23;
      --panel-2: #1c2430;
      --text: #e8eef5;
      --muted: #9aa8b8;
      --line: #2a3544;
      --accent: #76b900;
      --accent-2: #a4d65e;
      --chip: #243018;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "NVIDIA Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(1200px 500px at 10% -10%, rgba(118, 185, 0, 0.18), transparent 55%),
        var(--bg);
      color: var(--text);
    }
    .visually-hidden {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
    }
    header.page {
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 1.25rem 1.5rem 1rem;
      background: color-mix(in srgb, var(--bg) 88%, transparent);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0 0 0.35rem; font-size: 1.35rem; font-weight: 650; }
    h2 { margin: 0; font-size: 1.05rem; }
    .sub { color: var(--muted); font-size: 0.95rem; }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
      margin-top: 0.9rem;
    }
    .stat, .chip {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.35rem 0.75rem;
      font-size: 0.85rem;
    }
    .stat strong { color: var(--accent-2); }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
      margin-top: 1rem;
    }
    input[type="search"] {
      flex: 1 1 18rem;
      min-width: 12rem;
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 0.7rem;
      padding: 0.7rem 0.9rem;
      font-size: 1rem;
    }
    input[type="search"]:focus { outline: 2px solid var(--accent); }
    .filters { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    button.chip {
      cursor: pointer;
      color: var(--text);
      background: var(--chip);
    }
    button.chip[aria-pressed="true"] {
      background: var(--accent);
      color: #10140c;
      border-color: var(--accent);
    }
    main { padding: 1.25rem 1.5rem 3rem; max-width: 1100px; margin: 0 auto; }
    .empty {
      margin-top: 2rem;
      padding: 1.5rem;
      border: 1px dashed var(--line);
      border-radius: 1rem;
      color: var(--muted);
    }
    .doc {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 1rem;
      margin: 1rem 0;
      overflow: hidden;
    }
    .doc .doc-head {
      padding: 1rem 1.1rem 0.7rem;
    }
    .meta { color: var(--muted); font-size: 0.88rem; margin-top: 0.25rem; }
    .chunk {
      margin: 0 1rem 1rem;
      padding: 0.9rem 1rem;
      background: var(--panel-2);
      border-radius: 0.8rem;
      border: 1px solid var(--line);
    }
    .chunk .kicker {
      display: flex;
      gap: 0.5rem;
      align-items: center;
      margin-bottom: 0.45rem;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .chunk p { margin: 0; line-height: 1.45; white-space: pre-wrap; }
    mark {
      background: color-mix(in srgb, var(--accent) 45%, transparent);
      color: inherit;
      padding: 0 0.1em;
      border-radius: 0.2em;
    }
  </style>
</head>
<body>
  <header class="page">
    <h1>What did ingest store?</h1>
    <div class="sub" id="subtitle"></div>
    <div class="stats" id="stats"></div>
    <div class="toolbar">
      <label class="visually-hidden" for="q">Search extracted text</label>
      <input id="q" type="search" placeholder="Search extracted text, files, or types" autocomplete="off">
      <div class="filters" id="filters" role="group" aria-label="Content types"></div>
    </div>
  </header>
  <main>
    <div id="gallery"></div>
    <p class="empty" id="empty" hidden>No chunks match this search.</p>
  </main>
  <script id="index-data" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("index-data").textContent);
    const gallery = document.getElementById("gallery");
    const empty = document.getElementById("empty");
    const search = document.getElementById("q");
    const filters = document.getElementById("filters");
    const stats = document.getElementById("stats");
    let activeType = "all";

    document.getElementById("subtitle").textContent =
      data.table + " @ " + data.uri +
      (data.embeddingModel ? " · " + data.embeddingModel : "") +
      " · " + data.retrievalMode + " retrieval";

    const statBits = [
      ["Rows", data.rowCount],
      ["Documents", data.documentCount],
      ["Pages", data.pageCount],
      ["Empty text", data.emptyTextCount],
      ["Vector", data.hasVector ? "yes" : "no"],
      ["FTS", data.hasFts ? "yes" : "no"],
    ];
    stats.innerHTML = statBits.map(([label, value]) =>
      `<span class="stat">${label} <strong>${value}</strong></span>`
    ).join("");

    const types = ["all", ...Object.keys(data.contentTypes || {})];
    filters.innerHTML = types.map((type) => {
      const label = type === "all" ? "all" : `${type} ${data.contentTypes[type]}`;
      return `<button class="chip" type="button" data-type="${type}" aria-pressed="${type === "all"}">${label}</button>`;
    }).join("");

    function escapeHtml(text) {
      return String(text).replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
    }

    function highlight(text, query) {
      const escaped = escapeHtml(text);
      if (!query) return escaped;
      const needle = escapeHtml(query);
      const lower = escaped.toLowerCase();
      const q = needle.toLowerCase();
      let out = "";
      let cursor = 0;
      while (q) {
        const found = lower.indexOf(q, cursor);
        if (found < 0) break;
        out += escaped.slice(cursor, found) + "<mark>" + escaped.slice(found, found + q.length) + "</mark>";
        cursor = found + q.length;
      }
      return out + escaped.slice(cursor);
    }

    function render() {
      const query = search.value.trim().toLowerCase();
      const grouped = {};
      for (const chunk of data.previews) {
        if (activeType !== "all" && chunk.contentType !== activeType) continue;
        const hay = (chunk.text + " " + chunk.filename + " " + chunk.contentType).toLowerCase();
        if (query && !hay.includes(query)) continue;
        (grouped[chunk.source] ||= []).push(chunk);
      }
      const sources = Object.keys(grouped);
      empty.hidden = sources.length > 0;
      gallery.innerHTML = sources.map((source) => {
        const chunks = grouped[source];
        const filename = chunks[0].filename;
        const doc = (data.documents || []).find((item) => item.source === source);
        const meta = doc
          ? `${doc.pageCount} pages · ${doc.chunkCount} chunks`
          : `${chunks.length} previewed chunks`;
        const cards = chunks.map((chunk) => `
          <article class="chunk">
            <div class="kicker">
              <span>${escapeHtml(chunk.contentType)}</span>
              <span>page ${chunk.pageNumber == null ? "?" : chunk.pageNumber}</span>
            </div>
            <p>${highlight(chunk.text || "(empty)", query)}</p>
          </article>`).join("");
        return `<section class="doc">
          <div class="doc-head">
            <h2>${highlight(filename, query)}</h2>
            <div class="meta">${meta}</div>
          </div>
          ${cards}
        </section>`;
      }).join("");
    }

    filters.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-type]");
      if (!button) return;
      activeType = button.dataset.type;
      for (const node of filters.querySelectorAll("button")) {
        node.setAttribute("aria-pressed", String(node === button));
      }
      render();
    });
    search.addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""


def render_index_html(summary: IndexSummary) -> str:
    """Return a standalone HTML page for browsing ``summary``."""
    payload = json.dumps(
        {
            "table": summary.table_name,
            "uri": summary.lancedb_uri,
            "rowCount": summary.row_count,
            "documentCount": summary.document_count,
            "pageCount": summary.page_count,
            "retrievalMode": summary.retrieval_mode,
            "embeddingModel": summary.embedding_model_name,
            "hasVector": summary.has_vector,
            "hasFts": summary.has_fts,
            "emptyTextCount": summary.empty_text_count,
            "contentTypes": summary.content_types,
            "documents": [
                {
                    "source": document.source,
                    "filename": document.filename,
                    "pageCount": document.page_count,
                    "chunkCount": document.chunk_count,
                    "contentTypes": document.content_types,
                }
                for document in summary.documents
            ],
            "previews": [
                {
                    "source": preview.source,
                    "filename": preview.filename,
                    "pageNumber": preview.page_number,
                    "contentType": preview.content_type,
                    "text": preview.text,
                }
                for preview in summary.previews
            ],
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__TITLE__", html.escape(f"Retriever inspect · {summary.table_name}")).replace(
        "__PAYLOAD__", payload
    )
