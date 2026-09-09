# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollups and Rich renderings for page trace artifacts.

Every number reported here is derived from ``amortized_ms`` rather than the raw
``duration_ms``. An operator span covers a whole batch and reports its exact
measured duration on each page in that batch, so summing durations would
multiply the batch cost by its page count. Amortized values divide that cost
across the pages it covered and therefore stay additive.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from rich.console import Console
from rich.table import Table

# Categories reported in the breakdown, in the order operators tend to spend
# time in them. The "operator" category is excluded: operator spans are the
# parents of these, so their total is the denominator, not a peer.
_CATEGORY_ORDER = ("network", "gpu", "cpu", "io")


def _fmt_ms(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number >= 1000:
        return f"{number / 1000:,.2f} s"
    return f"{number:,.1f} ms"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "-"


def _category_label(field: str) -> str:
    return field[:-3] if field.endswith("_ms") else field


def summarize(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build a cross-document rollup over *traces*.

    The returned mapping is what both the Rich tables and ``--json`` render,
    so the two views never disagree.
    """
    operator_ms: dict[str, float] = defaultdict(float)
    operator_self_ms: dict[str, float] = defaultdict(float)
    operator_calls: dict[str, int] = defaultdict(int)
    category_ms: dict[str, float] = defaultdict(float)
    model_ms: dict[str, float] = defaultdict(float)

    documents: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    models: dict[tuple[Any, ...], dict[str, Any]] = {}
    library_versions: dict[str, None] = {}
    run_modes: dict[str, None] = {}
    schema_versions: dict[str, None] = {}

    total_pages = 0
    for trace in traces:
        document = trace.get("document") or {}
        summary = trace.get("document_summary") or {}
        run = trace.get("run") or {}
        library = trace.get("nemo_retriever") or {}

        if library.get("full_version") or library.get("version"):
            library_versions.setdefault(str(library.get("full_version") or library.get("version")), None)
        if run.get("run_mode"):
            run_modes.setdefault(str(run["run_mode"]), None)
        if trace.get("schema_version"):
            schema_versions.setdefault(str(trace["schema_version"]), None)

        page_count = int(document.get("page_count") or 0)
        total_pages += page_count
        documents.append(
            {
                "document_id": document.get("document_id"),
                "source_path": document.get("source_path"),
                "source_type": document.get("source_type"),
                "page_count": page_count,
                "status": document.get("status"),
                "wall_ms": float(document.get("wall_ms") or 0.0),
                "total_ms": float(summary.get("total_ms") or 0.0),
                "span_count": int(summary.get("span_count") or 0),
            }
        )

        for entry in summary.get("by_operator") or []:
            name = str(entry.get("operator") or "unknown")
            operator_ms[name] += float(entry.get("total_ms") or 0.0)
            operator_self_ms[name] += float(entry.get("self_ms") or 0.0)
            operator_calls[name] += int(entry.get("calls") or 0)
        for field, value in (summary.get("by_category") or {}).items():
            category_ms[_category_label(str(field))] += float(value or 0.0)
        for key, value in (summary.get("by_model") or {}).items():
            model_ms[str(key)] += float(value or 0.0)

        for descriptor in trace.get("models") or []:
            identity = (
                descriptor.get("model_key"),
                descriptor.get("name"),
                descriptor.get("version"),
                descriptor.get("backend"),
                descriptor.get("endpoint"),
            )
            models.setdefault(identity, dict(descriptor))

        for page in trace.get("page_summaries") or []:
            by_operator = page.get("by_operator") or {}
            slowest = max(by_operator.items(), key=lambda item: item[1], default=(None, 0.0))
            pages.append(
                {
                    "document_id": document.get("document_id"),
                    "page_number": page.get("page_number"),
                    "source_id": page.get("source_id"),
                    "total_ms": float(page.get("total_ms") or 0.0),
                    "wall_ms": float(page.get("wall_ms") or 0.0),
                    "span_count": int(page.get("span_count") or 0),
                    "slowest_operator": slowest[0],
                    "slowest_operator_ms": float(slowest[1] or 0.0),
                }
            )

    grand_total_ms = sum(operator_ms.values())
    by_operator = [
        {
            "operator": name,
            "calls": operator_calls[name],
            "total_ms": round(total, 3),
            "self_ms": round(operator_self_ms[name], 3),
            "ms_per_page": round(total / total_pages, 3) if total_pages else 0.0,
            "pct_of_total": round(100.0 * total / grand_total_ms, 2) if grand_total_ms else 0.0,
        }
        for name, total in sorted(operator_ms.items(), key=lambda item: -item[1])
    ]
    # Shares are against the operator total, so a category reads as "this much
    # of measured pipeline time went to network / GPU / CPU / I/O".
    by_category = [
        {
            "category": name,
            "total_ms": round(total, 3),
            "pct_of_total": round(100.0 * total / grand_total_ms, 2) if grand_total_ms else 0.0,
        }
        for name, total in sorted(
            ((name, value) for name, value in category_ms.items() if name != "operator"),
            key=lambda item: (
                _CATEGORY_ORDER.index(item[0]) if item[0] in _CATEGORY_ORDER else 99,
                item[0],
            ),
        )
    ]

    pages.sort(key=lambda page: -page["total_ms"])
    documents.sort(key=lambda document: -document["total_ms"])

    return {
        "schema_versions": list(schema_versions),
        "library_versions": list(library_versions),
        "run_modes": list(run_modes),
        "document_count": len(documents),
        "page_count": total_pages,
        "total_ms": round(grand_total_ms, 3),
        "ms_per_page": round(grand_total_ms / total_pages, 3) if total_pages else 0.0,
        "documents": documents,
        "by_operator": by_operator,
        "by_category": by_category,
        "by_model": [
            {"model_key": key, "total_ms": round(value, 3)}
            for key, value in sorted(model_ms.items(), key=lambda item: -item[1])
        ],
        "models": list(models.values()),
        "slowest_pages": pages,
    }


def render_summary(console: Console, summary: dict[str, Any], *, top: int) -> None:
    """Print the cross-document summary tables."""
    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Documents", str(summary["document_count"]))
    header.add_row("Pages", str(summary["page_count"]))
    header.add_row("Traced time", _fmt_ms(summary["total_ms"]))
    header.add_row("Per page", _fmt_ms(summary["ms_per_page"]))
    if summary["library_versions"]:
        header.add_row("nemo-retriever", ", ".join(summary["library_versions"]))
    if summary["run_modes"]:
        header.add_row("Run mode", ", ".join(summary["run_modes"]))
    console.print(header)

    operator_table = Table(title="Time by operator", title_justify="left", header_style="bold")
    operator_table.add_column("Operator")
    operator_table.add_column("Calls", justify="right")
    operator_table.add_column("Total", justify="right")
    operator_table.add_column("Self", justify="right")
    operator_table.add_column("Per page", justify="right")
    operator_table.add_column("Share", justify="right")
    for entry in summary["by_operator"][:top]:
        operator_table.add_row(
            str(entry["operator"]),
            str(entry["calls"]),
            _fmt_ms(entry["total_ms"]),
            _fmt_ms(entry["self_ms"]),
            _fmt_ms(entry["ms_per_page"]),
            _fmt_pct(entry["pct_of_total"]),
        )
    if not summary["by_operator"]:
        operator_table.add_row("(no operator spans)", "-", "-", "-", "-", "-")
    console.print(operator_table)

    category_table = Table(
        title="Time by category (share of traced time)", title_justify="left", header_style="bold"
    )
    category_table.add_column("Category")
    category_table.add_column("Total", justify="right")
    category_table.add_column("Share", justify="right")
    for entry in summary["by_category"]:
        category_table.add_row(str(entry["category"]), _fmt_ms(entry["total_ms"]), _fmt_pct(entry["pct_of_total"]))
    if not summary["by_category"]:
        category_table.add_row("(none recorded)", "-", "-")
    console.print(category_table)
    if not any(entry["total_ms"] for entry in summary["by_category"]):
        console.print(
            "[dim]Only operator-level spans were recorded. "
            "Re-run with --page-trace-detail full for network, GPU, and I/O detail.[/dim]"
        )

    if summary["models"]:
        model_table = Table(title="Models", title_justify="left", header_style="bold")
        model_table.add_column("Key")
        model_table.add_column("Name")
        model_table.add_column("Version")
        model_table.add_column("Backend")
        model_table.add_column("Time", justify="right")
        model_times = {entry["model_key"]: entry["total_ms"] for entry in summary["by_model"]}
        for descriptor in summary["models"]:
            key = str(descriptor.get("model_key") or "-")
            model_table.add_row(
                key,
                str(descriptor.get("name") or "-"),
                str(descriptor.get("version") or "-"),
                str(descriptor.get("backend") or "-"),
                _fmt_ms(model_times.get(key)) if key in model_times else "-",
            )
        console.print(model_table)

    if summary["document_count"] > 1:
        document_table = Table(title="Documents", title_justify="left", header_style="bold")
        document_table.add_column("Document")
        document_table.add_column("Pages", justify="right")
        document_table.add_column("Traced", justify="right")
        document_table.add_column("Wall", justify="right")
        document_table.add_column("Status")
        for document in summary["documents"][:top]:
            document_table.add_row(
                str(document["document_id"]),
                str(document["page_count"]),
                _fmt_ms(document["total_ms"]),
                _fmt_ms(document["wall_ms"]),
                str(document["status"] or "-"),
            )
        console.print(document_table)

    page_table = Table(title=f"Slowest pages (top {top})", title_justify="left", header_style="bold")
    page_table.add_column("Document")
    page_table.add_column("Page", justify="right")
    page_table.add_column("Traced", justify="right")
    page_table.add_column("Wall", justify="right")
    page_table.add_column("Spans", justify="right")
    page_table.add_column("Slowest operator")
    for page in summary["slowest_pages"][:top]:
        slowest = page["slowest_operator"]
        page_table.add_row(
            str(page["document_id"]),
            str(page["page_number"]),
            _fmt_ms(page["total_ms"]),
            _fmt_ms(page["wall_ms"]),
            str(page["span_count"]),
            f"{slowest} ({_fmt_ms(page['slowest_operator_ms'])})" if slowest else "-",
        )
    if not summary["slowest_pages"]:
        page_table.add_row("(no pages traced)", "-", "-", "-", "-", "-")
    console.print(page_table)


def page_detail(trace: dict[str, Any], page_number: int) -> dict[str, Any]:
    """Extract one page's summary and span waterfall from *trace*."""
    document = trace.get("document") or {}
    summary = next(
        (entry for entry in trace.get("page_summaries") or [] if int(entry.get("page_number") or -1) == page_number),
        None,
    )
    if summary is None:
        available = sorted(int(entry.get("page_number") or 0) for entry in trace.get("page_summaries") or [])
        raise KeyError(
            f"Page {page_number} is not present in {document.get('document_id') or 'this trace'}. "
            f"Traced pages: {available or 'none'}."
        )

    spans = [span for span in trace.get("spans") or [] if int(span.get("page_number") or -1) == page_number]
    return {
        "document": document,
        "page": summary,
        "spans": _order_waterfall(spans),
    }


def _order_waterfall(spans: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *spans* in parent-before-child order with a ``depth`` field.

    Spans whose parent lives on a different page (a batch-level span parented to
    an operator that touched other pages too) are treated as roots so nothing is
    dropped from the listing.
    """
    by_id = {str(span.get("span_id")): span for span in spans}
    children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        parent = span.get("parent_span_id")
        key = str(parent) if parent and str(parent) in by_id else None
        children[key].append(span)
    for bucket in children.values():
        bucket.sort(key=lambda span: (float(span.get("start_ms") or 0.0), str(span.get("name") or "")))

    ordered: list[dict[str, Any]] = []

    def _walk(parent_key: str | None, depth: int) -> None:
        for span in children.get(parent_key, ()):
            record = dict(span)
            record["depth"] = depth
            ordered.append(record)
            _walk(str(span.get("span_id")), depth + 1)

    _walk(None, 0)
    return ordered


def render_page(console: Console, detail: dict[str, Any]) -> None:
    """Print one page's span waterfall."""
    document = detail["document"]
    page = detail["page"]

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Document", str(document.get("document_id") or "-"))
    header.add_row("Source", str(document.get("source_path") or "-"))
    header.add_row("Page", str(page.get("page_number")))
    header.add_row("Traced time", _fmt_ms(page.get("total_ms")))
    header.add_row("Wall time", _fmt_ms(page.get("wall_ms")))
    header.add_row("Spans", str(page.get("span_count")))
    console.print(header)

    category_table = Table(title="Page time by category", title_justify="left", header_style="bold")
    category_table.add_column("Category")
    category_table.add_column("Total", justify="right")
    for field, value in (page.get("by_category") or {}).items():
        label = _category_label(str(field))
        if value and label != "operator":
            category_table.add_row(label, _fmt_ms(value))
    if not category_table.rows:
        category_table.add_row("(none recorded)", "-")
    console.print(category_table)

    waterfall = Table(title="Span waterfall", title_justify="left", header_style="bold")
    waterfall.add_column("Span")
    waterfall.add_column("Category")
    waterfall.add_column("Duration", justify="right")
    waterfall.add_column("Amortized", justify="right")
    waterfall.add_column("Batch", justify="right")
    waterfall.add_column("Detail")
    for span in detail["spans"]:
        indent = "  " * int(span.get("depth") or 0)
        status = str(span.get("status") or "ok")
        name = f"{indent}{span.get('name')}"
        if status != "ok":
            name = f"[red]{name}[/red]"
        waterfall.add_row(
            name,
            str(span.get("category") or "-"),
            _fmt_ms(span.get("duration_ms")),
            _fmt_ms(span.get("amortized_ms")),
            str(span.get("batch_size") or "-"),
            _span_detail(span),
        )
    if not detail["spans"]:
        waterfall.add_row("(no spans recorded for this page)", "-", "-", "-", "-", "-")
    console.print(waterfall)


def _span_detail(span: dict[str, Any]) -> str:
    """Render the most useful attributes of a span as a compact string."""
    parts: list[str] = []
    if span.get("model_key"):
        parts.append(f"model={span['model_key']}")
    attrs = span.get("attrs") or {}
    for key in ("model", "endpoint", "protocol", "http_status", "calls", "device"):
        value = attrs.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    if span.get("error"):
        parts.append(f"error={span['error']}")
    return ", ".join(parts) or "-"


def resolve_trace_for_document(traces: Sequence[dict[str, Any]], document_id: str | None) -> dict[str, Any]:
    """Pick the trace to inspect, requiring a document id when several loaded."""
    if document_id:
        for trace in traces:
            document = trace.get("document") or {}
            if str(document.get("document_id")) == document_id or str(document.get("source_path")) == document_id:
                return trace
        known = ", ".join(str((trace.get("document") or {}).get("document_id")) for trace in traces)
        raise KeyError(f"No loaded trace matches document '{document_id}'. Loaded: {known}.")
    if len(traces) == 1:
        return traces[0]
    known = ", ".join(str((trace.get("document") or {}).get("document_id")) for trace in traces)
    raise KeyError(f"{len(traces)} traces loaded; pass --document to choose one. Loaded: {known}.")


def iter_document_ids(traces: Iterable[dict[str, Any]]) -> list[str]:
    return [str((trace.get("document") or {}).get("document_id")) for trace in traces]
