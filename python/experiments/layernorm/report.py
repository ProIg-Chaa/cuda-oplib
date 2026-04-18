from typing import Iterable


def render_terminal_table(rows: Iterable[dict]) -> str:
    rows = list(rows)
    if not rows:
        return "No results."

    headers = [
        "name",
        "stage",
        "dtype",
        "correct",
        "max_abs_diff",
        "avg_ms",
        "speedup_vs_ref",
        "notes",
    ]

    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def fmt_line(values):
        return "  ".join(str(values[h]).ljust(widths[h]) for h in headers)

    out = [fmt_line({h: h for h in headers})]
    out.append(fmt_line({h: "-" * widths[h] for h in headers}))
    for row in rows:
        out.append(fmt_line({h: row.get(h, "") for h in headers}))
    return "\n".join(out)


def render_markdown_table(rows: Iterable[dict]) -> str:
    rows = list(rows)
    if not rows:
        return "No results."

    headers = [
        "name",
        "stage",
        "dtype",
        "correct",
        "max_abs_diff",
        "avg_ms",
        "speedup_vs_ref",
        "notes",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)

