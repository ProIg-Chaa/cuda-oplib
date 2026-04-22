from typing import Iterable


MAX_NOTES_WIDTH = 36
RIGHT_ALIGNED_HEADERS = {"max_abs_diff", "avg_ms", "speedup_vs_ref"}


def _truncate(value: object, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


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
            cell = str(row.get(header, ""))
            if header == "notes":
                cell = _truncate(cell, MAX_NOTES_WIDTH)
            widths[header] = max(widths[header], len(cell))

    widths["notes"] = min(widths["notes"], MAX_NOTES_WIDTH)

    def fmt_line(values):
        formatted = []
        for header in headers:
            cell = values[header]
            if header == "notes":
                cell = _truncate(cell, widths[header])
            else:
                cell = str(cell)

            if header in RIGHT_ALIGNED_HEADERS:
                formatted.append(cell.rjust(widths[header]))
            else:
                formatted.append(cell.ljust(widths[header]))
        return " | ".join(formatted)

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
