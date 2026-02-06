from __future__ import annotations

from typing import List

from .inference import EntitySpan


def color_for_label(label: str) -> str:
    palette = [
        "#FFCDD2", "#F8BBD0", "#E1BEE7", "#D1C4E9", "#C5CAE9",
        "#BBDEFB", "#B3E5FC", "#B2EBF2", "#B2DFDB", "#C8E6C9",
        "#DCEDC8", "#F0F4C3", "#FFF9C4", "#FFECB3", "#FFE0B2",
        "#FFCCBC", "#D7CCC8", "#CFD8DC"
    ]
    return palette[hash(label) % len(palette)]


def highlight_spans(text: str, spans: List[EntitySpan]) -> str:
    if not spans:
        return text
    spans = sorted(spans, key=lambda s: s.start)
    out = []
    last = 0
    for s in spans:
        out.append(text[last:s.start])
        color = color_for_label(s.label)
        out.append(f'<span style="background-color:{color}; padding:0 2px; border-radius:3px;">{text[s.start:s.end]}<sub style="font-size:0.7em; color:#333;">{s.label}</sub></span>')
        last = s.end
    out.append(text[last:])
    return "".join(out)
