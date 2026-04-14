"""
PDF report generation — Clinical Sanctuary styling.

Three report types share a common style/helper module so the visual language
stays consistent across every PDF the app ever emits:

  1. Single screening report           (build_screening_pdf)
  2. Patient data export               (build_patient_export_pdf)
  3. Clinician patient summary         (build_patient_summary_pdf)

All returns are `BytesIO` objects ready to hand to `StreamingResponse`.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette (Clinical Sanctuary) ────────────────────────────────────────────────

INK = colors.HexColor("#1F2430")
INK_SOFT = colors.HexColor("#5A6170")
INK_FAINT = colors.HexColor("#8B909E")
CREAM = colors.HexColor("#F8F5EF")
CREAM_DARK = colors.HexColor("#EFE9DD")
TEAL = colors.HexColor("#2D6A6A")
CLAY = colors.HexColor("#C06F4D")
SAGE = colors.HexColor("#7B9E7D")
RUST = colors.HexColor("#B24634")
BORDER = colors.HexColor("#D9D2C3")


# ── Styles ──────────────────────────────────────────────────────────────────────


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}

    s["title"] = ParagraphStyle(
        name="title",
        parent=base["Heading1"],
        fontName="Times-Roman",
        fontSize=26,
        textColor=INK,
        leading=30,
        alignment=TA_LEFT,
        spaceAfter=2,
    )
    s["subtitle"] = ParagraphStyle(
        name="subtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=INK_SOFT,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=14,
    )
    s["h2"] = ParagraphStyle(
        name="h2",
        parent=base["Heading2"],
        fontName="Times-Bold",
        fontSize=14,
        textColor=TEAL,
        leading=18,
        spaceBefore=14,
        spaceAfter=6,
    )
    s["h3"] = ParagraphStyle(
        name="h3",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=INK,
        leading=14,
        spaceBefore=8,
        spaceAfter=3,
    )
    s["body"] = ParagraphStyle(
        name="body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        textColor=INK,
        leading=14,
        spaceAfter=6,
    )
    s["meta"] = ParagraphStyle(
        name="meta",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=INK_SOFT,
        leading=12,
    )
    s["quote"] = ParagraphStyle(
        name="quote",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=10.5,
        textColor=INK,
        leading=14,
        leftIndent=18,
        rightIndent=18,
        spaceBefore=4,
        spaceAfter=6,
    )
    s["crisis"] = ParagraphStyle(
        name="crisis",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        textColor=INK,
        leading=14,
        backColor=colors.HexColor("#F5E6DE"),
        borderPadding=8,
    )
    s["footer"] = ParagraphStyle(
        name="footer",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=INK_FAINT,
        leading=10,
        alignment=TA_CENTER,
    )
    return s


# ── Helpers ────────────────────────────────────────────────────────────────────


def _draw_header(canvas, _doc):
    """Letterhead at the top of every page."""
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, A4[1] - 1.1 * cm, A4[0], 1.1 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(1.8 * cm, A4[1] - 0.7 * cm, "DepScreen")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(
        A4[0] - 1.8 * cm,
        A4[1] - 0.7 * cm,
        "Clinical screening companion — Kingdom of Bahrain",
    )
    canvas.restoreState()


def _draw_footer(canvas, doc):
    """Footer with page number and generation timestamp."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_FAINT)
    canvas.drawCentredString(
        A4[0] / 2,
        0.8 * cm,
        f"Generated {datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC · Page {doc.page}",
    )
    canvas.restoreState()


def _page_decorations(canvas, doc):
    _draw_header(canvas, doc)
    _draw_footer(canvas, doc)


def _doc(buffer: BytesIO, title: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=title,
        author="DepScreen",
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.9 * cm,  # leaves room for the teal letterhead
        bottomMargin=1.5 * cm,
    )


def _hr(color=BORDER) -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5, color=color, spaceBefore=4, spaceAfter=8)


def _kv_table(rows: list[tuple[str, str]], s: dict[str, ParagraphStyle]) -> Table:
    """Two-column key/value table — labels muted, values ink."""
    data = []
    for k, v in rows:
        val = v if v not in (None, "") else "—"
        data.append([
            Paragraph(f"<font color='#5A6170'>{k}</font>", s["meta"]),
            Paragraph(str(val), s["body"]),
        ])
    t = Table(data, colWidths=[4.5 * cm, None])
    t.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, BORDER),
        ])
    )
    return t


def _safe(value: Any, fallback: str = "—") -> str:
    if value is None:
        return fallback
    if isinstance(value, str) and not value.strip():
        return fallback
    return str(value)


def _format_date(d: Any) -> str:
    if d is None:
        return "—"
    if isinstance(d, str):
        try:
            d = datetime.fromisoformat(d.replace("Z", "+00:00"))
        except ValueError:
            return d
    if isinstance(d, datetime):
        return d.strftime("%d/%m/%Y %H:%M")
    return str(d)


def _severity_color(severity: str) -> colors.Color:
    if severity == "severe":
        return RUST
    if severity == "moderate":
        return CLAY
    if severity == "mild":
        return TEAL
    return SAGE


CRISIS_FOOTER_HTML = (
    "<b>Support, any time</b><br/>"
    "Kingdom of Bahrain — if you or someone near you feels in immediate danger, "
    "<b>999</b> is the national emergency line (24/7). For confidential support: "
    "<b>Shamsaha 3844 7588 / 6671 0901</b>. Salmaniya Medical Complex's psychiatric "
    "emergency department is available any time."
)


# ── Report 1: Single screening ─────────────────────────────────────────────────


def build_screening_pdf(screening: dict, patient: dict) -> BytesIO:
    """Compact editorial screening report for a single session.

    Args:
        screening: dict with keys like id, created_at, severity_label,
                   severity_score, symptoms (list of {criterion, confidence}),
                   detected_sentences (list of str), llm_explanation,
                   flagged_for_review.
        patient: dict with keys like full_name, email, date_of_birth, cpr_number.
    """
    s = _styles()
    buf = BytesIO()
    doc = _doc(buf, f"Screening Report — {patient.get('full_name', 'Patient')}")
    story: list[Any] = []

    story.append(Paragraph("Screening report", s["title"]))
    story.append(
        Paragraph(
            f"{_safe(patient.get('full_name'))} · "
            f"Session {screening.get('id', '')[:8]} · "
            f"{_format_date(screening.get('created_at'))}",
            s["subtitle"],
        )
    )

    # ── Severity headline ────────
    severity = screening.get("severity_label") or "—"
    score = screening.get("severity_score")
    sev_color = _severity_color(severity)

    severity_text = (
        f"<font size='20' color='{sev_color.hexval()}'><b>{severity.title()}</b></font>"
        f"<br/><font size='9' color='#5A6170'>"
        f"Composite score {score if score is not None else '—'} · "
        "Based on DSM-5 criterion detection"
        "</font>"
    )
    story.append(Paragraph(severity_text, s["body"]))
    story.append(_hr())

    # ── Patient identifiers ────────
    story.append(Paragraph("Patient", s["h2"]))
    story.append(
        _kv_table(
            [
                ("Full name", _safe(patient.get("full_name"))),
                ("Email", _safe(patient.get("email"))),
                ("Date of birth", _format_date(patient.get("date_of_birth"))),
                ("CPR number", _safe(patient.get("cpr_number"))),
                ("MRN", _safe(patient.get("medical_record_number"))),
            ],
            s,
        )
    )

    # ── Symptoms detected ────────
    symptoms = screening.get("symptoms") or []
    story.append(Paragraph("Patterns noticed", s["h2"]))
    if not symptoms:
        story.append(Paragraph("No specific symptom patterns were detected.", s["body"]))
    else:
        data = [["Criterion", "Confidence"]]
        for sym in symptoms:
            conf = sym.get("confidence")
            conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
            data.append([sym.get("criterion", "—").replace("_", " ").title(), conf_str])
        t = Table(data, colWidths=[9 * cm, 4 * cm])
        t.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), CREAM_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, BORDER),
            ])
        )
        story.append(t)

    # ── Evidence ────────
    sentences = screening.get("detected_sentences") or []
    if sentences:
        story.append(Paragraph("Moments that stood out", s["h2"]))
        for sent in sentences[:10]:
            story.append(Paragraph(f"“{sent}”", s["quote"]))

    # ── LLM explanation ────────
    expl = screening.get("llm_explanation")
    if expl:
        story.append(Paragraph("Clinical interpretation", s["h2"]))
        story.append(Paragraph(_safe(expl).replace("\n", "<br/>"), s["body"]))

    # ── Crisis footer ────────
    if screening.get("flagged_for_review") or "suicidal" in " ".join(
        [(sy.get("criterion") or "").lower() for sy in symptoms]
    ):
        story.append(Spacer(1, 10))
        story.append(Paragraph(CRISIS_FOOTER_HTML, s["crisis"]))

    story.append(Spacer(1, 14))
    story.append(
        Paragraph(
            "This is a screening companion, not a medical diagnosis. "
            "It is one small window into how the patient is doing. A licensed "
            "clinician can see much more.",
            s["meta"],
        )
    )

    doc.build(story, onFirstPage=_page_decorations, onLaterPages=_page_decorations)
    buf.seek(0)
    return buf


# ── Report 2: Patient data export ──────────────────────────────────────────────


def build_patient_export_pdf(patient: dict, export: dict) -> BytesIO:
    """Full personal data export — every field the app stores about a patient.

    Args:
        patient: demographics dict
        export: {
            'screenings': [...],
            'medications': [...],
            'allergies': [...],
            'diagnoses': [...],
            'emergency_contacts': [...],
            'care_plans': [...],
            'documents': [...],
        }
    """
    s = _styles()
    buf = BytesIO()
    doc = _doc(buf, f"Personal Data Export — {patient.get('full_name', 'Patient')}")
    story: list[Any] = []

    story.append(Paragraph("Your DepScreen record", s["title"]))
    story.append(
        Paragraph(
            f"A complete export of what we have on file for "
            f"{_safe(patient.get('full_name'))}. Generated "
            f"{_format_date(datetime.utcnow())}.",
            s["subtitle"],
        )
    )

    # ── Profile ────────
    story.append(Paragraph("About you", s["h2"]))
    story.append(
        _kv_table(
            [
                ("Full name", _safe(patient.get("full_name"))),
                ("Email", _safe(patient.get("email"))),
                ("Phone", _safe(patient.get("phone"))),
                ("Date of birth", _format_date(patient.get("date_of_birth"))),
                ("Gender", _safe(patient.get("gender"))),
                ("Nationality", _safe(patient.get("nationality"))),
                ("CPR number", _safe(patient.get("cpr_number"))),
                ("MRN", _safe(patient.get("medical_record_number"))),
                ("Blood type", _safe(patient.get("blood_type"))),
                ("Preferred language", _safe(patient.get("language_preference"))),
                ("Timezone", _safe(patient.get("timezone"))),
            ],
            s,
        )
    )

    # ── Medications ────────
    meds = export.get("medications") or []
    story.append(Paragraph(f"Medications ({len(meds)})", s["h2"]))
    if not meds:
        story.append(Paragraph("No medications on record.", s["body"]))
    else:
        data = [["Name", "Dosage", "Frequency", "Prescribed by", "Started"]]
        for m in meds:
            data.append([
                _safe(m.get("name")),
                _safe(m.get("dosage")),
                _safe(m.get("frequency")),
                _safe(m.get("prescribed_by")),
                _format_date(m.get("start_date")),
            ])
        story.append(_grid(data, s))

    # ── Allergies ────────
    allergies = export.get("allergies") or []
    story.append(Paragraph(f"Allergies ({len(allergies)})", s["h2"]))
    if not allergies:
        story.append(Paragraph("No known allergies.", s["body"]))
    else:
        data = [["Allergen", "Type", "Severity", "Reaction"]]
        for a in allergies:
            data.append([
                _safe(a.get("allergen")),
                _safe(a.get("allergy_type")),
                _safe(a.get("severity")),
                _safe(a.get("reaction")),
            ])
        story.append(_grid(data, s))

    # ── Diagnoses ────────
    dxs = export.get("diagnoses") or []
    story.append(Paragraph(f"Diagnoses ({len(dxs)})", s["h2"]))
    if not dxs:
        story.append(Paragraph("No diagnoses on record.", s["body"]))
    else:
        data = [["Condition", "ICD-10", "Status", "Diagnosed", "By"]]
        for d in dxs:
            data.append([
                _safe(d.get("condition")),
                _safe(d.get("icd10_code")),
                _safe(d.get("status")),
                _format_date(d.get("diagnosed_date")),
                _safe(d.get("diagnosed_by")),
            ])
        story.append(_grid(data, s))

    # ── Emergency contacts ────────
    contacts = export.get("emergency_contacts") or []
    story.append(Paragraph(f"Emergency contacts ({len(contacts)})", s["h2"]))
    if not contacts:
        story.append(Paragraph("No emergency contacts on record.", s["body"]))
    else:
        data = [["Name", "Phone", "Relation", "Primary"]]
        for c in contacts:
            data.append([
                _safe(c.get("contact_name")),
                _safe(c.get("phone")),
                _safe(c.get("relation")),
                "Yes" if c.get("is_primary") else "—",
            ])
        story.append(_grid(data, s))

    # ── Screenings ────────
    screenings = export.get("screenings") or []
    story.append(Paragraph(f"Screening history ({len(screenings)})", s["h2"]))
    if not screenings:
        story.append(Paragraph("No screenings on record.", s["body"]))
    else:
        data = [["Date", "Severity", "Score", "Flagged"]]
        for sc in screenings:
            data.append([
                _format_date(sc.get("created_at")),
                _safe(sc.get("severity_label")).title(),
                _safe(sc.get("severity_score")),
                "Yes" if sc.get("flagged_for_review") else "—",
            ])
        story.append(_grid(data, s))

    # ── Documents ────────
    docs = export.get("documents") or []
    story.append(Paragraph(f"Uploaded documents ({len(docs)})", s["h2"]))
    if not docs:
        story.append(Paragraph("No documents uploaded.", s["body"]))
    else:
        for d in docs:
            story.append(
                Paragraph(
                    f"<b>{_safe(d.get('title'))}</b> "
                    f"<font color='#5A6170'>· {_safe(d.get('doc_type'))} · "
                    f"{_format_date(d.get('created_at'))}</font>",
                    s["body"],
                )
            )

    doc.build(story, onFirstPage=_page_decorations, onLaterPages=_page_decorations)
    buf.seek(0)
    return buf


# ── Report 3: Clinician patient summary ────────────────────────────────────────


def build_patient_summary_pdf(patient: dict, export: dict, clinician_name: str | None = None) -> BytesIO:
    """Clinician-facing summary optimised for clinical review."""
    s = _styles()
    buf = BytesIO()
    doc = _doc(buf, f"Patient Summary — {patient.get('full_name', 'Patient')}")
    story: list[Any] = []

    story.append(Paragraph("Patient clinical summary", s["title"]))
    story.append(
        Paragraph(
            f"{_safe(patient.get('full_name'))} · "
            f"CPR {_safe(patient.get('cpr_number'))} · "
            f"DOB {_format_date(patient.get('date_of_birth'))}"
            + (f"<br/>Prepared for {clinician_name}" if clinician_name else ""),
            s["subtitle"],
        )
    )

    # Active summary band
    active_meds = [m for m in (export.get("medications") or []) if m.get("is_active") is not False]
    active_dxs = [d for d in (export.get("diagnoses") or []) if d.get("status") == "active"]
    life_threatening = [
        a for a in (export.get("allergies") or [])
        if (a.get("severity") or "").lower() in {"life_threatening", "life-threatening", "severe"}
    ]
    latest_screenings = (export.get("screenings") or [])[:5]

    story.append(
        _kv_table(
            [
                ("Active medications", f"{len(active_meds)}"),
                ("Active diagnoses", f"{len(active_dxs)}"),
                ("High-severity allergies", f"{len(life_threatening)}"),
                ("Recent screenings", f"{len(latest_screenings)} in view"),
            ],
            s,
        )
    )

    # ── Demographics ────────
    story.append(Paragraph("Demographics", s["h2"]))
    story.append(
        _kv_table(
            [
                ("Full name", _safe(patient.get("full_name"))),
                ("Email", _safe(patient.get("email"))),
                ("Phone", _safe(patient.get("phone"))),
                ("Date of birth", _format_date(patient.get("date_of_birth"))),
                ("Gender", _safe(patient.get("gender"))),
                ("Nationality", _safe(patient.get("nationality"))),
                ("CPR", _safe(patient.get("cpr_number"))),
                ("MRN", _safe(patient.get("medical_record_number"))),
                ("Blood type", _safe(patient.get("blood_type"))),
            ],
            s,
        )
    )

    # ── Allergies (high severity first) ────────
    allergies = export.get("allergies") or []
    if allergies:
        story.append(Paragraph("Allergies", s["h2"]))
        data = [["Allergen", "Severity", "Reaction", "Notes"]]
        sort_key = {"life_threatening": 0, "life-threatening": 0, "severe": 1, "moderate": 2, "mild": 3}
        for a in sorted(allergies, key=lambda x: sort_key.get((x.get("severity") or "").lower(), 99)):
            data.append([
                _safe(a.get("allergen")),
                _safe(a.get("severity")),
                _safe(a.get("reaction")),
                _safe(a.get("notes")),
            ])
        story.append(_grid(data, s))

    # ── Medications ────────
    if active_meds:
        story.append(Paragraph("Active medications", s["h2"]))
        data = [["Name", "Dosage", "Frequency", "Started", "Prescribed by"]]
        for m in active_meds:
            data.append([
                _safe(m.get("name")),
                _safe(m.get("dosage")),
                _safe(m.get("frequency")),
                _format_date(m.get("start_date")),
                _safe(m.get("prescribed_by")),
            ])
        story.append(_grid(data, s))

    # ── Diagnoses ────────
    if active_dxs:
        story.append(Paragraph("Active diagnoses", s["h2"]))
        data = [["Condition", "ICD-10", "Diagnosed", "By"]]
        for d in active_dxs:
            data.append([
                _safe(d.get("condition")),
                _safe(d.get("icd10_code")),
                _format_date(d.get("diagnosed_date")),
                _safe(d.get("diagnosed_by")),
            ])
        story.append(_grid(data, s))

    # ── Screening trajectory ────────
    if latest_screenings:
        story.append(Paragraph("Recent screenings", s["h2"]))
        data = [["Date", "Severity", "Score", "Flagged", "Notes"]]
        for sc in latest_screenings:
            data.append([
                _format_date(sc.get("created_at")),
                _safe(sc.get("severity_label")).title(),
                _safe(sc.get("severity_score")),
                "Yes" if sc.get("flagged_for_review") else "—",
                _safe((sc.get("clinician_notes") or "")[:80]),
            ])
        story.append(_grid(data, s))

    # ── Care plan ────────
    care_plans = export.get("care_plans") or []
    active_cps = [cp for cp in care_plans if cp.get("status") in (None, "active", "review_needed")]
    if active_cps:
        story.append(Paragraph("Care plan", s["h2"]))
        for cp in active_cps[:2]:
            story.append(KeepTogether([
                Paragraph(f"<b>{_safe(cp.get('title'))}</b>", s["h3"]),
                Paragraph(
                    f"Status: <b>{_safe(cp.get('status'))}</b> · "
                    f"Review: {_format_date(cp.get('review_date'))}",
                    s["meta"],
                ),
                Paragraph(_safe(cp.get("description")), s["body"]),
            ]))

    # ── Emergency contact ────────
    contacts = export.get("emergency_contacts") or []
    primary = next((c for c in contacts if c.get("is_primary")), contacts[0] if contacts else None)
    if primary:
        story.append(Paragraph("Emergency contact (primary)", s["h2"]))
        story.append(
            _kv_table(
                [
                    ("Name", _safe(primary.get("contact_name"))),
                    ("Phone", _safe(primary.get("phone"))),
                    ("Relation", _safe(primary.get("relation"))),
                ],
                s,
            )
        )

    # Honest-limitations note
    story.append(Spacer(1, 10))
    story.append(Paragraph(CRISIS_FOOTER_HTML, s["crisis"]))

    doc.build(story, onFirstPage=_page_decorations, onLaterPages=_page_decorations)
    buf.seek(0)
    return buf


# ── Private helpers ────────────────────────────────────────────────────────────


def _grid(data: list[list[str]], s: dict[str, ParagraphStyle]) -> Table:
    """Clinical-style table: cream header, subtle row dividers."""
    # Wrap cells as Paragraphs so text wraps on long values
    wrapped = [[Paragraph(str(c), s["body"]) for c in row] for row in data]

    # Force header bolding
    header = [Paragraph(f"<b>{c}</b>", s["body"]) for c in data[0]]
    wrapped[0] = header

    col_count = len(data[0])
    col_widths = [None] * col_count  # let ReportLab auto-size

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CREAM_DARK),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, BORDER),
        ])
    )
    return t


__all__ = [
    "build_screening_pdf",
    "build_patient_export_pdf",
    "build_patient_summary_pdf",
]
