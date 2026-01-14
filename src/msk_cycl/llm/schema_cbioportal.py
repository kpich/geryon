"""cBioPortal-specific schema formatting for LLM context.

Handles cBioPortal naming conventions and reduces verbosity for large schemas.
Future data sources (TCGA, UK Biobank, etc.) should have their own formatters.
"""

from msk_cycl.llm.schema import ColumnInfo, DatabaseSchema, TableInfo


def schema_to_cbioportal_context(
    schema: DatabaseSchema,
    max_columns_per_table: int = 10,
) -> str:
    """Format cBioPortal schema as concise markdown for LLM.

    Handles cBioPortal-specific conventions:
    - Genomic tables: CNA (wide gene table), gene_matrix, mutations
    - Clinical tables: clinical_patient, clinical_sample
    - Timeline tables: timeline_* (50+ tables)
    - Meta tables: meta_* (skip entirely)

    Parameters
    ----------
    schema : DatabaseSchema
        Database schema from discover_schema()
    max_columns_per_table : int
        Maximum columns to show per table (default: 10)

    Returns
    -------
    str
        Markdown-formatted schema description
    """
    lines = []

    categorized = _categorize_cbioportal_tables(schema.tables)

    if categorized["genomic"]:
        lines.append("# Genomic Data")
        lines.append("")
        for table in categorized["genomic"]:
            lines.extend(_format_cbioportal_table(table, max_columns_per_table))

    if categorized["clinical"]:
        lines.append("# Clinical Data")
        lines.append("")
        for table in categorized["clinical"]:
            lines.extend(_format_cbioportal_table(table, max_columns_per_table))

    if categorized["timeline"]:
        lines.append("# Timeline Data")
        lines.append("")
        lines.append(
            f"{len(categorized['timeline'])} timeline tables available "
            "(timeline_bmi, timeline_ecog_kps, timeline_treatment, etc.)"
        )
        lines.append(
            "- Each has: PATIENT_ID, START_DATE, STOP_DATE, plus event-specific columns"
        )
        lines.append("")

    if categorized["other"]:
        lines.append("# Other Tables")
        lines.append("")
        for table in categorized["other"]:
            lines.extend(_format_cbioportal_table(table, max_columns_per_table))

    return "\n".join(lines)


def _categorize_cbioportal_tables(
    tables: list[TableInfo],
) -> dict[str, list[TableInfo]]:
    """Group tables by cBioPortal category."""
    genomic = []
    clinical = []
    timeline = []
    other = []

    for table in tables:
        if table.name.startswith("meta_"):
            continue
        elif table.name in [
            "CNA",
            "gene_matrix",
            "mutations_extended",
            "nonsignedout_mutations",
        ]:
            genomic.append(table)
        elif table.name.startswith("clinical_"):
            clinical.append(table)
        elif table.name.startswith("timeline_"):
            timeline.append(table)
        else:
            other.append(table)

    return {
        "genomic": genomic,
        "clinical": clinical,
        "timeline": timeline,
        "other": other,
    }


def _format_cbioportal_table(table: TableInfo, max_columns: int) -> list[str]:
    """Format single table as concise summary.

    Special handling for wide tables (>50 columns like CNA).
    """
    if table.name == "CNA":
        return _format_cbioportal_cna_table(table)

    if len(table.columns) > 50:
        return _format_wide_table(table)

    return _format_regular_table(table, max_columns)


def _format_cbioportal_cna_table(table: TableInfo) -> list[str]:
    """Special formatting for CNA table (706 gene columns)."""
    lines = []
    lines.append(
        f"## Table: {table.name} "
        f"({table.row_count:,} rows, {len(table.columns)} columns)"
    )
    lines.append("")
    lines.append(
        "**Wide genomic table**: 706 gene columns (CNA values: -2, -1, 0, 1, 2)"
    )
    lines.append("- Column names are gene symbols (KRAS, TP53, EGFR, etc.)")
    lines.append("- Use for filtering patients by copy number alteration status")
    lines.append(
        "- Example filter: `table: 'CNA', column: 'KRAS', operator: '==', value: 1`"
    )
    lines.append("")

    lines.append("| Column | Type | Description |")
    lines.append("|--------|------|-------------|")
    lines.append("| PATIENT_ID | VARCHAR | Patient identifier |")

    example_genes = [col for col in table.columns if col.name != "PATIENT_ID"][:3]
    for gene in example_genes:
        lines.append(f"| {gene.name} | DOUBLE | CNA value for {gene.name} gene |")

    lines.append(f"| ... | ... | {len(table.columns) - 4} more gene columns |")
    lines.append("")

    return lines


def _format_wide_table(table: TableInfo) -> list[str]:
    """Format tables with >50 columns as summary."""
    lines = []
    lines.append(
        f"## Table: {table.name} "
        f"({table.row_count:,} rows, {len(table.columns)} columns)"
    )
    lines.append("")
    lines.append(f"**Wide table**: {len(table.columns)} columns")

    key_cols = [col for col in table.columns if col.name == "PATIENT_ID"]
    example_cols = [col for col in table.columns if col.name != "PATIENT_ID"][:3]

    lines.append("")
    lines.append("| Column | Type | Sample Values |")
    lines.append("|--------|------|---------------|")

    for col in key_cols + example_cols:
        sample_str = _format_samples(col.sample_values)
        lines.append(f"| {col.name} | {col.type} | {sample_str} |")

    more_cols = len(table.columns) - len(key_cols) - len(example_cols)
    lines.append(f"| ... | ... | {more_cols} more columns |")
    lines.append("")

    return lines


def _format_regular_table(table: TableInfo, max_columns: int) -> list[str]:
    """Format regular table with column filtering."""
    lines = []
    lines.append(f"## Table: {table.name} ({table.row_count:,} rows)")
    lines.append("")

    interesting_cols = _select_interesting_columns(table.columns, max_columns)

    if len(interesting_cols) < len(table.columns):
        lines.append(
            f"Showing {len(interesting_cols)} of {len(table.columns)} columns:"
        )
        lines.append("")

    lines.append("| Column | Type | Sample Values | Distinct |")
    lines.append("|--------|------|---------------|----------|")

    for col in interesting_cols:
        sample_str = _format_samples(col.sample_values)
        distinct_str = str(col.distinct_count) if col.distinct_count else ""
        lines.append(f"| {col.name} | {col.type} | {sample_str} | {distinct_str} |")

    lines.append("")
    return lines


def _select_interesting_columns(
    columns: list[ColumnInfo], max_cols: int
) -> list[ColumnInfo]:
    """Select most interesting columns for LLM context.

    Prioritizes:
    1. Key columns (PATIENT_ID, SAMPLE_ID, survival columns)
    2. Categorical columns with sample values and <100 distinct values
    """
    priority_names = {"PATIENT_ID", "SAMPLE_ID", "OS_MONTHS", "OS_STATUS"}
    priority_cols = [col for col in columns if col.name in priority_names]

    interesting = [
        col
        for col in columns
        if col not in priority_cols
        and len(col.sample_values) > 0
        and (col.distinct_count is None or col.distinct_count < 100)
    ]

    interesting.sort(
        key=lambda c: c.distinct_count if c.distinct_count else float("inf")
    )

    selected = priority_cols + interesting[: max_cols - len(priority_cols)]

    return selected


def _format_samples(sample_values: list[str | int | float]) -> str:
    """Format sample values concisely (max 3 samples)."""
    if not sample_values:
        return ""

    samples = [str(v) for v in sample_values[:3]]
    sample_str = ", ".join(f'"{s}"' if " " in s else s for s in samples)

    if len(sample_values) > 3:
        sample_str += ", ..."

    return sample_str
