#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.data_dir    = "${projectDir}/../geryon_data"
params.output_dir  = "${projectDir}/../plots"
params.parquet_dir = ""  // required for pvalComparison; skipped if empty

process ratingsOverTime {
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    output:
    path "ratings_over_time.pdf"

    script:
    """
    uv run python -m geryon.plot.ratings_over_time \
        --data-dir ${params.data_dir} \
        --output ratings_over_time.pdf
    """
}

process ratingsByDepth {
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    output:
    path "ratings_by_depth.pdf"

    script:
    """
    uv run python -m geryon.plot.ratings_by_depth \
        --data-dir ${params.data_dir} \
        --output ratings_by_depth.pdf
    """
}

process scoreByDepth {
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    output:
    path "score_by_depth.pdf"

    script:
    """
    uv run python -m geryon.plot.score_by_depth \
        --data-dir ${params.data_dir} \
        --output score_by_depth.pdf
    """
}

process scoreOverTime {
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    output:
    path "score_over_time.pdf"

    script:
    """
    uv run python -m geryon.plot.score_over_time \
        --data-dir ${params.data_dir} \
        --output score_over_time.pdf
    """
}

process executeAgainstVal {
    errorStrategy 'terminate'

    output:
    path "val_results.csv"

    script:
    """
    uv run python -m geryon.eval.batch \
        --data-dir ${params.data_dir} \
        --parquet-dir ${params.parquet_dir} \
        --output val_results.csv
    """
}

process pvalComparison {
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path val_results

    output:
    path "pval_comparison.pdf"

    script:
    """
    uv run python -m geryon.plot.pval_comparison \
        --input ${val_results} \
        --output pval_comparison.pdf
    """
}

process qvalComparison {
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path val_results

    output:
    path "qval_comparison.pdf"

    script:
    """
    uv run python -m geryon.plot.qval_comparison \
        --input ${val_results} \
        --output qval_comparison.pdf
    """
}

process topValHypotheses {
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path val_results

    output:
    path "top_val_hypotheses.txt"

    script:
    """
    uv run python -m geryon.plot.top_val_hyps \
        --input ${val_results} \
        --data-dir ${params.data_dir} \
        --output top_val_hypotheses.txt
    """
}

workflow {
    ratingsOverTime()
    ratingsByDepth()
    scoreByDepth()
    scoreOverTime()

    if (params.parquet_dir) {
        val_csv = executeAgainstVal().first()
        pvalComparison(val_csv)
        qvalComparison(val_csv)
        topValHypotheses(val_csv)
    }
}
