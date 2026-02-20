#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.data_dir   = "${projectDir}/../geryon_data"
params.output_dir = "${projectDir}/../plots"

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

workflow {
    ratingsOverTime()
    ratingsByDepth()
}
