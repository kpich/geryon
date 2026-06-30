#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.data_dir    = "${projectDir}/../geryon_data"
params.output_dir  = "${projectDir}/../plots"

process costOverTime {
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    output:
    path "cost_over_time.pdf"

    script:
    """
    uv run python -m geryon.plot.cost_over_time \
        --data-dir ${params.data_dir} \
        --output cost_over_time.pdf
    """
}

workflow {
    costOverTime()
}
