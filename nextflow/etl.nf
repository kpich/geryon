#!/usr/bin/env nextflow

/*
 * MSK CYCL ETL Pipeline
 *
 * Extracts TSV files from MSK-IMPACT data, transforms to parquet format
 * for efficient querying and downstream analysis.
 */

nextflow.enable.dsl = 2

// ============================================================================
// Parameters
// ============================================================================

params.data_root = '/Users/pichottk/data/msk-impact/msk_solid_heme'
params.output_base = "${System.getProperty('user.home')}/data/msk_cycle_data"
params.file_pattern = '*.txt'

// ============================================================================
// Processes
// ============================================================================

process extractTSV {
    tag "${tsv_file.baseName}"

    input:
    path tsv_file

    output:
    tuple path("${tsv_file.baseName}.parquet"), path("${tsv_file.baseName}.profile.json")

    script:
    """
    python -m msk_cycl.etl.process_cbioportal_file \
        --input ${tsv_file} \
        --output ${tsv_file.baseName}.parquet \
        --log-level INFO
    """
}

process publishResults {
    publishDir "${params.output_base}/${workflow.start.format('yyyy-MM-dd')}",
               mode: 'copy',
               overwrite: false

    input:
    tuple path(parquet), path(profile)

    output:
    tuple path(parquet), path(profile)

    script:
    """
    echo "Publishing ${parquet.name} + ${profile.name}"
    """
}

// ============================================================================
// Workflow
// ============================================================================

workflow {
    // Create channel from input TSV files
    tsv_files = Channel
        .fromPath("${params.data_root}/${params.file_pattern}")
        .ifEmpty { error "No TSV files found in ${params.data_root}" }

    // Extract and transform
    parquet_files = extractTSV(tsv_files)

    // Publish to timestamped directory
    publishResults(parquet_files)
}

workflow.onComplete {
    log.info ""
    log.info "Pipeline execution summary"
    log.info "=========================="
    log.info "Completed at : ${workflow.complete}"
    log.info "Duration     : ${workflow.duration}"
    log.info "Success      : ${workflow.success}"
    log.info "Output dir   : ${params.output_base}/${workflow.start.format('yyyy-MM-dd')}"
    log.info ""
}
