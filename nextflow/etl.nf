#!/usr/bin/env nextflow

/*
 * Geryon ETL Pipeline
 *
 * Extracts TSV files from MSK-IMPACT data, transforms to parquet format
 * for efficient querying and downstream analysis.
 */

nextflow.enable.dsl = 2

// ============================================================================
// Parameters
// ============================================================================

params.data_root = '/Users/pichottk/data/msk-impact/msk_solid_heme'
params.output_base = "${System.getProperty('user.home')}/data/geryon_data"
params.file_pattern = '*.txt'
params.holdout_seed = 42

// Human-readable name for this data version; becomes the output directory name and is
// what hypotheses and chain definitions refer to. Defaults to the run date, which is
// what unnamed runs have always produced.
params.version = workflow.start.format('yyyy-MM-dd')

// ============================================================================
// Processes
// ============================================================================

process extractTSV {
    tag "${tsv_file.baseName}"
    cpus 1
    memory '2 GB'

    input:
    path tsv_file

    output:
    tuple path("${tsv_file.baseName}.parquet"), path("${tsv_file.baseName}.profile.json")

    script:
    """
    python -m geryon.etl.process_cbioportal_file \
        --input ${tsv_file} \
        --output ${tsv_file.baseName}.parquet \
        --log-level INFO
    """
}

process writeVersionMarker {
    publishDir "${params.output_base}/${params.version}",
               mode: 'copy', overwrite: false

    output:
    path "VERSION.json"

    script:
    """
    python -m geryon.etl.data_version \
        --dir . \
        --name '${params.version}' \
        --data-root '${params.data_root}' \
        --holdout-seed ${params.holdout_seed}
    """
}

process createPatientSplit {
    publishDir "${params.output_base}/${params.version}",
               mode: 'copy', overwrite: false

    input:
    path parquet

    output:
    path "patient_split.parquet"

    script:
    """
    python -m geryon.etl.create_patient_split \
        --input ${parquet} \
        --output patient_split.parquet \
        --seed ${params.holdout_seed}
    """
}

process splitByPatient {
    publishDir "${params.output_base}/${params.version}",
               mode: 'copy', overwrite: false

    input:
    path parquet_files
    path split_table
    path version_marker

    output:
    path "explore"
    path "validation"

    script:
    """
    python -m geryon.etl.split_by_patient \
        --input . \
        --output . \
        --split-table ${split_table}
    """
}

process publishResults {
    publishDir "${params.output_base}/${params.version}",
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

    // Generate exploration/validation patient split
    clinical_parquet = parquet_files
        .filter { parquet, profile -> parquet.name == 'data_clinical_patient.parquet' }
        .map { parquet, profile -> parquet }
    split_table = createPatientSplit(clinical_parquet)

    // Name this data version so hypotheses and chain definitions can refer to it.
    version_marker = writeVersionMarker()

    // Partition every table into explore/ and validation/ subdirs so the holdout
    // is enforced at the data layer (the inner-loop session reads explore/ only).
    all_parquet = parquet_files.map { parquet, profile -> parquet }.collect()
    splitByPatient(all_parquet, split_table, version_marker)

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
    log.info "Data version : ${params.version}"
    log.info "Output dir   : ${params.output_base}/${params.version}"
    log.info ""
}
