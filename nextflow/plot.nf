#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.data_dir   = "${projectDir}/../geryon_data/sessions"
params.output_dir = "${projectDir}/../plots"
params.chain      = ""  // empty = pool every chain; cost ignores it

// Publishing: each subworkflow owns one subdir of output_dir, named like the
// subworkflow (COST -> cost/). Plots go in the subdir itself, and every other
// published file (tables, rerun results) goes in its data/.

// One process for every plot: each is `python -m geryon.plot.<module>` writing
// <module>.pdf. Extra CLI args ride along with the module name. Plots are cheap and
// their only input is a path string, so caching them would make `-resume` serve
// stale plots after new sessions land.
process plot {
    tag "${module}"
    errorStrategy 'terminate'
    cache false
    // Written under subdir/ so the copy lands there; publishDir can't name an input.
    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    tuple val(subdir), val(module), val(extra_args)

    output:
    path "${subdir}/${module}.pdf"

    script:
    """
    mkdir -p ${subdir}
    uv run python -m geryon.plot.${module} \
        --data-dir ${params.data_dir} \
        --output ${subdir}/${module}.pdf ${extra_args}
    """
}

// Reruns every hypothesis script on the validation split in the docker sandbox. The
// slow step, so it is cached: `sessions_key` changes whenever any hypotheses.jsonl
// does. Edits to holdout_rerun.py itself don't invalidate it; drop -resume then.
process holdoutRerun {
    errorStrategy 'terminate'
    publishDir "${params.output_dir}/critic_holdout/data", mode: 'copy', overwrite: true

    input:
    val sessions_key
    val chain_arg

    output:
    path 'holdout_runs.jsonl'

    script:
    """
    uv run python -m geryon.plot.holdout_rerun \
        --data-dir ${params.data_dir} \
        --output holdout_runs.jsonl ${chain_arg}
    """
}

// Joins the rerun with explore results and critiques, and computes the baselines.
process holdoutTable {
    errorStrategy 'terminate'
    cache false
    publishDir "${params.output_dir}/critic_holdout/data", mode: 'copy', overwrite: true

    input:
    path runs

    output:
    tuple path('holdout_table.csv'), path('holdout_forecasts.csv')

    script:
    """
    uv run python -m geryon.plot.holdout_table \
        --data-dir ${params.data_dir} \
        --runs ${runs} \
        --table holdout_table.csv \
        --forecasts holdout_forecasts.csv
    """
}

process plotHoldout {
    tag "${module}"
    errorStrategy 'terminate'
    cache false
    publishDir "${params.output_dir}/critic_holdout", mode: 'copy', overwrite: true

    input:
    tuple val(module), path(table), path(forecasts)

    output:
    path "${module}.pdf"

    script:
    """
    uv run python -m geryon.plot.${module} \
        --table ${table} \
        --forecasts ${forecasts} \
        --output ${module}.pdf
    """
}

def chainArg() {
    return params.chain ? "--chain ${params.chain}" : ""
}

workflow COST {
    plot(channel.of(['cost', 'cost_over_time', '']))
}

// The critic's ratings of the hypotheses, as scored on the exploration data.
workflow CRITIC_EXPLORATION {
    plot(
        channel.of(
            'ratings_over_time', 'score_over_time',
            'ratings_by_depth', 'score_by_depth', 'trust_by_depth',
        ).map { m -> ['critic_exploration', m, chainArg()] }
    )
}

// How the critic's judgements hold up when each script is rerun on held-out patients.
workflow CRITIC_HOLDOUT {
    sessions_key = channel.fromPath("${params.data_dir}/**/hypotheses.jsonl")
        .map { f -> "${f}:${f.size()}:${f.lastModified()}" }
        .toSortedList()
    tables = holdoutTable(holdoutRerun(sessions_key, chainArg()))
    plotHoldout(
        channel.of(
            'holdout_effect', 'holdout_paired', 'holdout_error_over_time',
            'holdout_coverage', 'holdout_q_by_trust',
        )
            .combine(tables)
    )
}

workflow {
    COST()
    CRITIC_EXPLORATION()
    CRITIC_HOLDOUT()
}
