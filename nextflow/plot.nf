#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.data_dir   = "${projectDir}/../geryon_data"
params.output_dir = "${projectDir}/../plots"
params.chain      = ""  // empty = pool every chain; cost ignores it

// One process for every plot: each is `python -m geryon.plot.<module>` writing
// <module>.pdf. Extra CLI args ride along with the module name.
process plot {
    tag "${module}"
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    tuple val(module), val(extra_args)

    output:
    path "${module}.pdf"

    script:
    """
    uv run python -m geryon.plot.${module} \
        --data-dir ${params.data_dir} \
        --output ${module}.pdf ${extra_args}
    """
}

def chainArg() {
    return params.chain ? "--chain ${params.chain}" : ""
}

workflow COST {
    plot(channel.of(['cost_over_time', '']))
}

workflow RATINGS_OVER_TIME {
    plot(channel.of('ratings_over_time', 'score_over_time').map { m -> [m, chainArg()] })
}

workflow RATINGS_BY_DEPTH {
    plot(
        channel.of('ratings_by_depth', 'score_by_depth', 'trust_by_depth')
            .map { m -> [m, chainArg()] }
    )
}

workflow {
    COST()
    RATINGS_OVER_TIME()
    RATINGS_BY_DEPTH()
}
