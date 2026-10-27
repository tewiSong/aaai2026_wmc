#!/bin/bash -l
# Build experiments/env/tf/deepgoplus_tf.sif (cotainr, LUMI-C) for running the DeepGOPlus 1.0.6 CNN.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module load CrayEnv cotainr/2025.7.1
export TMPDIR="/tmp/$USER/cotainr_tf_$(date +%s)"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
cotainr build "$D/deepgoplus_tf.sif" --system=lumi-c --conda-env="$D/deepgoplus_tf_env.yml" --accept-licenses
singularity exec "$D/deepgoplus_tf.sif" python -c "import tensorflow as tf, numpy, pandas; print('tf', tf.__version__, 'numpy', numpy.__version__)"
