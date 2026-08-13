#!/usr/bin/env bash
set -euo pipefail
export SPARK_LOCAL_IP=127.0.0.1
cd /workspace/spark-claim
mkdir -p /workspace/submissions
sbt -batch "runMain claim.TrainApp /workspace/data /workspace/submissions 10"
