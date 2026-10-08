#!/usr/bin/env bash
# Downloads the DataCo Smart Supply Chain dataset (public mirror of the Mendeley/Kaggle release).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
rm -rf /tmp/dataco_src
git clone --depth 1 https://github.com/ashishpatel26/DataCo-Smart-Supply-Chain-for-Big-Data-Analysis.git /tmp/dataco_src
cp /tmp/dataco_src/DataCoSupplyChainDataset.csv data/
rm -rf /tmp/dataco_src
echo "data/DataCoSupplyChainDataset.csv ready"
