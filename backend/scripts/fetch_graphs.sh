#!/usr/bin/env bash
# Download the public SocioPatterns school contact graphs (CC BY-NC-SA 3.0) from the Netzschleuder mirror.
#   Marseille high school 2013, proximity layer: Mastrandrea et al., PLoS ONE 2015
#   Lyon primary school 2009: Stehle et al., PLoS ONE 2011
# Files land in backend/data_raw/ (git-ignored). Non-commercial use only; cite the papers above.
set -euo pipefail
cd "$(dirname "$0")/../data_raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data_raw"; cd "$(dirname "$0")/../data_raw"; }
fetch() { # name url
  if [ ! -f "$1/edges.csv" ]; then
    curl -fsSL -o "$1.csv.zip" "$2"
    mkdir -p "$1" && unzip -oq "$1.csv.zip" -d "$1" && rm -f "$1.csv.zip"
  fi
}
fetch hs2013_proximity https://networks.skewed.de/net/sp_high_school/files/proximity.csv.zip
fetch primary2009 https://networks.skewed.de/net/sp_primary_school/files/sp_primary_school.csv.zip
echo "ok: $(pwd)"
