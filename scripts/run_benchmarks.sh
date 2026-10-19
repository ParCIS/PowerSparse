#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 || $# > 8 )); then
  cat >&2 <<'USAGE'
Usage: bash scripts/run_benchmarks.sh BUILD_DIR DATASET_ROOT {spmm|sddmm} DEVICE OUTPUT.csv [N_LIST] [REPEATS] [CHECK]
N_LIST is comma-separated (default: 32,128). REPEATS defaults to 100; CHECK to 0.
By default, all .mtx files directly in DATASET_ROOT are selected.
Set DATASET_LIST to a manifest to select specific inputs.
Paths in the manifest are relative to DATASET_ROOT. Blank lines and # comments are ignored.
USAGE
  exit 2
fi

build_dir=$1
dataset_root=$2
op=$3
device=$4
output=$5
n_list=${6:-32,128}
repeats=${7:-100}
check=${8:-0}
manifest=${DATASET_LIST:-}
[[ $op == spmm || $op == sddmm ]] || { echo 'Operator must be spmm or sddmm.' >&2; exit 2; }
[[ $device =~ ^[0-9]+$ && $repeats =~ ^[1-9][0-9]*$ && $check =~ ^[01]$ ]] || {
  echo 'Invalid DEVICE, REPEATS or CHECK.' >&2; exit 2;
}
[[ $n_list =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || { echo 'Invalid N_LIST.' >&2; exit 2; }
IFS=',' read -r -a widths <<< "$n_list"
for n in "${widths[@]}"; do
  [[ $n == 32 || $n == 128 ]] || {
    echo "$op benchmark: N=$n was not selected in main; please use N=32 or N=128." >&2
    exit 2
  }
done
binary="$build_dir/bin/powersparse_$op"
[[ -x $binary ]] || { echo "Missing executable: $binary" >&2; exit 1; }
matrices=()
if [[ -n $manifest ]]; then
  [[ -r $manifest ]] || { echo "Cannot read manifest: $manifest" >&2; exit 1; }
  while IFS= read -r entry || [[ -n $entry ]]; do
    entry=${entry%$'\r'}
    [[ $entry =~ ^[[:space:]]*$ || $entry =~ ^[[:space:]]*# ]] && continue
    matrices+=("$dataset_root/$entry")
  done < "$manifest"
else
  shopt -s nullglob
  matrices=("$dataset_root"/*.mtx)
fi
(( ${#matrices[@]} > 0 )) || { echo 'No matrices selected.' >&2; exit 1; }
for matrix in "${matrices[@]}"; do
  [[ -f $matrix && -r $matrix ]] || { echo "Missing matrix: $matrix" >&2; exit 1; }
done
mkdir -p -- "$(dirname -- "$output")"
# Refuse to overwrite a previous run. A failed run stays explicitly partial.
[[ ! -e $output && ! -e $output.partial ]] || { echo "Output already exists: $output (or .partial)" >&2; exit 1; }
(set -o noclobber; "$binary" --header > "$output.partial")
for matrix in "${matrices[@]}"; do
  for n in "${widths[@]}"; do
    "$binary" "$matrix" "$n" "$device" "$check" "$repeats" >> "$output.partial"
  done
done
mv -- "$output.partial" "$output"
echo "Results saved to $output" >&2
