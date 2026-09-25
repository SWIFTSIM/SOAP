#!/bin/bash
#
# Integration test for the chunk-file restart logic in
# compute_halo_properties.py (see chunk_tasks.py: ChunkTask.__call__).
#
# SOAP normally deletes the per-chunk scratch files once the combined output
# has been written. Passing --keep-scratch-files leaves them in place, which
# lets us run SOAP repeatedly against the same scratch directory and check:
#
#   1. initial run                -> no chunks reused (none exist yet)
#   2. identical rerun            -> every chunk reused from the scratch files,
#                                    and the catalogue is identical to run 1
#   3. rerun with one chunk file  -> only the surviving chunks are reused, the
#      deleted                       missing one is recomputed, combining the
#                                    mixed old/new chunks does not trip the
#                                    metadata consistency check, and the
#                                    catalogue is still identical to run 1
#   4. rerun with a property      -> no chunks reused (the stored dataset list
#      disabled in the config        no longer matches), and the disabled
#                                    property is absent from the new output
#
# This uses the same small DMO box as run_small_volume.sh.
#
set -eo pipefail

# Load the correct modules if we're running on cosma
if [[ $(hostname) == *cosma* ]] ; then
  module purge
  module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
  source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate
fi

NRANKS=8
# The box needs enough halos to fill this many non-empty chunks; drop to 1 if
# that ever stops being true for the test data.
CHUNKS=2
SNAP=18
CONFIG=tests/small_volume.yml

SCRATCH_DIR=./test_output/SOAP-tmp
OUTPUT_FILE=$(printf "./test_output/halo_properties_%04d.hdf5" ${SNAP})
WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}" "${SCRATCH_DIR}"' EXIT

# Start from a clean scratch directory so run 1 really is a cold start
rm -rf "${SCRATCH_DIR}"

# Download the required data
python tests/helpers.py

# Group membership (input for compute_halo_properties)
mpirun -np ${NRANKS} python -u SOAP/group_membership.py \
    --sim-name=DM_test --snap-nr=${SNAP} ${CONFIG}

run_soap () {
    # $1 = config file, $2 = log file
    rm -f "${OUTPUT_FILE}"
    mpirun -np ${NRANKS} python -u SOAP/compute_halo_properties.py \
        --sim-name=DM_test --snap-nr=${SNAP} --chunks=${CHUNKS} --dmo \
        --keep-scratch-files "$1" 2>&1 | tee "$2"
}

count_reused () {
    # Number of chunks that were restarted from a pre-existing scratch file
    grep -c "using pre-existing file for chunk" "$1" || true
}

compare_catalogues () {
    # Assert two catalogues have identical per-halo datasets (names and values)
    python - "$1" "$2" <<'EOF'
import sys
import h5py
import numpy as np


def per_halo_datasets(fname):
    """Every dataset whose first axis has length equal to the number of halos."""
    out = {}
    with h5py.File(fname, "r") as f:
        n_halo = f["InputHalos/HaloCatalogueIndex"].shape[0]

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape[:1] == (n_halo,):
                out[name] = obj[...]

        f.visititems(visit)
    return out


a = per_halo_datasets(sys.argv[1])
b = per_halo_datasets(sys.argv[2])
if set(a) != set(b):
    print(
        f"dataset sets differ: missing={sorted(set(a) - set(b))}, "
        f"extra={sorted(set(b) - set(a))}"
    )
    sys.exit(1)
bad = []
for name in sorted(a):
    x, y = a[name], b[name]
    if np.issubdtype(x.dtype, np.floating):
        same = x.shape == y.shape and np.array_equal(x, y, equal_nan=True)
    else:
        same = np.array_equal(x, y)
    if not same:
        bad.append(name)
if bad:
    print("datasets with differing values: " + ", ".join(bad))
    sys.exit(1)
EOF
}

fail () {
    echo "FAIL: $1"
    exit 1
}

echo
echo "=== Run 1: cold start, keeping scratch files ==="
run_soap ${CONFIG} "${WORKDIR}/run1.log"
[[ $(count_reused "${WORKDIR}/run1.log") -eq 0 ]] \
    || fail "run 1 reused chunk files that should not exist yet"
cp "${OUTPUT_FILE}" "${WORKDIR}/out1.hdf5"

echo
echo "=== Run 2: identical rerun, expect all ${CHUNKS} chunks reused ==="
run_soap ${CONFIG} "${WORKDIR}/run2.log"
n=$(count_reused "${WORKDIR}/run2.log")
[[ ${n} -eq ${CHUNKS} ]] || fail "run 2 reused ${n}/${CHUNKS} chunks"
compare_catalogues "${WORKDIR}/out1.hdf5" "${OUTPUT_FILE}" \
    || fail "run 2 catalogue differs from run 1"

echo
echo "=== Run 3: one scratch file deleted, expect $((CHUNKS - 1)) chunks reused ==="
mapfile -t chunk_files < <(find "${SCRATCH_DIR}" -name 'chunk_*.hdf5' | sort)
[[ ${#chunk_files[@]} -eq ${CHUNKS} ]] \
    || fail "expected ${CHUNKS} scratch files, found ${#chunk_files[@]}: ${chunk_files[*]}"
rm -f "${chunk_files[0]}"
echo "Deleted ${chunk_files[0]}"
run_soap ${CONFIG} "${WORKDIR}/run3.log"
n=$(count_reused "${WORKDIR}/run3.log")
[[ ${n} -eq $((CHUNKS - 1)) ]] || fail "run 3 reused ${n} chunks, expected $((CHUNKS - 1))"
compare_catalogues "${WORKDIR}/out1.hdf5" "${OUTPUT_FILE}" \
    || fail "run 3 catalogue differs from run 1"

echo
echo "=== Run 4: property disabled in config, expect no chunks reused ==="
python - "${WORKDIR}/modified.yml" <<'EOF'
import sys, yaml

with open("tests/small_volume.yml") as f:
    cfg = yaml.safe_load(f)
# Turn off a property that is otherwise computed for every SO variation
cfg["SOProperties"]["properties"]["SpinParameter"] = False
with open(sys.argv[1], "w") as f:
    yaml.safe_dump(cfg, f)
EOF
run_soap "${WORKDIR}/modified.yml" "${WORKDIR}/run4.log"
n=$(count_reused "${WORKDIR}/run4.log")
[[ ${n} -eq 0 ]] || fail "run 4 reused ${n} chunks after changing the property set"
python - "${OUTPUT_FILE}" <<'EOF'
import sys, h5py

with h5py.File(sys.argv[1], "r") as f:
    assert "SpinParameter" not in f["SO/200_crit"], \
        "SpinParameter should be absent from run 4 output"
    assert "SpinParameter" in f["BoundSubhalo"], \
        "disabling SO/SpinParameter should not affect BoundSubhalo/SpinParameter"
EOF

echo
echo "All restart integration checks passed."
