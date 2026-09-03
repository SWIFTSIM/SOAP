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
#                                    and the output schema is unchanged
#   3. rerun with one chunk file  -> only the surviving chunks are reused, the
#      deleted                       missing one is recomputed, and combining
#                                    the mixed old/new chunks does not trip the
#                                    metadata consistency check
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

SCRATCH_DIR=./output/SOAP-tmp
OUTPUT_FILE=$(printf "./output/halo_properties_%04d.hdf5" ${SNAP})
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

dump_schema () {
    # Sorted list of every dataset path in an HDF5 file
    python - "$1" <<'EOF'
import sys, h5py

paths = []
with h5py.File(sys.argv[1], "r") as f:
    f.visititems(lambda name, obj: paths.append(name) if isinstance(obj, h5py.Dataset) else None)
print("\n".join(sorted(paths)))
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
dump_schema "${WORKDIR}/out1.hdf5" > "${WORKDIR}/schema1.txt"

echo
echo "=== Run 2: identical rerun, expect all ${CHUNKS} chunks reused ==="
run_soap ${CONFIG} "${WORKDIR}/run2.log"
n=$(count_reused "${WORKDIR}/run2.log")
[[ ${n} -eq ${CHUNKS} ]] || fail "run 2 reused ${n}/${CHUNKS} chunks"
dump_schema "${OUTPUT_FILE}" > "${WORKDIR}/schema2.txt"
diff "${WORKDIR}/schema1.txt" "${WORKDIR}/schema2.txt" \
    || fail "run 2 output schema differs from run 1"

echo
echo "=== Run 3: one scratch file deleted, expect $((CHUNKS - 1)) chunks reused ==="
chunk_files=( "${SCRATCH_DIR}"/*/chunk_*.hdf5 )
rm -f "${chunk_files[0]}"
echo "Deleted ${chunk_files[0]}"
run_soap ${CONFIG} "${WORKDIR}/run3.log"
n=$(count_reused "${WORKDIR}/run3.log")
[[ ${n} -eq $((CHUNKS - 1)) ]] || fail "run 3 reused ${n} chunks, expected $((CHUNKS - 1))"
dump_schema "${OUTPUT_FILE}" > "${WORKDIR}/schema3.txt"
diff "${WORKDIR}/schema1.txt" "${WORKDIR}/schema3.txt" \
    || fail "run 3 output schema differs from run 1"

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
