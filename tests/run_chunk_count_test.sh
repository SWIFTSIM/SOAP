#!/bin/bash
#
# Integration test: the combined catalogue must not depend on how many chunks
# the volume is split into.
#
# compute_halo_properties.py sorts the final output by (cell index, catalogue
# index) (see combine_chunks.spatial_sort), which is a total order that does
# not depend on --chunks. Each halo's properties are computed from its own
# particles on a single rank, so runs with different chunk counts must produce
# identical per-halo datasets. This catches halos being dropped or duplicated
# at chunk boundaries, box-wrap bugs, and metadata-combination regressions.
#
# Uses the same small DMO box as run_small_volume.sh.
#
set -eo pipefail

# Load the correct modules if we're running on cosma
if [[ $(hostname) == *cosma* ]] ; then
  module purge
  module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
  source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate
fi

NRANKS=8
SNAP=18
CONFIG=tests/small_volume.yml
# The box needs enough halos to fill the largest of these with non-empty
# chunks; reduce if that ever stops being true for the test data.
CHUNK_COUNTS=(1 2 4)

SCRATCH_DIR=./test_output/SOAP-tmp
OUTPUT_FILE=$(printf "./test_output/halo_properties_%04d.hdf5" ${SNAP})
WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}" "${SCRATCH_DIR}"' EXIT

# Start from a clean scratch directory
rm -rf "${SCRATCH_DIR}"

# Download the required data
python tests/helpers.py

# Group membership (input for compute_halo_properties)
mpirun -np ${NRANKS} python -u SOAP/group_membership.py \
    --sim-name=DM_test --snap-nr=${SNAP} ${CONFIG}

for nchunks in "${CHUNK_COUNTS[@]}" ; do
    echo
    echo "=== compute_halo_properties with --chunks=${nchunks} ==="
    rm -f "${OUTPUT_FILE}"
    mpirun -np ${NRANKS} python -u SOAP/compute_halo_properties.py \
        --sim-name=DM_test --snap-nr=${SNAP} --chunks=${nchunks} --dmo ${CONFIG}
    cp "${OUTPUT_FILE}" "${WORKDIR}/chunks_${nchunks}.hdf5"
done

echo
echo "=== Comparing catalogues ==="
python - "${WORKDIR}" "${CHUNK_COUNTS[@]}" <<'EOF'
import sys
import h5py
import numpy as np

workdir = sys.argv[1]
counts = sys.argv[2:]


def per_halo_datasets(fname):
    """Every dataset whose first axis has length equal to the number of halos."""
    out = {}
    with h5py.File(fname, "r") as f:
        n_halo = f["InputHalos/HaloCatalogueIndex"].shape[0]

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape[:1] == (n_halo,):
                out[name] = obj[...]

        f.visititems(visit)
    return n_halo, out


ref_count = counts[0]
ref_n, ref = per_halo_datasets(f"{workdir}/chunks_{ref_count}.hdf5")
print(f"chunks={ref_count}: {ref_n} halos, {len(ref)} per-halo datasets (reference)")

ok = True
for count in counts[1:]:
    n, cur = per_halo_datasets(f"{workdir}/chunks_{count}.hdf5")
    if n != ref_n:
        print(f"FAIL chunks={count}: {n} halos vs reference {ref_n}")
        ok = False
        continue
    if set(cur) != set(ref):
        print(
            f"FAIL chunks={count}: dataset set differs "
            f"(missing={sorted(set(ref) - set(cur))}, "
            f"extra={sorted(set(cur) - set(ref))})"
        )
        ok = False
    for name in sorted(set(cur) & set(ref)):
        a, b = ref[name], cur[name]
        if np.issubdtype(a.dtype, np.floating):
            identical = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        else:
            identical = np.array_equal(a, b)
        if not identical:
            print(f"FAIL chunks={count}: {name} differs from reference")
            ok = False
    if ok:
        print(f"chunks={count}: identical to reference")

sys.exit(0 if ok else 1)
EOF

echo
echo "Chunk-count invariance check passed."
