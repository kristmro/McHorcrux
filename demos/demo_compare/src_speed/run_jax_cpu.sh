#!/usr/bin/env bash
# demos/demo_compare/run_compare.sh
# ────────────────────────────────────────────────────────────────────────────────
# Add / update the JAX-CPU timings (column “jax_cpu”) in speed_compare.csv.  
# Skip rows that already have a non-empty jax_cpu entry.

PY=python
JAX_CPU_SCRIPT=demos/demo_compare/src_speed/speed_compare_cpu_jax.py   # new script
OUTFILE=speed_compare.csv

# ── ensure CSV exists and has the jax_cpu column ───────────────────────────────
if [ ! -f "${OUTFILE}" ]; then
  echo "N,dt,simtime,jax_time,numpy_time,torch_time,jax_cpu" > "${OUTFILE}"
elif ! head -n 1 "${OUTFILE}" | grep -q "jax_cpu"; then
  # append the header only once if the column is missing
  sed -i '1s/$/,jax_cpu/' "${OUTFILE}"
fi

# ── parameter grids ────────────────────────────────────────────────────────────
Ns=(40 80 160 320)
dts=(0.1 0.05 0.01 0.005)
sts=(40 80 160 320)

# ── main loop ──────────────────────────────────────────────────────────────────
for N in "${Ns[@]}"; do
  for dt in "${dts[@]}"; do
    for st in "${sts[@]}"; do

      # Does the row exist **and** already contain a non-empty jax_cpu field?
      if awk -F',' -v n="$N" -v d="$dt" -v s="$st" \
          '($1==n && $2==d && $3==s && $7!=""){found=1} END{exit !found}' \
          "${OUTFILE}"; then
        echo "Skipping   N=${N}, dt=${dt}, st=${st}  (jax_cpu present)"
        continue
      fi

      echo "Running    N=${N}, dt=${dt}, st=${st}  (fill jax_cpu)"
      ${PY} "${JAX_CPU_SCRIPT}" --N "${N}" --dt "${dt}" --st "${st}" --out "${OUTFILE}"

    done
  done
done

echo "All done — updated results are in ${OUTFILE}"
