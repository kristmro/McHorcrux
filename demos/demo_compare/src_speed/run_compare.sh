#!/usr/bin/env bash
# demos/demo_compare/run_compare.sh — resume-able per (N,dt,st) runner

PY=python
SCRIPT=demos/demo_compare/speed_compare.py
OUTFILE=speed_compare.csv

# ensure CSV exists with header
if [ ! -f "${OUTFILE}" ]; then
  echo "N,dt,simtime,jax_time,numpy_time,torch_time" > "${OUTFILE}"
fi

Ns=(40 80 160 320)
dts=(0.1 0.05 0.01 0.005)
sts=(40 80 160 320)

for N in "${Ns[@]}"; do
  for dt in "${dts[@]}"; do
    for st in "${sts[@]}"; do

      # skip if already done
      if grep -qE "^${N},${dt},${st}," "${OUTFILE}"; then
        echo "Skipping  N=${N}, dt=${dt}, st=${st}"
        continue
      fi

      echo "Running   N=${N}, dt=${dt}, st=${st}"
      $PY "${SCRIPT}" --N "${N}" --dt "${dt}" --st "${st}" --out "${OUTFILE}"

    done
  done
done

echo "All done — results in ${OUTFILE}"
