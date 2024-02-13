#!/bin/bash
metals=("U" "Am" "Cu" "Zn" "Mg")

check_metal() {
  metal=$1
  if [ -f outs/train_charge_metal/${metal}.out ]; then
    echo "Output file for ${metal} already exists. Skipping."
    return 1
  fi

  if [ -f Output/MengetChargeMetal/${metal}_*/metrics.json ]; then
    echo "Metrics file already exists for ${metal}. Skipping."
    return 1
  fi

  return 0
}
for metal in "${metals[@]}"; do
    while [ $(ps uaxw | grep "train_charge_metal.py" | grep -v grep | wc -l) -ge 5 ]; do
        sleep 1
    done
    if check_metal "${metal}"; then
      ( nohup python Source/models/MEGNet_FCNN/experiments/train_charge_metal.py "${metal}" > outs/train_charge_metal/${metal}.out & )
      echo "Started processing ${metal}"
      sleep 20
    fi
  done
