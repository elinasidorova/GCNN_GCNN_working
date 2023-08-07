#!/bin/bash
metals=("Sc" "Y" "La" "Ce" "Pr" "Nd" "Pm" "Sm" "Eu" "Gd" "Tb" "Dy" "Ho" "Er" "Tm" "Yb" "Lu" "Ac" "Th" "Pa" "U" "Np" "Pu" "Am" "Cm" "Bk" "Cf")
out_folder="train_general_testonly_cond"
output_folder="Output/WithCond/5fold"
n_streems=2

check_metal() {
  metal=$1
  if [ -f outs/${out_folder}/${metal}.out ]; then
    echo "Out file for ${metal} already exists. Skipping."
    return 1
  fi

  if [ -f ${output_folder}/${metal}_*/metrics.json ]; then
    echo "Metrics file already exists for ${metal}. Skipping."
    return 1
  fi

  return 0
}
for metal in "${metals[@]}"; do
  while [ $(ps uaxw | grep "python Source/models/GCNN_FCNN/experiments/train_with_cond/train.py" | grep -v grep | wc -l) -ge ${n_streems} ]; do
    sleep 1
  done
  if check_metal "${metal}"; then
    (nohup python Source/models/GCNN_FCNN/experiments/train_with_cond/train.py "${metal}" > outs/${out_folder}/${metal}.out &)
    echo "Started processing ${metal}"
    sleep 20
  fi
done
