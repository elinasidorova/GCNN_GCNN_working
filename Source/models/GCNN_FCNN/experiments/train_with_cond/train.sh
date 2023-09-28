#!/bin/bash
metals=("Li" "Be" "Na" "Mg" "Al" "K" "Ca" "Sc" "Ti" "V" "Cr" "Mn" "Fe" "Co" "Ni" "Cu" "Zn" "Ga" "Rb" "Sr" "Y" "Zr" "Mo" "Rh" "Pd" "Ag" "Cd" "In" "Sn" "Sb" "Cs" "Ba" "Hf" "Re" "Pt" "Au" "Hg" "Tl" "Pb" "Bi" "La" "Ce" "Pr" "Nd" "Pm" "Sm" "Eu" "Gd" "Tb" "Dy" "Ho" "Er" "Tm" "Yb" "Lu" "Ac" "Th" "Pa" "U" "Np" "Pu" "Am" "Cm" "Bk" "Cf")
out_folder="train_general_testonly_cond"
output_folder="Output/WithCond/5fold"
n_streams=2

function check_metal() {
  metal=$1
  if [ -f outs/${out_folder}/"${metal}".out ]; then
    echo "Out file for ${metal} already exists. Skipping."
    return 1
  fi

  for file in "${output_folder}"/"${metal}"_*/metrics.json; do
    if [ -f "$file" ]; then
      echo "Metrics file already exists for ${metal}. Skipping."
      return 1
    fi
    break
  done

  return 0
}

for metal in "${metals[@]}"; do
  while [ "$(pgrep 'python Source/models/GCNN_FCNN/experiments/train_with_cond/train.py' | wc -l)" -ge ${n_streams} ]; do
    sleep 1
  done
  if check_metal "${metal}"; then
    (nohup python Source/models/GCNN_FCNN/experiments/train_with_cond/train.py "${metal}" > outs/${out_folder}/${metal}.out &)
    echo "Started processing ${metal}"
    sleep 20
  fi
done
