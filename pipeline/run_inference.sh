# script to run inference
input_dicoms=${1}
breast_model=${2}
dense_model=${3}
output_dir=${4}

python3 scripts/change_mode.py ${output_dir}

# preprocess images
preproc_dir="${output_dir}/preproc_imgs"
mkdir ${preproc_dir}
python3 scripts/change_mode.py ${preproc_dir}
python3 scripts/execute_libra_preprocessing.py -i ${input_dicoms} -o ${preproc_dir}

# run breast segmentation model
breast_masks="${output_dir}/breast_masks_inference"
mkdir ${breast_masks}
python3 scripts/change_mode.py ${breast_masks}
python3 scripts/inference.py "${preproc_dir}/breast_net_data/image/" ${breast_masks} ${breast_model}

# produce density model inputs
dense_preproc_dir="${output_dir}/dense_inputs"
mkdir ${dense_preproc_dir}
python3 scripts/change_mode.py ${dense_preproc_dir}
python3 scripts/preprocess_density_inputs.py "${preproc_dir}/breast_net_data/image/" ${breast_masks} ${dense_preproc_dir}

# run dense tissue segmentation model
dense_masks="${output_dir}/dense_masks_inference"
mkdir ${dense_masks}
python3 scripts/change_mode.py ${dense_masks}
python3 scripts/inference.py ${dense_preproc_dir} ${dense_masks} ${dense_model}

# postprocess masks
python3 scripts/postprocess_masks.py ${breast_masks} ${preproc_dir}
python3 scripts/postprocess_masks.py ${dense_masks} ${preproc_dir}

# PD calculation
python3 scripts/calculate_pd.py ${breast_masks} ${dense_masks} ${output_dir}
