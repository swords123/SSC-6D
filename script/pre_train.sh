export CUDA_VISIBLE_DEVICES=0
category=can
number='1'
sub_version='1'
out_dir=output_pre

model_name="v${number}.${sub_version}"

echo $model_name

python tools/train_all.py \
--category ${category} \
--model_name ${model_name} \
--out_dir ${out_dir} \
--nepoch 10 \
--start_epoch 1 \
--start_real_epoch 10000 \
--aug

