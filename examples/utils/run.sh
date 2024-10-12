MODEL=bert_base
python utils/save_model.py --model $MODEL --pretrained_model /data/models/bert-base-uncased --model_dir $MODEL
python utils/savedmodel2frozengraph.py --model_dir $MODEL


python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task tuning \
  --mononn_home /mononn \
  --mononn_dump_dir ./"$MODEL"_mononn_bs1 \
  --output_nodes Identity:0

python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task tuning \
  --output_nodes Identity:0 \
  --mononn_disable

python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task inference \
  --mononn_home /mononn \
  --mononn_spec_dir ./"$MODEL"_mononn_bs1 \
  --output_nodes Identity:0