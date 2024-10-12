MODEL=bert_base
python utils/save_model.py --model $MODEL --pretrained_model /home/v-jundapan/data/models/bert-base-uncased --model_dir $MODEL
python utils/savedmodel2frozengraph.py --model_dir $MODEL


/home/v-jundapan/miniconda/envs/mononn/bin/python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task tuning \
  --mononn_home /home/v-jundapan/mononn \
  --mononn_dump_dir ./"$MODEL"_mononn_bs1 \
  --output_nodes Identity:0

/home/v-jundapan/miniconda/envs/mononn/bin/python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task tuning \
  --output_nodes Identity:0 \
  --mononn_disable

python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task inference \
  --mononn_home /home/v-jundapan/mononn \
  --mononn_spec_dir ./"$MODEL"_mononn_bs1 \
  --output_nodes Identity:0