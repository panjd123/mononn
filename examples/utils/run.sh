BATCH_SIZE=12

# SEQ_LEN=128

# MODEL=bert_base

# python utils/save_model.py --model $MODEL --pretrained_model /data/models/bert-base-uncased --model_dir $MODEL
# python utils/savedmodel2frozengraph.py --model_dir $MODEL

# python run_mononn.py \
#   --model $MODEL \
#   --data_file data/bert_bs1.npy \
#   --task tuning \
#   --mononn_home /mononn \
#   --batch_size $BATCH_SIZE \
#   --seq_length $SEQ_LEN \
#   --mononn_dump_dir ./"$MODEL"_mononn_bs"$BATCH_SIZE"_sl"$SEQ_LEN" \
#   --output_nodes Identity:0

# python run_mononn.py \
#   --model $MODEL \
#   --data_file data/bert_bs1.npy \
#   --task inference \
#   --mononn_home /mononn \
#   --mononn_spec_dir ./"$MODEL"_mononn_bs"$BATCH_SIZE"_sl"$SEQ_LEN" \
#   --output_nodes Identity:0


SEQ_LEN=256
MODEL=bert_tiny_pesudo

python utils/save_model.py --model $MODEL --model_dir $MODEL --seq_length $SEQ_LEN
python utils/savedmodel2frozengraph.py --model_dir $MODEL

python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task tuning \
  --mononn_home /mononn \
  --batch_size $BATCH_SIZE \
  --seq_length $SEQ_LEN \
  --mononn_dump_dir ./"$MODEL"_mononn_bs"$BATCH_SIZE"_sl"$SEQ_LEN" \
  --output_nodes Identity:0

python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task inference \
  --mononn_home /mononn \
  --mononn_spec_dir ./"$MODEL"_mononn_bs"$BATCH_SIZE"_sl"$SEQ_LEN" \
  --output_nodes Identity:0
