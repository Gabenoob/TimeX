export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer
des='Timexer-MS'
patch_len=24


python3 -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/M5/ \
  --data_path df_raw.pkl \
  --model_id M5_192_24 \
  --model $model_name \
  --data M5 \
  --target HOBBIES_1_008_CA_1 \
  --features MS \
  --seq_len 192 \
  --pred_len 28 \
  --label_len 64 \
  --e_layers 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len $patch_len \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --itr 1