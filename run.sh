CUDA_VISIBLE_DEVICES=1,3 PORT=29811 \
./tools/dist_train.sh configs/wsood_v2/wsood_v2_r152_adamw_fpn_1x_dota_ms_le90.py 2 --auto-resume