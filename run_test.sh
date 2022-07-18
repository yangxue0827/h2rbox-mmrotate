CUDA_VISIBLE_DEVICES=0,1,2 PORT=29817 \
./tools/dist_test.sh configs/wsood_v2/wsood_v2_r152_adamw_fpn_1x_dota_ms_le90.py \
        work_dirs/wsood_v2_r152_adamw_fpn_1x_dota_ms_le90/latest.pth 3 --format-only\
        --eval-options submission_dir=work_dirs/wsood_v2_r152_adamw_fpn_1x_dota_ms_le90/Task1_results
