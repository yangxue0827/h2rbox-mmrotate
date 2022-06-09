CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29816 \
./tools/dist_test.sh configs/wsood_v2/wsood_v2_r50_adamw_fpn_m2o_1x_dota_le90.py \
        work_dirs/wsood_v2_r50_adamw_fpn_m2o_1x_dota_le90/latest.pth 4 --format-only\
        --eval-options submission_dir=work_dirs/wsood_v2_r50_adamw_fpn_m2o_1x_dota_le90/Task1_results
