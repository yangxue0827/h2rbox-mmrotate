CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29813 \
./tools/dist_test.sh configs/wsood_v2/wsood_v2_r50_adamw_fpn_3x_dota_le90_.py \
        work_dirs/wsood_v2_r50_adamw_fpn_3x_dota_le90_/latest.pth 4 --format-only\
        --eval-options submission_dir=work_dirs/wsood_v2_r50_adamw_fpn_3x_dota_le90_/Task1_results
