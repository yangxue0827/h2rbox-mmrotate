CUDA_VISIBLE_DEVICES=1,2,3 PORT=29813 \
./tools/dist_test.sh configs/rotated_reppoints/rotated_reppoints_r50_adamw_fpn_1x_dota_oc.py \
        work_dirs/rotated_reppoints_r50_adamw_fpn_1x_dota_oc/latest.pth 3 --format-only\
        --eval-options submission_dir=work_dirs/rotated_reppoints_r50_adamw_fpn_1x_dota_oc/Task1_results
