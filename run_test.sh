CUDA_VISIBLE_DEVICES=0,2,1,3 PORT=29817 \
./tools/dist_test.sh configs/h2rbox/h2rbox_r50_adamw_fpn_1x_dota_ms_le90.py \
        work_dirs/h2rbox_r50_adamw_fpn_1x_dota_ms_le90/latest.pth 4 --format-only\
        --eval-options submission_dir=work_dirs/h2rbox_r50_adamw_fpn_1x_dota_ms_le90/Task1_results
