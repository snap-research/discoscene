set -x 
PORT=13456 ./scripts/training_demos/discoscene_res256.sh 8 \
    ./data/clevr/clevr_images/ \
    ./data/clevr/clevr_train.json \
    --use_object True \
    --cam_path none \
    --fg_use_dirs False \
    --add_scene_d True \
    --batch_size 8 \
    --enable_amp False \
    --num_fp16_res 4 \
    --resolution 256 \
    --use_sr True \
    --use_stylegan2_d True \
    --d_fmaps_factor 0.5 \
    --d_lr 0.002 \
    --g_lr 0.002 \
    --enable_beta_mult True \
    --num_bbox 2 \
    --ps_type clevr \
    --bg_nerf_resolution 64 \
    --use_pg False \
    --use_ada False \
    --object_use_ada False \
    --job_name training/clevr256 \
    --r1_gamma 1
