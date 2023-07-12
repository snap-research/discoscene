set -x 
PORT=13456 ./scripts/training_demos/discoscene_res256.sh 8
    ./data/waymo/waymo_images \
    ./data/waymo/waymo_train.pkl \
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
    --ps_type waymo \ 
    --bg_nerf_resolution 4 \
    --use_pg False \
    --use_ada True \
    --ada_type linear \
    --ada_milestone 60_000 \
    --ada_w_color True \
    --use_bbox_2d True \
    --object_resolution 128 \
    --object_use_ada True \
    --object_ada_target_p 0.2 \
    --object_ada_type fixed \
    --objectada_w_spatial False \
    --num_bbox 2 \
    --use_mask False \
    --job_name training/waymo256
