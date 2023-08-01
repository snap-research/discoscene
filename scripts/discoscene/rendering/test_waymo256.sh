set -x
python render_scene.py \
    --checkpoint ./checkpoints/discoscene_waymo.pth \
    --work_dir work_dirs/rendering/waymo \
    --num 10 --step 70 --seed 1 \
    --render_type rotate_object \
    --generate_html True \
    --dataset_type waymo \
    discoscene \
    --val_anno_path ./data/waymo/waymo_val.pkl \
    --val_data_file_format dir \
    --num_bbox 2
