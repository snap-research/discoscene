set -x
python render_scene.py \
    --checkpoint ./checkpoints/discoscene_3dfront.pth \
    --work_dir work_dirs/rendering/3dfront/ \
    --num 10 --step 70 --seed 1 \
    --render_type  move_camera \
    --generate_html True \
    --dataset_type 3dfront \
    --ssaa 2 \
    discoscene \
    --val_anno_path ./data/3dfront/3dfront_val.json \
    --val_data_file_format dir \
    --num_bbox 5 
