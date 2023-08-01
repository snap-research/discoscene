set -x 
python render_scene.py \
    --checkpoint ./checkpoints/discoscene_clevr.pth \
    --work_dir work_dirs/rendering/clevr \
    --num 10 --step 70 --seed 1 \
    --render_type add_object \
    --generate_html True \
    --dataset_type clevr \
    discoscene \
    --val_anno_path ./data/clevr/clevr_val.json \
    --val_data_file_format dir \
    --num_bbox 2 
