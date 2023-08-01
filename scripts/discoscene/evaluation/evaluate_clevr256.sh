set -x
./scripts/test_metrics_discoscene.sh  3 \
    ./data/clevr/clevr_images \
    ./data/clevr/clevr_train.json \
    ./checkpoints/discoscene_clevr.pth \
    50000 \
    fid
