echo "This script downloads models used in the DiscoScene."

mkdir -p checkpoints 

echo "Start downloading clevr checkpoint..."
wget https://www.dropbox.com/s/6ji7wiucf2pdy7v/discoscene_clevr-checkpoint-268800.pth?dl=0 -O ./checkpoints/discoscene_clevr.pth
echo "Done!"

echo "Start downloading 3dfront checkpoint..."
wget https://www.dropbox.com/s/fzor7jpgr0lmouz/discoscene_3dfront-checkpoint-211200.pth?dl=0 -O ./checkpoints/discoscene_3dfront.pth
echo "Done!"

echo "Start downloading waymo checkpoint..."
wget https://www.dropbox.com/s/yh5re8ow9p66zc4/discoscene_waymo-checkpoint-326400.pth?dl=0 -O ./checkpoints/discoscene_waymo.pth
echo "Done!"
