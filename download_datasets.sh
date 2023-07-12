echo "This script downloads datasets used in the DiscoScene."
echo "Choose from the following options:"
echo "0 - Clevr Dataset"
echo "1 - 3DFront Dataset"
echo "2 - Waymo Dataset"
read -p "Enter dataset ID: " ds_id

mkdir -p ./data/
if [ $ds_id == 0 ]
then
    echo "You chose Clevr Dataset"
    mkdir -p data/clevr
    cd data/clevr
    echo "Start downloading ..."
    wget https://www.dropbox.com/s/eov7g0zrt75citb/clevr_train.json?dl=0 -O clevr_train.json
    wget https://www.dropbox.com/s/6n7sbujou38hvjd/clevr_val.json?dl=0 -O clevr_val.json
    wget https://storage.googleapis.com/mchai/yxu5/clevr2_ann.zip -O clevr_images.zip 
    echo "done! Start unzipping ..."
    unzip clevr_images.zip 
    mv clevr2_ann/images clevr_images
    rm -rf clevr2_ann
    echo "done!"
elif [ $ds_id == 1 ]
then
    echo "You chose 3DFront Dataset"
    mkdir -p data/3dfront
    cd data/3dfront
    echo "Start downloading ..."
    echo "Only validation annotation..."
    wget https://www.dropbox.com/s/en7xaorxs3ysbkp/3dfront_val.json?dl=0 -O 3dfront_val.json
    echo "done! Start unzipping ..."
    echo "done!"
elif [ $ds_id == 2 ]
then
    echo "You chose Waymo Dataset"
    mkdir -p data/waymo
    cd data/waymo
    echo "Start downloading ..."
    wget https://www.dropbox.com/s/5vv1ak8qgbavd65/waymo_train.pkl?dl=0 -O waymo_train.pkl
    wget https://www.dropbox.com/s/72bbpt1j4su34yv/waymo_val.pkl?dl=0 -O waymo_val.pkl
    wget https://storage.googleapis.com/mchai/yxu5/training_pad.zip -O waymo_images.zip
    echo "done! Start unzipping ..."
    mkdir waymo_images && unzip waymo_images.zip -d waymo_images
    echo "done!"
else
    echo "You entered an invalid ID!"
fi
    

