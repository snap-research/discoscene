# DiscoScene: Spatially Disentangled Generative Radiance Field for Controllable 3D-aware Scene Synthesis <br><sub>Official PyTorch implementation of the CVPR 2023 Highlight paper</sub>

<img src="./docs/contents/framework.jpg"/>

**Figure:** *Framework of DiscoScene.*

> **DiscoScene: Spatially Disentangled Generative Radiance Field for Controllable 3D-aware Scene Synthesis** <br>
> Yinghao Xu, Menglei Chai, Zifan Shi, Sida Peng, Ivan Skorokhodov, Aliaksandr Siarohin, Ceyuan Yang, Yujun Shen, Hsin-Ying Lee, Bolei Zhou, Sergey Tulyakov <br>

[[Paper](https://arxiv.org/abs/2212.11984)]
[[Project Page](https://snap-research.github.io/discoscene)]
[[Demo](https://www.youtube.com/watch?v=Fvenkw7yeok)]


This work presents DisCoScene: a 3D-aware generative model for high-quality and controllable scene synthesis.
The key ingredient of our approach is a very abstract object-level representation (3D bounding boxes without semantic annotation) as the scene layout prior, which is simple to obtain, general to describe various scene contents, and yet informative to disentangle objects and background. Moreover, it serves as an intuitive user control for scene editing.
Based on such a prior, our model spatially disentangles the whole scene into object-centric generative radiance fields by learning on only 2D images with the global-local discrimination. Our model obtains the generation fidelity and editing flexibility of individual objects while being able to efficiently compose objects and the background into a complete scene. We demonstrate state-of-the-art performance on many scene datasets, including the challenging Waymo outdoor dataset.



## Requirements
* All our model are trained and tested on V100, and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.11.0.
* CUDA 11.3 or later.
* Users can use the following commands to install the packages
```bash
conda create -n discoscene python=3.8
conda activate discoscene
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```
## Preparing datasets
We provide a [script](download_datasets.sh) to download the datasets. Clevr and 3D-Front are synthetic datasets and we will release the rendering scripts soon.
```
bash download_datasets.sh
```

## Test Demo

Download pretrained model with the [script](./download_models.sh)
```
bash download_models.sh
```
Users can use the following command to generate demo videes
```shell
python render.py \
    --work_dir ${WORK_DIR} \
    --checkpoint ${MODEL_PATH} \
    --num ${NUM} \
    --seed ${SEED} \
    --step ${STEP} \
    --render_type ${RENDER_TYPE} \
    --generate_html ${SAVE_HTML} \
    --dataset_type ${DATASET} \
    --ssaa ${ANTIALIASING} \
    discoscene \
    --val_anno_path ${VAL_ANN}\
    --val_data_file_format dir \
    --num_bbox ${NUM_BBOX} 
```

where

- `WORK_DIR` refers to the path to save the results.
- `MODEL_PATH` refers to the path of the pretrained model.
- `NUM` refers to the number of samples to synthesize.
- `SEED` refers to the random seed used for sampling.
- `STEP` refers to the number of steps for the generated video.
- `RENDER_TYPE` refers to the controlablity of the rendered videos, including `rotate_object`, `move_object`, `add_object`, `delete_object`, `rotate_camera`, `move_camera`.
- `SAVE_HTML` controls whether to save images as an HTML for better visualization when rendering videos.
- `DATASET` refers to the type of dataset, including `clevr`, `3dfront` and `waymo`.
- `SSAA` refers to the ratio of supersampling anti-aliasing.
- `VAL_ANN` refers to the layout information.
- `NUM_BBOX` refers the bouding box number of the annotation file.

We include scripts for rendering demo video on [Clevr](./scripts/discoscene/rendering/test_clevr256.sh), [3D-Front](./scripts/discoscene/rendering/test_3dfront256.sh) and [Waymo](./scripts/discoscene/rendering/test_waymo256.sh) dataset. 
Users can use following commands to generate demo videos.
```
bash ./scripts/discoscene/rendering/test_clevr256.sh
bash ./scripts/discoscene/rendering/test_3dfront256.sh
bash ./scripts/discoscene/rendering/test_waymo256.sh
```

## Training

For example, users can use the following command to train DiscoScene in the resolution of 256x256

```shell
./scripts/training_demos/discoscene_res256.sh\
    ${NUM_GPUS} \
    ${DATA_PATH} \
    [OPTIONS]
```

where

- `NUM_GPUS` refers to the number of GPUs used for training.
- `DATA_PATH` refers to the path to the dataset (`zip` format is strongly recommended).
- `[OPTIONS]` refers to any additional option to pass. Detailed instructions on available options can be found via `python train.py discoscene --help`.

**NOTE:** This demo script uses `discoscene_res256` as the default `job_name`, which is particularly used to identify experiments. Concretely, a directory with name `job_name` will be created under the root working directory, which is set as `work_dirs/` by default. To prevent overwriting previous experiments, an exception will be raised to interrupt the training if the `job_name` directory has already existed. Please use `--job_name=${JOB_NAME}` option to specify a new job name.

We include the training scripts on [Clevr](./scripts/discoscene/training/train_clevr256.sh), [3D-Front](./scripts/discoscene/training/train_3dfront256.sh) and [Waymo](./scripts/discoscene/training/train_clevr256.sh). Users can use these bash files to train our model
```
bash ./scripts/discoscene/training/train_clevr256.sh
bash ./scripts/discoscene/training/train_3dfront256.sh
bash ./scripts/discoscene/training/train_waymo256.sh
```

### Evaluation

Users can use the following command to evaluate a well-trained model

```shell
./scripts/test_metrics_discoscene.sh \
    ${NUM_GPUS} \
    ${DATA_PATH} \
    ${ANNOTATION_PATH} \
    ${MODEL_PATH} \
    ${NUM_FAKE_SAMPLES} \
    fid 
    [OPTIONS]
```
Here is the evaluation example for Clevr
```
bash scripts/discoscene/evaluation/evaluate_clevr256.sh
```

## BibTeX

```bibtex
@InProceedings{Xu_2023_CVPR,
    author    = {Xu, Yinghao and Chai, Menglei and Shi, Zifan and Peng, Sida and Skorokhodov, Ivan and Siarohin, Aliaksandr and Yang, Ceyuan and Shen, Yujun and Lee, Hsin-Ying and Zhou, Bolei and Tulyakov, Sergey},
    title     = {DisCoScene: Spatially Disentangled Generative Radiance Fields for Controllable 3D-Aware Scene Synthesis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
}
```
