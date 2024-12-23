<h1 align="center">
  Self-Aligning Depth-regularized Radiance Fields for Asynchronous RGB-D Sequences 
  
  <br>

  <a href="https://air.tsinghua.edu.cn/en/"><img height="50" src="assets/AIR.png"> </a>
</h1>

This is the code for Self-Aligning Depth-regularized Radiance Fields for Asynchronous RGB-D Sequences.


<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2211.07459)

</div>

<div align="center">
  <img src="./assets/main.png" width="100%" height="100%">
</div>

*The time-pose function is modeled using a 1-D multi-resolution hash grid with direct and speed losses. After bootstrapping the scene representation networks with pure RGB signals, the predicted depth sensor poses are used for jointly optimizing the NeRFs' parameters. At each timestamp, only one modality of sensor signals is provided, thus only one loss term is activated (shown on the right).*

## Install

```
git clone https://github.com/saythe17/AsyncNeRF.git
cd AsyncNeRF

conda env create -f environment.yml
conda activate async-nerf
```

The codebase has been mainly tested against CUDA >= 11.3 and V100/3090 GPUs. 

### Custom Data for the Time-Pose Function

If creating a custom dataset manually, the expected directory structure is:
- datadir
  - {time-stamp}.txt: 4 * 4 transformation matrix

### Custom Data for Scene Representation Learning

If creating a custom dataset manually, the expected directory structure is:
- /coordinates.pt: [Torch file](https://pytorch.org/docs/stable/generated/torch.save.html) that should contain the following keys:
  - 'origin_drb': Origin of scene in real-world units
  - 'pose_scale_factor': Scale factor mapping from real-world unit (ie: meters) to [-1, 1] range
- '/{val|train}/rgbs/': JPEG or PNG images
- '/{val|train}/depthvis_{track}': depth maps for specific depth track
- '/{val|train}/metadata_rgb/': Image-specific image metadata saved as a torch file. Each image should have a corresponding metadata file with the following file format: {rgb_stem}.pt. Each metadata file should contain the following keys:
  - 'W': Image width
  - 'H': Image height
  - 'intrinsics': Image intrinsics in the following form: [fx, fy, cx, cy]
  - 'c2w': Camera pose. 3x4 camera matrix with the convention used in the original [NeRF repo](https://github.com/bmild/nerf), ie: x: down, y: right, z: backwards, followed by the following transformation: ```torch.cat([camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]], -1)```
- '/{val|train}/metadata_depth_{track}/': Image-specific metadata saved as a torch file. Each depth map should have a corresponding metadata file with the following file format: {depth_stem}.pt.
- '/{val|train}/pose_gt_{track}/': ground-truth camera pose for specific track, including track='rgb'


## Training

### Time-Pose Function

```
python ./time2pose/train.py --datapath $DATAPATH --exp_name $EXP_NAME --config_file $CONFIG --test_datapath $DEPTH_POSE
```

### Scene Representation Learning
1. Generate the training partitions for each submodule: 

  ```
  python scripts/create_cluster_masks.py --config configs/mega-nerf/${DATASET_NAME}.yml --dataset_path $DATASET_PATH --output $MASK_PATH --grid_dim $GRID_X $GRID_Y
  ```

  - **Note:** this can be run across multiple GPUs by instead running 
  
  ```
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS --max_restarts 0 scripts/create_cluster_masks.py <args>
  ```
2. Train each submodule: 

```
python mega_nerf/train.py --config_file configs/mega-nerf/${DATASET_NAME}.yml --exp_name $EXP_PATH --dataset_path $DATASET_PATH --chunk_paths $SCRATCH_PATH --cluster_mask_path ${MASK_PATH}/${SUBMODULE_INDEX} --centroid $CENTROID_ID --have_gt_poses
```

  - **Note:** training with against full scale data will write hundreds of GBs / several TBs of shuffled data to disk. You can downsample the training data using ```train_scale_factor``` option.
  - **Note:** we provide [a utility script](parscripts/run_8.txt) based on [parscript](https://github.com/mtli/parscript) to start multiple training jobs in parallel. It can run through the following command: 
    
```
CONFIG_FILE=configs/mega-nerf/${DATASET_NAME}.yaml EXP_PREFIX=$EXP_PATH DATASET_PATH=$DATASET_PATH CHUNK_PREFIX=$SCRATCH_PATH MASK_PATH=$MASK_PATH python -m parscript.dispatcher parscripts/run_8.txt -g $NUM_GPUS
```

3. Merge the trained submodules into a unified Mega-NeRF model: 
```
python scripts/merge_submodules.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml --ckpt_prefix ${EXP_PREFIX}- --centroid_path ${MASK_PATH}/params.pt --output $MERGED_OUTPUT
```

## Evaluation

Single-GPU evaluation: 
```
python mega_nerf/eval.py --config_file configs/nerf/${DATASET_NAME}.yaml --exp_name $EXP_NAME --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT
```

Multi-GPU evaluation: 
```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS mega_nerf/eval.py --config_file configs/nerf/${DATASET_NAME}.yaml --exp_name $EXP_NAME --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT
```


## Acknowledgements

Large parts of this codebase are based on existing work in the [Mega-NeRF](https://github.com/cmusatyalab/mega-nerf). This work was sponsored by Tsinghua-Toyota Joint Research Fund (20223930097).

## Citing
If you find our work useful, please consider citing:

```BibTeX
@misc{huang2024selfaligningdepthregularizedradiancefields,
      title={Self-Aligning Depth-regularized Radiance Fields for Asynchronous RGB-D Sequences}, 
      author={Yuxin Huang and Andong Yang and Yuantao Chen and Runyi Yang and Zhenxin Zhu and Chao Hou and Hao Zhao and Guyue Zhou},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2211.07459}, 
}
```
