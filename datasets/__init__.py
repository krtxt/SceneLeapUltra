import logging
from torch.utils.data import Dataset
from .sceneleappro_cached import SceneLeapProDatasetCached, ForMatchSceneLeapProDatasetCached
from .sceneleapplus_cached import SceneLeapPlusDatasetCached

def build_datasets(data_cfg, stage=None):
    """
    Build SceneLeapDataset for training, validation and testing.
    data_cfg should include:
      - train: {root_dir, succ_grasp_dir}
      - val: {root_dir, succ_grasp_dir}
      - test: {root_dir, succ_grasp_dir}
      - mode: "object_centric"/"camera_centric"/"camera_centric_obj_mean_normalized"
      - name: "sceneleap"
    """
    logging.info("Configuring datasets...")
    if getattr(data_cfg, 'name', '').lower() == "sceneleap":
        train_set = val_set = test_set = None
        mode = getattr(data_cfg, 'mode', 'camera_centric')

        if stage in ["fit", None]:
            train_set = SceneLeapDatasetCached(
                root_dir=data_cfg.train.root_dir,
                succ_grasp_dir=data_cfg.train.succ_grasp_dir,
                obj_root_dir=data_cfg.train.obj_root_dir,
                max_grasps_per_object=data_cfg.train.max_grasps_per_object,
                mode=data_cfg.train.mode
            )
            val_set = ForMatchSceneLeapDatasetCached(
                root_dir=data_cfg.val.root_dir,
                succ_grasp_dir=data_cfg.val.succ_grasp_dir,
                obj_root_dir=data_cfg.val.obj_root_dir,
                max_grasps_per_object=data_cfg.val.max_grasps_per_object,
                mode=data_cfg.val.mode
            )
            logging.info(f"Train set size: {len(train_set)}")
            logging.info(f"Validation set size: {len(val_set)}")

        if stage in ["test", None]:
            test_set = ForMatchSceneLeapDatasetCachedWithIds(
                root_dir=data_cfg.test.root_dir,
                succ_grasp_dir=data_cfg.test.succ_grasp_dir,
                obj_root_dir=data_cfg.test.obj_root_dir,
                max_grasps_per_object=data_cfg.test.max_grasps_per_object,
                mode=data_cfg.test.mode
            )
            logging.info(f"Test set size: {len(test_set)}")

        return train_set, val_set, test_set
    elif getattr(data_cfg, 'name', '').lower() == "prosceneleap":
        train_set = val_set = test_set = None
        mode = getattr(data_cfg, 'mode', 'camera_centric')

        if stage in ["fit", None]:
            train_set = SceneLeapProDatasetCached(
                root_dir=data_cfg.train.root_dir,
                succ_grasp_dir=data_cfg.train.succ_grasp_dir,
                obj_root_dir=data_cfg.train.obj_root_dir,
                mode=data_cfg.train.mode,
                max_grasps_per_object=getattr(data_cfg.train, 'max_grasps_per_object', 200),
                mesh_scale=getattr(data_cfg.train, 'mesh_scale', 0.1),
                num_neg_prompts=getattr(data_cfg.train, 'num_neg_prompts', 4),
                enable_cropping=getattr(data_cfg.train, 'enable_cropping', True),
                max_points=getattr(data_cfg.train, 'max_points', 10000),
                cache_version=getattr(data_cfg.train, 'cache_version', 'v2.0_train_only')
            )
            val_set = ForMatchSceneLeapProDatasetCached(
                root_dir=data_cfg.val.root_dir,
                succ_grasp_dir=data_cfg.val.succ_grasp_dir,
                obj_root_dir=data_cfg.val.obj_root_dir,
                mode=data_cfg.val.mode,
                max_grasps_per_object=getattr(data_cfg.val, 'max_grasps_per_object', 200),
                mesh_scale=getattr(data_cfg.val, 'mesh_scale', 0.1),
                num_neg_prompts=getattr(data_cfg.val, 'num_neg_prompts', 4),
                enable_cropping=getattr(data_cfg.val, 'enable_cropping', True),
                max_points=getattr(data_cfg.val, 'max_points', 10000),
                cache_version=getattr(data_cfg.val, 'cache_version', 'v1.0_formatch'),
                cache_mode="val"
            )
            logging.info(f"Train set size: {len(train_set)}")
            logging.info(f"Validation set size: {len(val_set)}")

        if stage in ["test", None]:
            test_set = ForMatchSceneLeapProDatasetCached(
                root_dir=data_cfg.test.root_dir,
                succ_grasp_dir=data_cfg.test.succ_grasp_dir,
                obj_root_dir=data_cfg.test.obj_root_dir,
                mode=data_cfg.test.mode,
                max_grasps_per_object=getattr(data_cfg.test, 'max_grasps_per_object', 200),
                mesh_scale=getattr(data_cfg.test, 'mesh_scale', 0.1),
                num_neg_prompts=getattr(data_cfg.test, 'num_neg_prompts', 4),
                enable_cropping=getattr(data_cfg.test, 'enable_cropping', True),
                max_points=getattr(data_cfg.test, 'max_points', 10000),
                cache_version=getattr(data_cfg.test, 'cache_version', 'v1.0_formatch'),
                cache_mode="test"
            )
            logging.info(f"Test set size: {len(test_set)}")

        return train_set, val_set, test_set
    elif getattr(data_cfg, 'name', '').lower() == "plussceneleap":
        train_set = val_set = test_set = None
        mode = getattr(data_cfg, 'mode', 'camera_centric')

        if stage in ["fit", None]:
            train_set = SceneLeapPlusDatasetCached(
                root_dir=data_cfg.train.root_dir,
                succ_grasp_dir=data_cfg.train.succ_grasp_dir,
                obj_root_dir=data_cfg.train.obj_root_dir,
                num_grasps=getattr(data_cfg.train, 'num_grasps', 8),
                mode=data_cfg.train.mode,
                max_grasps_per_object=getattr(data_cfg.train, 'max_grasps_per_object', 200),
                mesh_scale=getattr(data_cfg.train, 'mesh_scale', 0.1),
                num_neg_prompts=getattr(data_cfg.train, 'num_neg_prompts', 4),
                enable_cropping=getattr(data_cfg.train, 'enable_cropping', True),
                max_points=getattr(data_cfg.train, 'max_points', 10000),
                use_object_mask = getattr(data_cfg.train, 'use_object_mask', False),
                use_negative_prompts = getattr(data_cfg.train, 'use_negative_prompts', True),
                grasp_sampling_strategy=getattr(data_cfg.train, 'grasp_sampling_strategy', 'random'),
                cache_version=getattr(data_cfg.train, 'cache_version', 'v1.0_plus'),
                cache_mode=getattr(data_cfg.train, 'cache_mode', 'train'),
                # 穷尽采样参数
                use_exhaustive_sampling=getattr(data_cfg.train, 'use_exhaustive_sampling', False),
                exhaustive_sampling_strategy=getattr(data_cfg.train, 'exhaustive_sampling_strategy', 'sequential')
            )
            val_set = SceneLeapPlusDatasetCached(
                root_dir=data_cfg.val.root_dir,
                succ_grasp_dir=data_cfg.val.succ_grasp_dir,
                obj_root_dir=data_cfg.val.obj_root_dir,
                num_grasps=getattr(data_cfg.val, 'num_grasps', 8),
                mode=data_cfg.val.mode,
                max_grasps_per_object=getattr(data_cfg.val, 'max_grasps_per_object', 200),
                mesh_scale=getattr(data_cfg.val, 'mesh_scale', 0.1),
                num_neg_prompts=getattr(data_cfg.val, 'num_neg_prompts', 4),
                enable_cropping=getattr(data_cfg.val, 'enable_cropping', True),
                max_points=getattr(data_cfg.val, 'max_points', 10000),
                use_object_mask = getattr(data_cfg.val, 'use_object_mask', False),
                use_negative_prompts = getattr(data_cfg.val, 'use_negative_prompts', True),
                grasp_sampling_strategy=getattr(data_cfg.val, 'grasp_sampling_strategy', 'random'),
                cache_version=getattr(data_cfg.val, 'cache_version', 'v1.0_plus'),
                cache_mode=getattr(data_cfg.val, 'cache_mode', 'val'),
                # 穷尽采样参数
                use_exhaustive_sampling=getattr(data_cfg.val, 'use_exhaustive_sampling', False),
                exhaustive_sampling_strategy=getattr(data_cfg.val, 'exhaustive_sampling_strategy', 'sequential')
            )
            logging.info(f"Train set size: {len(train_set)}")
            logging.info(f"Validation set size: {len(val_set)}")

        if stage in ["test", None]:
            test_set = SceneLeapPlusDatasetCached(
                root_dir=data_cfg.test.root_dir,
                succ_grasp_dir=data_cfg.test.succ_grasp_dir,
                obj_root_dir=data_cfg.test.obj_root_dir,
                num_grasps=getattr(data_cfg.test, 'num_grasps', 8),
                mode=data_cfg.test.mode,
                max_grasps_per_object=getattr(data_cfg.test, 'max_grasps_per_object', 200),
                mesh_scale=getattr(data_cfg.test, 'mesh_scale', 0.1),
                num_neg_prompts=getattr(data_cfg.test, 'num_neg_prompts', 4),
                enable_cropping=getattr(data_cfg.test, 'enable_cropping', True),
                max_points=getattr(data_cfg.test, 'max_points', 10000),
                use_object_mask = getattr(data_cfg.test, 'use_object_mask', False),
                use_negative_prompts = getattr(data_cfg.test, 'use_negative_prompts', True),
                grasp_sampling_strategy=getattr(data_cfg.test, 'grasp_sampling_strategy', 'random'),
                cache_version=getattr(data_cfg.test, 'cache_version', 'v1.0_plus'),
                cache_mode=getattr(data_cfg.test, 'cache_mode', 'test'),
                # 穷尽采样参数
                use_exhaustive_sampling=getattr(data_cfg.test, 'use_exhaustive_sampling', False),
                exhaustive_sampling_strategy=getattr(data_cfg.test, 'exhaustive_sampling_strategy', 'sequential')
            )
            logging.info(f"Test set size: {len(test_set)}")

        return train_set, val_set, test_set
    else:
        logging.error(f"Unknown dataset name: {getattr(data_cfg, 'name', None)}")
        return None, None, None