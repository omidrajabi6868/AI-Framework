import argparse
import json
from pathlib import Path
import torch

from Dataset.Dataset import COCO, VOC, GlobalDataset
from Dataset.augmentations import build_augmentations
from Network.Network import NYOLO, NMediaPipe
from Network.utils import load_state_dict_compat
from Train.Train import TYOLO
from Validation.DetectionSegmentation import ValDetectionSegmentation
from Validation.PoseEstimation import ValPoseEstimation
from Validation.ObjectTracking import ValObjectTracking
from Validation.FallDetection import ValFallDetection

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def json_dict_or_file(v):
    if isinstance(v, dict):
        return v
    s = str(v).strip()
    p = Path(s)
    if p.exists():
        try:
            with p.open("r") as f:
                data = json.load(f)
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"Failed to read JSON file '{s}': {exc}")
    else:
        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected JSON object string or JSON file path for dict argument. Error: {exc}"
            )

    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object (dict).")
    return data


def load_run_config(path: str):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"run_config file not found: {config_path}")
    with config_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("run_config JSON must be an object (dict) at top level.")

    # Normalize dict-like fields so config can provide either a dict value
    # or a JSON object string / JSON file path.
    dict_fields = ("image_dir", "annotation_dir", "meta_data", "global_sources")
    for key in dict_fields:
        if key in data:
            data[key] = json_dict_or_file(data[key])
    return data


def build_split_sample_transforms(args):
    supported_tasks = {"detection", "classification", "segmentation", "instance_segmentation", "pose_estimation", "fall_detection", "object_tracking"}
    if args.task not in supported_tasks:
        raise ValueError(f"Unsupported task '{args.task}'. Choose one of: {sorted(supported_tasks)}")

    aug_task = args.task
    if aug_task in {"fall_detection", "object_tracking"}:
        aug_task = "detection"

    train_aug = build_augmentations(
        task=aug_task,
        split="train",
        enabled=args.aug_enable,
        hflip_p=args.aug_hflip_p,
        vflip_p=args.aug_vflip_p,
        affine_p=args.aug_affine_p,
        affine_degrees=args.aug_affine_degrees,
        affine_translate=args.aug_affine_translate,
        affine_scale_min=args.aug_affine_scale_min,
        affine_scale_max=args.aug_affine_scale_max,
        affine_min_box_size=args.aug_affine_min_box_size,
        color_p=args.aug_color_p,
        brightness=args.aug_brightness,
        contrast=args.aug_contrast,
        saturation=args.aug_saturation,
        hue=args.aug_hue,
        noise_p=args.aug_noise_p,
        noise_std=args.aug_noise_std,
    )
    return {"train": train_aug, "val": None, "test": None}


def main(args):
    if args.task not in {"detection", "instance_segmentation", "pose_estimation", "fall_detection", "object_tracking"}:
        raise NotImplementedError(
            "Current main.py model path supports only: detection, instance_segmentation, pose_estimation, fall_detection, object_tracking."
        )

    sample_transforms = build_split_sample_transforms(args)
    if args.dataset_name == "coco":
        dataset = COCO(
            image_dirs=args.image_dir,
            annotation_dirs=args.annotation_dir,
            class_file=args.meta_data["class_file"],
            data_file=args.meta_data["data_file"],
            batch_size=args.batch_size,
            split="train",
            out_dir=args.out_dir,
            with_ground_truth=True,
            sample_transforms=sample_transforms,
            resize_mode=args.resize_mode,
            image_size=args.image_size,
            mosaic_enable=args.mosaic_enable,
            mosaic_p=args.mosaic_p,
            multiscale=args.multiscale,
            multiscale_min_size=args.multiscale_min_size,
            multiscale_max_size=args.multiscale_max_size,
            multiscale_step=args.multiscale_step,
            multiscale_interval=args.multiscale_interval,
            use_weighted_sampler=args.use_weighted_sampler,
            sampler_power=args.sampler_power,
            sampler_min_weight=args.sampler_min_weight,
        )
        
        if not Path(f"{args.out_dir}/train.txt").exists() or not Path(f"{args.out_dir}/val.txt").exists():
            dataset.coco_to_yolo_call(out_dir=args.out_dir)

    elif args.dataset_name == "voc":
        dataset = VOC(image_dirs=args.image_dir,
            annotation_dirs=args.annotation_dir,
            class_file=args.meta_data["class_file"],
            data_file=args.meta_data["data_file"],
            batch_size=args.batch_size,
            split="train",
            out_dir=args.out_dir,
            with_ground_truth=True,
            sample_transforms=sample_transforms,
            resize_mode=args.resize_mode,
            image_size=args.image_size,
            mosaic_enable=args.mosaic_enable,
            mosaic_p=args.mosaic_p,
            multiscale=args.multiscale,
            multiscale_min_size=args.multiscale_min_size,
            multiscale_max_size=args.multiscale_max_size,
            multiscale_step=args.multiscale_step,
            multiscale_interval=args.multiscale_interval,
            use_weighted_sampler=args.use_weighted_sampler,
            sampler_power=args.sampler_power,
            sampler_min_weight=args.sampler_min_weight,
        )

        train_txt = Path(f"{args.out_dir}/train.txt")
        val_txt = Path(f"{args.out_dir}/val.txt")
        obj_dir = Path(f"{args.out_dir}/obj")
        should_build_voc = (
            not train_txt.exists()
            or not val_txt.exists()
            or train_txt.stat().st_size == 0
            or val_txt.stat().st_size == 0
            or not obj_dir.exists()
        )

        if should_build_voc:
            dataset.voc_to_yolo_call(out_dir=args.out_dir)

    elif args.dataset_name == "global":
        dataset = GlobalDataset(
            source_datasets=args.global_sources,
            class_file=args.meta_data["class_file"],
            data_file=args.meta_data["data_file"],
            batch_size=args.batch_size,
            split="train",
            out_dir=args.out_dir,
            with_ground_truth=True,
            sample_transforms=sample_transforms,
            resize_mode=args.resize_mode,
            image_size=args.image_size,
            mosaic_enable=args.mosaic_enable,
            mosaic_p=args.mosaic_p,
            multiscale=args.multiscale,
            multiscale_min_size=args.multiscale_min_size,
            multiscale_max_size=args.multiscale_max_size,
            multiscale_step=args.multiscale_step,
            multiscale_interval=args.multiscale_interval,
            use_weighted_sampler=args.use_weighted_sampler,
            sampler_power=args.sampler_power,
            sampler_min_weight=args.sampler_min_weight,
        )

        train_txt = Path(f"{args.out_dir}/train.txt")
        val_txt = Path(f"{args.out_dir}/val.txt")
        obj_dir = Path(f"{args.out_dir}/obj")
        should_build_global = (
            args.global_force_rebuild
            or not train_txt.exists()
            or not val_txt.exists()
            or train_txt.stat().st_size == 0
            or val_txt.stat().st_size == 0
            or not obj_dir.exists()
        )
        if should_build_global:
            dataset.build_global_dataset(
                out_dir=args.out_dir,
                clear_output=args.global_clear_output,
                copy_images=args.global_copy_images,
            )
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")
    

    dataset.get_stats_for_split(dataset.dataset_root, 'train')
    dataset.get_stats_for_split(dataset.dataset_root, 'val')

    if args.task in {"detection", "instance_segmentation"}:
        if args.model_name == "ipc8":
            yolo = NYOLO(
                args.model_name,
                args.model_size,
                num_classes=dataset.class_num,
                task=args.task,
                num_masks=args.seg_num_masks,
                num_protos=args.seg_num_protos,
            )
            yolo.build()
            train = TYOLO(yolo, dataset, learning_rate=args.learning_rate, 
                            cost_function_name='yolo_cost_function', 
                            isScheduled=args.isScheduled, 
                            scheduler_name=args.scheduler_name, 
                            total_steps=args.epoch_num,
                            pct_start=0.1,
                            final_div_factor=100,
                            subdivisions=args.subdivisions,
                            use_amp=args.use_amp,
                            class_balance=args.class_balance,
                            class_balance_power=args.class_balance_power,
                            class_balance_min=args.class_balance_min,
                            class_balance_max=args.class_balance_max,
                            class_balance_eps=args.class_balance_eps,
                            pretrained_weights=args.pretrained_weights,
                            seg_weight=args.seg_weight,
                            mask_bce_weight=args.mask_bce_weight,
                            mask_dice_weight=args.mask_dice_weight,
                        )
            train.start(f'{args.model_backup}/{args.saved_name}', epoch_num=args.epoch_num, report_epoch=10, resume=args.resume)

            if args.export_onnx:
                best_ckpt = torch.load(f'{args.model_backup}/{args.saved_name}_best.pt')
                direction, remapped = load_state_dict_compat(yolo.model, best_ckpt['model_state_dict'], strict=True)
                if remapped > 0:
                    print(f"Checkpoint key remap applied: {direction} ({remapped} tensors)")
                yolo.export_onnx(
                    f"{args.model_backup}/{args.saved_name}_best.onnx",
                    f"{args.model_backup}/{args.saved_name}_best.pt",
                    input_size=(args.image_size, args.image_size),
                    metadata_json_path=f"{args.model_backup}/out.json",
                    export_segmentation=args.export_segmentation_onnx,
                )

            yolo.build_predictor(dataset.target_classes, max_input_size=args.image_size)
        
        val = ValDetectionSegmentation(yolo.predictor, dataset, task=args.task, split="val")
        val.start()
    elif args.task == "pose_estimation":
        media_pipe = NMediaPipe(
            name=args.model_name,
            task=args.task,
            model_path=args.model_path,
        )
        media_pipe.build()
        media_pipe.build_predictor()

        val = ValPoseEstimation(args.model_name, media_pipe.predictor, dataset)
        val.test_performance('dataset/custom_coco_yolo/pose_estimation_outputs', source='dataset/custom_coco_yolo/videos/saved_frames', save_visuals=True, save_txt=True)
        print("Pose estimation task is not yet implemented in main.py.")
    elif args.task == "fall_detection":
        if args.model_name == "ipc8":
            yolo = NYOLO(
                args.model_name,
                args.model_size,
                num_classes=dataset.class_num,
                task="detection",
                num_masks=args.seg_num_masks,
                num_protos=args.seg_num_protos,
            )
            yolo.build()

            if args.pretrained_weights:
                ckpt = torch.load(args.pretrained_weights, map_location="cpu")
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    direction, remapped = load_state_dict_compat(yolo.model, ckpt["model_state_dict"], strict=True)
                else:
                    direction, remapped = load_state_dict_compat(yolo.model, ckpt, strict=True)
                if remapped > 0:
                    print(f"Checkpoint key remap applied: {direction} ({remapped} tensors)")

            yolo.build_predictor(dataset.target_classes, max_input_size=args.image_size)
        else:
            raise ValueError(f"Unsupported model for fall detection: {args.model_name}")

        if not args.fall_source:
            raise ValueError("fall_detection requires --fall_source to be set.")

        val = ValFallDetection(
            yolo.predictor,
            dataset,
            person_class=args.fall_person_class,
            conf_tresh=args.fall_conf_tresh,
            iou_tresh=args.fall_iou_tresh,
            track_iou=args.fall_track_iou,
            max_missed=args.fall_max_missed,
            history=args.fall_history,
            aspect_thresh=args.fall_aspect_thresh,
            height_ratio=args.fall_height_ratio,
            drop_ratio=args.fall_drop_ratio,
            drop_window=args.fall_drop_window,
            persist_frames=args.fall_persist_frames,
            recover_frames=args.fall_recover_frames,
            tracker_type=args.fall_tracker_type,
            byte_high=args.fall_byte_high,
            byte_low=args.fall_byte_low,
            use_hungarian=args.fall_use_hungarian,
            use_motion=args.fall_use_motion,
        )
        val.test_performance(
            output_dir=args.fall_output_dir,
            source=args.fall_source,
            save_visuals=args.fall_save_visuals,
            save_txt=args.fall_save_txt,
        )
    elif args.task == "object_tracking":
        if args.model_name == "ipc8":
            yolo = NYOLO(
                args.model_name,
                args.model_size,
                num_classes=dataset.class_num,
                task="detection",
                num_masks=args.seg_num_masks,
                num_protos=args.seg_num_protos,
            )
            yolo.build()

            if args.pretrained_weights:
                ckpt = torch.load(args.pretrained_weights, map_location="cpu")
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    direction, remapped = load_state_dict_compat(yolo.model, ckpt["model_state_dict"], strict=True)
                else:
                    direction, remapped = load_state_dict_compat(yolo.model, ckpt, strict=True)
                if remapped > 0:
                    print(f"Checkpoint key remap applied: {direction} ({remapped} tensors)")

            yolo.build_predictor(dataset.target_classes, max_input_size=args.image_size)
        else:
            raise ValueError(f"Unsupported model for object tracking: {args.model_name}")

        if not args.track_source:
            raise ValueError("object_tracking requires --track_source to be set.")

        val = ValObjectTracking(
            yolo.predictor,
            dataset,
            tracker_type=args.track_type,
            conf_tresh=args.track_conf_tresh,
            iou_tresh=args.track_iou_tresh,
            track_iou=args.track_track_iou,
            max_missed=args.track_max_missed,
            use_hungarian=args.track_use_hungarian,
            use_motion=args.track_use_motion,
            byte_high=args.track_byte_high,
            byte_low=args.track_byte_low,
        )
        val.test_performance(
            output_dir=args.track_output_dir,
            source=args.track_source,
            save_visuals=args.track_save_visuals,
            save_txt=args.track_save_txt,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--run_config", type=str, default="json_run.json")
    
    parser.add_argument("--model_name", type=str, default="ipc8")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--saved_name", type=str, default="checkpoint")
    parser.add_argument("--export_onnx", type=str2bool, default=True)
    parser.add_argument("--export_segmentation_onnx", type=str2bool, default=False)
    parser.add_argument("--task", type=str, default="detection")
    parser.add_argument("--seg_num_masks", type=int, default=32)
    parser.add_argument("--seg_num_protos", type=int, default=32)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--mask_bce_weight", type=float, default=1.0)
    parser.add_argument("--mask_dice_weight", type=float, default=1.0)
    parser.add_argument("--dataset_name", type=str, default="global")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--image_dir",
        type=json_dict_or_file,
        default={
            "train": "/home/orchid/ipc_yolo/dataset/coco2017/train2017",
            "valid": "/home/orchid/ipc_yolo/dataset/coco2017/val2017",
            "test": "/home/orchid/ipc_yolo/dataset/coco2017/test2017",
        },
    )
    parser.add_argument(
        "--annotation_dir",
        type=json_dict_or_file,
        default={
            "train": "/home/orchid/ipc_yolo/dataset/coco2017/annotations",
            "valid": "/home/orchid/ipc_yolo/dataset/coco2017/annotations",
            "test": "/home/orchid/ipc_yolo/dataset/coco2017/annotations",
        },
    )
    parser.add_argument("--epoch_num", type=int, default=100)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--model_size", type=str, default="n")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--isScheduled", type=str2bool, default=True)
    parser.add_argument("--scheduler_name", type=str, default="OneCycleLR")
    parser.add_argument(
        "--meta_data",
        type=json_dict_or_file,
        default={
            "class_file": "/home/orchid/PythonProject/dataset/custom_coco_yolo/object.names",
            "data_file": "/home/orchid/PythonProject/dataset/custom_coco_yolo/obj.data",
        },
    )
    parser.add_argument("--out_dir", type=str, default="/home/orchid/PythonProject/dataset/custom_coco_yolo")
    parser.add_argument("--model_backup", type=str, default="/home/orchid/PythonProject/dataset/custom_coco_yolo/backups")
    
    parser.add_argument("--resize_mode", type=str, default="letterbox")
    parser.add_argument("--image_size", type=int, default=608)
    parser.add_argument("--multiscale", type=str2bool, default=True)
    parser.add_argument("--multiscale_min_size", type=int, default=320)
    parser.add_argument("--multiscale_max_size", type=int, default=608)
    parser.add_argument("--multiscale_step", type=int, default=32)
    parser.add_argument("--multiscale_interval", type=int, default=256)
    parser.add_argument("--mosaic_enable", type=str2bool, default=False)
    parser.add_argument("--mosaic_p", type=float, default=0.0)

    parser.add_argument("--use_weighted_sampler", type=str2bool, default=False)
    parser.add_argument("--sampler_power", type=float, default=1.0)
    parser.add_argument("--sampler_min_weight", type=float, default=1e-3)
    
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--class_balance", type=str2bool, default=True)
    parser.add_argument("--class_balance_power", type=float, default=1.0)
    parser.add_argument("--class_balance_min", type=float, default=0.25)
    parser.add_argument("--class_balance_max", type=float, default=4.0)
    parser.add_argument("--class_balance_eps", type=float, default=1e-6)
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="/home/orchid/PythonProject/dataset/custom_coco_yolo/backups/yolov8n.pt",
    )
    
    parser.add_argument("--aug_enable", type=str2bool, default=False)
    parser.add_argument("--aug_hflip_p", type=float, default=0.5)
    parser.add_argument("--aug_vflip_p", type=float, default=0.0)
    parser.add_argument("--aug_affine_p", type=float, default=0.0)
    parser.add_argument("--aug_affine_degrees", type=float, default=10.0)
    parser.add_argument("--aug_affine_translate", type=float, default=0.1)
    parser.add_argument("--aug_affine_scale_min", type=float, default=0.9)
    parser.add_argument("--aug_affine_scale_max", type=float, default=1.1)
    parser.add_argument("--aug_affine_min_box_size", type=float, default=2.0)
    parser.add_argument("--aug_color_p", type=float, default=0.6)
    parser.add_argument("--aug_brightness", type=float, default=0.2)
    parser.add_argument("--aug_contrast", type=float, default=0.2)
    parser.add_argument("--aug_saturation", type=float, default=0.2)
    parser.add_argument("--aug_hue", type=float, default=0.05)
    parser.add_argument("--aug_noise_p", type=float, default=0.0)
    parser.add_argument("--aug_noise_std", type=float, default=0.02)

    parser.add_argument("--fall_source", type=str, default="")
    parser.add_argument("--fall_output_dir", type=str, default="/home/orchid/PythonProject/dataset/global/fall_outputs")
    parser.add_argument("--fall_save_visuals", type=str2bool, default=True)
    parser.add_argument("--fall_save_txt", type=str2bool, default=True)
    parser.add_argument("--fall_person_class", type=str, default="person")
    parser.add_argument("--fall_conf_tresh", type=float, default=0.25)
    parser.add_argument("--fall_iou_tresh", type=float, default=0.5)
    parser.add_argument("--fall_track_iou", type=float, default=0.3)
    parser.add_argument("--fall_max_missed", type=int, default=10)
    parser.add_argument("--fall_history", type=int, default=30)
    parser.add_argument("--fall_aspect_thresh", type=float, default=1.2)
    parser.add_argument("--fall_height_ratio", type=float, default=0.6)
    parser.add_argument("--fall_drop_ratio", type=float, default=0.04)
    parser.add_argument("--fall_drop_window", type=int, default=10)
    parser.add_argument("--fall_persist_frames", type=int, default=10)
    parser.add_argument("--fall_recover_frames", type=int, default=15)
    parser.add_argument("--fall_tracker_type", type=str, default="byte")
    parser.add_argument("--fall_byte_high", type=float, default=0.5)
    parser.add_argument("--fall_byte_low", type=float, default=0.1)
    parser.add_argument("--fall_use_hungarian", type=str2bool, default=True)
    parser.add_argument("--fall_use_motion", type=str2bool, default=True)

    parser.add_argument("--track_source", type=str, default="")
    parser.add_argument("--track_output_dir", type=str, default="/home/orchid/PythonProject/dataset/global/track_outputs")
    parser.add_argument("--track_save_visuals", type=str2bool, default=True)
    parser.add_argument("--track_save_txt", type=str2bool, default=True)
    parser.add_argument("--track_type", type=str, default="byte")
    parser.add_argument("--track_conf_tresh", type=float, default=0.25)
    parser.add_argument("--track_iou_tresh", type=float, default=0.5)
    parser.add_argument("--track_track_iou", type=float, default=0.3)
    parser.add_argument("--track_max_missed", type=int, default=10)
    parser.add_argument("--track_use_hungarian", type=str2bool, default=True)
    parser.add_argument("--track_use_motion", type=str2bool, default=True)
    parser.add_argument("--track_byte_high", type=float, default=0.5)
    parser.add_argument("--track_byte_low", type=float, default=0.1)
    
    parser.add_argument(
        "--global_sources",
        type=json_dict_or_file,
        default={
            "coco": {
                "root": "/home/orchid/PythonProject/dataset/custom_coco_yolo",
                # Optional: image-level trigger classes.
                "include_classes": ["person", "car", "motorbike", "truck", "bicycle"],
            },
            "voc": {
                "root": "/home/orchid/PythonProject/dataset/VOC",
                # Optional: per-source split mapping when names differ.
                "split_map": {"train": "train", "val": "valid"},
                "include_classes": ["car", "motorbike", "bicycle"]
            },
        },
    )
    parser.add_argument("--global_force_rebuild", type=str2bool, default=False)
    parser.add_argument("--global_clear_output", type=str2bool, default=False)
    parser.add_argument("--global_copy_images", type=str2bool, default=False)

    pre_args, _ = parser.parse_known_args()
    run_cfg = load_run_config(pre_args.run_config)
    parser.set_defaults(**run_cfg)
    args = parser.parse_args()
    main(args)
