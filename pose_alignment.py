import cv2
from pathlib import Path
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pyrender
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hmr2.utils.semantic_renderer import SemanticRenderer
from hmr2.utils.mesh_renderer import MeshRenderer
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

global HMR2_MODEL, HMR2_MODEL_CFG, DETECTRON2_MODEL
HMR2_MODEL = None
HMR2_MODEL_CFG = None
DETECTRON2_MODEL = None


def get_smpl_result(img_cv2, HMR2_MODEL, HMR2_MODEL_CFG, DETECTRON2_MODEL):
    det_out = DETECTRON2_MODEL(img_cv2)

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    
    # Run HMR2.0 on all detected humans
    dataset = ViTDetDataset(HMR2_MODEL_CFG, img_cv2, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []
    
    for batch in dataloader:
        batch = recursive_to(batch, args.device)
        with torch.no_grad():
            out = HMR2_MODEL(batch)

        pred_cam = out['pred_cam']
        pred_smpl_parameter = out['pred_smpl_params']
        
        smpl_output = HMR2_MODEL.smpl(**{k: v.float() for k,v in pred_smpl_parameter.items()}, pose2rot=False)
        pred_vertices = smpl_output.vertices
        out['pred_vertices'] = pred_vertices.reshape(batch['img'].shape[0], -1, 3)
        
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        
        scaled_focal_length = HMR2_MODEL_CFG.EXTRA.FOCAL_LENGTH / HMR2_MODEL_CFG.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        # Render the result
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            # Add all verts and cams to list
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)

        smpl_outs = {k: v.detach().cpu().numpy() for k, v in pred_smpl_parameter.items()}
        results_dict_for_rendering = {"verts":all_verts, "cam_t":all_cam_t, 
            "render_res":img_size[n].cpu().numpy(), "smpls":smpl_outs,
            "scaled_focal_length":scaled_focal_length.cpu().numpy()}
        return results_dict_for_rendering


def align_pose(reference_img_cv2, driving_pose_img_cv2):
    global HMR2_MODEL, HMR2_MODEL_CFG, DETECTRON2_MODEL

    if HMR2_MODEL is None:
        HMR2_MODEL, HMR2_MODEL_CFG = load_hmr2(DEFAULT_CHECKPOINT)
        HMR2_MODEL = HMR2_MODEL.to(args.device)

    if DETECTRON2_MODEL is None:
        # Load detector
        from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = '/mnt/petrelfs/liuwenran/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl'
        # detectron2_cfg.train.init_checkpoint = str(Path(hmr2.__file__).parent.parent/'detectron2'/"model_final_f05665.pkl")
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        DETECTRON2_MODEL = DefaultPredictor_Lazy(detectron2_cfg)
        DETECTRON2_MODEL.model.to(args.device)

    reference_dict = get_smpl_result(reference_img_cv2, HMR2_MODEL, HMR2_MODEL_CFG, DETECTRON2_MODEL)
    driving_dict = get_smpl_result(driving_pose_img_cv2, HMR2_MODEL, HMR2_MODEL_CFG, DETECTRON2_MODEL)

    result_dict = {key: value for key, value in driving_dict.items()}
    result_dict["smpls"] = driving_dict['smpls']

    # if args.view_transfer:
    result_dict["cam_t"] = reference_dict["cam_t"]
    result_dict["scaled_focal_length"] = reference_dict["scaled_focal_length"]
    result_dict["smpls"]['betas'] = reference_dict['smpls']['betas']
    # transfer reference SMPL shape to driving SMPLs
    # if args.figure_transfer:
        
    smpl_output = HMR2_MODEL.smpl(**{k: torch.Tensor(v[[0]]).to(args.device).float() for k,v in result_dict["smpls"].items()}, pose2rot=False)
    pred_vertices = smpl_output.vertices
    result_dict["verts"][0] = pred_vertices.reshape(-1, 3).detach().cpu().numpy()
    result_dict["render_res"] =  reference_dict["render_res"]

    # render driving pose to reference img
    renderer = Renderer(HMR2_MODEL_CFG, faces=HMR2_MODEL.smpl.faces)
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=result_dict["scaled_focal_length"],
    )
    cam_view = renderer.render_rgba_multiple(result_dict["verts"], cam_t=result_dict["cam_t"], render_res=[reference_img_cv2.shape[1], reference_img_cv2.shape[0]], **misc_args)

    # Overlay image
    input_img = reference_img_cv2.astype(np.float32)[:,:,::-1]/255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
    input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

    output_folder = '/mnt/petrelfs/liuwenran/forks/4D-Humans/input/align_result'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder,"images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder,"smpl_results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder,"mesh_view"), exist_ok=True)
        
    cv2.imwrite(os.path.join(output_folder, 'images', 'input_img_overlay.png'), 255*input_img_overlay[:, :, ::-1])

    white_background_img = np.ones_like(input_img)
    white_background_img_overlay = white_background_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
    cv2.imwrite(os.path.join(output_folder, 'mesh_view', 'white_background_img_overlay.png'), 255*white_background_img_overlay[:, :, ::-1])

    np.save(str(os.path.join(output_folder, "smpl_results", 'smpl_aligned.npy')), result_dict)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--figure_scale', type=int, default=None, help='Adjust the figure scale to better fit extreme shape')

    # parser.add_argument('--transfer_shape_to_driving', type=bool, default=True, help='If True, transfer reference shape to driving smpl. Otherwise, transfer driving poses to reference shape to 3.')

    args = parser.parse_args()

    reference_img_path = '/mnt/petrelfs/liuwenran/forks/4D-Humans/input/tangxuanzong.jpg'
    reference_img_cv2 = cv2.imread(reference_img_path)

    driving_img_path = '/mnt/petrelfs/liuwenran/forks/4D-Humans/input/vroid_dance_0.png'
    driving_img_cv2 = cv2.imread(driving_img_path)


    align_pose(reference_img_cv2, driving_img_cv2)

    
    