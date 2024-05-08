import cv2
from pathlib import Path
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pyrender
from pathlib import Path

if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils.renderer import Renderer, cam_crop_to_full
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

if __name__ == "__main__":
    # Replace 'input_video_path' and 'output_video_path' with the actual paths to your video files
    #input_video_path = 'input_video.mp4'
    #output_video_path = '00337_transfer_test/output.mp4'
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    # parser.add_argument('--driving_path', type=str, default="driving_videos/001", help='Folder path to driving imgs sequence')
    # parser.add_argument('--reference_path', type=str, default="reference_imgs/images/ref.png", help='Path to reference img')
    # parser.add_argument('--output_folder', type=str, default="output", help='Path to result imgs')
    parser.add_argument('--figure_transfer', dest='figure_transfer', action='store_true', default=False, help='If true, transfer SMPL shape parameter.')
    parser.add_argument('--view_transfer', dest='view_transfer', action='store_true', default=False, help='If true, transfer camera parameter.')

    args = parser.parse_args()

    reference_img_file = '/mnt/petrelfs/share_data/liuwenran/mooreaa/testbench/all_imgs_human.txt'
    reference_img_lines = open(reference_img_file, 'r').read().splitlines()
    reference_smpl_path = '/mnt/petrelfs/share_data/liuwenran/mooreaa/testbench/images_smpl_results'

    driving_smpl_path = '/mnt/petrelfs/share_data/liuwenran/mooreaa/testbench/pose_raw_smpl_results/ubc_mv'
    driving_smpl_group_path = '/mnt/petrelfs/share_data/liuwenran/mooreaa/testbench/pose_raw_smpl_group_results/ubc_mv'
    output_root_folder = '/mnt/petrelfs/share_data/liuwenran/mooreaa/testbench/images_align_pose_raw_results'

    driving_paths = os.listdir(os.path.join(driving_smpl_path))
    driving_paths = [path for path in driving_paths if os.path.splitext(path)[1].lower() == ".npy"]
    driving_paths.sort(key=lambda x: int(x.split('.')[0]))
    driving_paths = [os.path.join(driving_smpl_path, path) for path in driving_paths]

    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(args.device)

    for line in reference_img_lines:
        img_name, _ = os.path.splitext(os.path.basename(line))
        driving_pose_name = driving_smpl_group_path.split('/')[-1]

        output_folder = os.path.join(output_root_folder, img_name, driving_pose_name)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder,"images"), exist_ok=True)
        os.makedirs(os.path.join(output_folder,"smpl_results"), exist_ok=True)
        os.makedirs(os.path.join(output_folder,"mesh_view"), exist_ok=True)
        
        reference_img = cv2.imread(line)
        reference_file = os.path.join(reference_smpl_path, img_name + '.npy')
        reference_dict = np.load(str(reference_file), allow_pickle=True).item()
        
        smooth_smpl_path = os.path.join(driving_smpl_group_path, "smpls_group.npz")
        if os.path.exists(smooth_smpl_path):
            result_dict_list = np.load(smooth_smpl_path, allow_pickle=True)
            result_dict_first = np.load(driving_paths[0], allow_pickle=True).item()
            for smpl_outs, cam_t, file_path in tqdm(zip(result_dict_list["smpl"], result_dict_list["camera"], driving_paths)):
                img_fn, _ = os.path.splitext(os.path.basename(file_path))
                result_dict = {key: value for key, value in result_dict_first.items()}
                result_dict["smpls"] = smpl_outs
                result_dict["cam_t"] = cam_t
                if args.view_transfer:
                    scaled_focal_length = reference_dict["scaled_focal_length"]
                    result_dict["cam_t"] = reference_dict["cam_t"]
                    result_dict["scaled_focal_length"] = scaled_focal_length
                # transfer reference SMPL shape to driving SMPLs
                if args.figure_transfer:
                    result_dict["smpls"]['betas'] = reference_dict['smpls']['betas']
                    
                smpl_output = model.smpl(**{k: torch.Tensor(v[[0]]).to(args.device).float() for k,v in result_dict["smpls"].items()}, pose2rot=False)
                pred_vertices = smpl_output.vertices
                result_dict["verts"][0] = pred_vertices.reshape(-1, 3).detach().cpu().numpy()
                result_dict["render_res"] =  reference_dict["render_res"]

                # render driving pose to reference img
                renderer = Renderer(model_cfg, faces=model.smpl.faces)
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(result_dict["verts"], cam_t=result_dict["cam_t"], render_res=[reference_img.shape[1], reference_img.shape[0]], **misc_args)

                # Overlay image
                input_img = reference_img.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                cv2.imwrite(os.path.join(output_folder, 'images', f'{img_fn}.png'), 255*input_img_overlay[:, :, ::-1])

                white_background_img = np.ones_like(input_img)
                white_background_img_overlay = white_background_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                cv2.imwrite(os.path.join(output_folder, 'mesh_view', f'{img_fn}.png'), 255*white_background_img_overlay[:, :, ::-1])

                np.save(str(os.path.join(output_folder, "smpl_results", f'{img_fn}.npy')),
                    result_dict)
                
                import ipdb;ipdb.set_trace();