import os
import numpy as np
import cv2
from PIL import Image

def read_im(im_path):
    # 读取图片并Resize，保持统一大小用于拼接
    im = np.asarray(Image.open(im_path).convert('RGB'))
    resize_h_w = (256, 128) # 你的项目图片尺寸
    if (im.shape[0], im.shape[1]) != resize_h_w:
        im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    return im

def add_border(im, border_width, value):
    # 给图片加边框 (绿=对, 红=错)
    if im.ndim == 2:
        im = np.stack((im,)*3, axis=-1)
    
    # Pad borders
    im = np.pad(im, ((border_width, border_width), (border_width, border_width), (0, 0)), 
                mode='constant', constant_values=0)
    
    # Color the border
    im[:border_width, :, :] = value
    im[-border_width:, :, :] = value
    im[:, :border_width, :] = value
    im[:, -border_width:, :] = value
    return im

def make_im_grid(ims, n_rows, n_cols, space, pad_val):
    # 将多张图片拼接成网格
    h, w = ims[0].shape[:2]
    H = h * n_rows + space * (n_rows - 1)
    W = w * n_cols + space * (n_cols - 1)
    
    # Canvas
    ret_im = (np.ones([H, W, 3]) * pad_val).astype(np.uint8)
    
    for n, im in enumerate(ims):
        r = n // n_cols
        c = n % n_cols
        h1 = r * (h + space)
        h2 = h1 + h
        w1 = c * (w + space)
        w2 = w1 + w
        ret_im[h1:h2, w1:w2, :] = im
    return ret_im

def visualize_ranked_results(dist_vec, q_id, q_cam, q_img_path, g_ids, g_cams, g_img_paths, save_path, top_k=5):
    """
    生成单张Query的Ranking List图片
    """
    # 1. 排序
    sort_inds = np.argsort(dist_vec) # 距离越小越相似
    
    # 2. 收集可视化的图片
    ims = [add_border(read_im(q_img_path), 5, [255, 255, 255])] # Query图片用白框
    
    match_count = 0
    rank_count = 0
    
    for ind in sort_inds:
        # 跳过 junk (相同的ID且相同的相机)
        if (q_id == g_ids[ind]) and (q_cam == g_cams[ind]):
            continue
            
        rank_count += 1
        is_match = (q_id == g_ids[ind])
        
        # 绿色: 正确, 红色: 错误
        color = [0, 255, 0] if is_match else [255, 0, 0]
        
        g_im = read_im(g_img_paths[ind])
        g_im = add_border(g_im, 5, color)
        ims.append(g_im)
        
        if rank_count >= top_k:
            break
            
    # 3. 拼接图片
    grid_im = make_im_grid(ims, 1, len(ims), space=5, pad_val=255)
    
    # 4. 保存
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    Image.fromarray(grid_im).save(save_path)