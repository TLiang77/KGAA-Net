
import numpy as np
import torch
from torch.autograd import Variable
from tools import eval_regdb, eval_sysu, eval_llcm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.nn.functional as F
import time

def get_der_params(config):
    dataset = getattr(config, 'dataset', 'sysu')
    test_mode = getattr(config, 'test_mode', 'all')
    
    if dataset == 'sysu' and test_mode == 'indoor':
        return {
            'method': 'logit_weight',
            'temperature': 5.0,
            'prior_weights': [1.0, 1.0, 1.0],
            'sensitivity': 2.0,
            'top_k': 10,
            'entropy_func': 'original'
        }
    elif dataset == 'sysu' and test_mode == 'all':
        return {
            'method': 'logit_weight',
            'temperature': 10.0,
            'prior_weights': [1.0, 1.2, 1.5],
            'sensitivity': 2.0,
            'top_k': 10,
            'entropy_func': 'original'
        }
    elif dataset in ['regdb', 'llcm']:
        return {
            'method': 'logit_weight',
            'temperature': 10.0,
            'prior_weights': [1.0, 1.0, 1.0],
            'sensitivity': 5.0,
            'top_k': 10,
            'entropy_func': 'original'
        }
    else:
        return {
            'method': 'logit_weight',
            'temperature': 5.0,
            'prior_weights': [1.0, 1.0, 1.0],
            'sensitivity': 2.0,
            'top_k': 10,
            'entropy_func': 'original'
        }

def compute_entropy_original(dist_mat, temperature=10.0, top_k=10):
    sim_mat = -dist_mat 
    topk_sim, _ = torch.topk(torch.from_numpy(sim_mat), k=top_k, dim=1)
    probs = F.softmax(topk_sim * temperature, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy

def compute_entropy_normalized(dist_mat, temperature=1.0, top_k=10):
    sim_mat = -dist_mat 
    topk_sim, _ = torch.topk(torch.from_numpy(sim_mat), k=top_k, dim=1)
    probs = F.softmax(topk_sim / temperature, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    max_entropy = np.log(top_k)
    return entropy / max_entropy

def adaptive_ensemble_similarity_der(query_feats_dict, gallery_feat, device, config):

    params = get_der_params(config)
    
    dist_mats = {}
    entropies = {}
    gallery_norm = F.normalize(torch.from_numpy(gallery_feat).to(device), p=2, dim=1)
    
    for key, q_feat in query_feats_dict.items():
        q_norm = F.normalize(torch.from_numpy(q_feat).to(device), p=2, dim=1)
        dist = -torch.mm(q_norm, gallery_norm.t()).cpu().numpy()
        dist_mats[key] = dist
        
        if params['entropy_func'] == 'original':
            entropy = compute_entropy_original(dist, params['temperature'], params['top_k'])
        else:
            entropy = compute_entropy_normalized(dist, params['temperature'], params['top_k'])
        
        entropies[key] = entropy.to(device)
    
    entropy_stack = torch.stack([entropies['ir'], entropies['text'], entropies['fusion']], dim=1)
    
    if params['method'] == 'logit_weight':
        prior_weights = torch.tensor(params['prior_weights']).to(entropy_stack.device)
        prior_logits = torch.log(prior_weights)
        sensitivity = params['sensitivity']
        final_logits = prior_logits - (entropy_stack * sensitivity)
        adaptive_weights = F.softmax(final_logits, dim=1).cpu().numpy()
    else:
        scale = params['scale']
        raw_weights = torch.exp(-entropy_stack * scale)
        prior_bias = torch.tensor(params['prior_bias']).to(entropy_stack.device)
        weighted_weights = raw_weights * prior_bias
        adaptive_weights = (weighted_weights / torch.sum(weighted_weights, dim=1, keepdim=True)).cpu().numpy()
    final_dist = (adaptive_weights[:, 0:1] * dist_mats['ir'] + 
                  adaptive_weights[:, 1:2] * dist_mats['text'] + 
                  adaptive_weights[:, 2:3] * dist_mats['fusion'])
    
    return final_dist

def test(base, loader, config, device):
    embed_dim = base.embed_dim
    base.set_eval()

    print('Extracting Query Feature...')
    print('Test Mode: ', config.test_modality)
    result_dict = dict()
    
    assert loader.dataset in ['sysu', 'regdb', 'llcm'], 'Invalid dataset!'
    assert any(m in config.test_modality for m in ['IR', 'Fusion', 'Text']), 'Invalid test modality!'

    use_der = getattr(config, 'DER', False)

    total_extract_time = 0.0     # 特征提取总耗时 (s)
    total_match_time = 0.0       # 相似度计算与匹配/重加权总耗时 (s)
    
    # 清空 PyTorch CUDA 显存峰值统计器
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    ptr = 0
    if 'IR' in config.test_modality:
        query_feat_ir = np.zeros((loader.n_query, embed_dim))
    if 'Fusion' in config.test_modality:
        query_feat_fusion = np.zeros((loader.n_query, embed_dim))
        if config.CAT_EVAL or use_der:
            query_feat_ir_f = np.zeros((loader.n_query, embed_dim))
            query_feat_text_f = np.zeros((loader.n_query, embed_dim))
    if 'Text' in config.test_modality:
        query_feat_text = np.zeros((loader.n_query, embed_dim))
        
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(loader.query_loader):
            input = batch_dict['img'].to(device)
            batch_num = input.size(0)
            
            if 'Fusion' in config.test_modality or 'Text' in config.test_modality:
                text = batch_dict['text'].to(device).long()

            torch.cuda.synchronize(device)
            start_t = time.perf_counter()

            if 'IR' in config.test_modality:
                feat_ir_map = base.encode_image_featmap(input, 'ir')
                if config.Fix_Visual:
                    feat_ir = base.backup_pool(feat_ir_map).squeeze()
                    feat_ir = base.backup_classifier(feat_ir)
                else:
                    feat_ir = base.base_model.visual.__getattr__(config.pooling)(feat_ir_map).squeeze()
                    feat_ir = base.classifier(feat_ir)
                query_feat_ir[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                
            if 'Fusion' in config.test_modality:
                if config.Feat_Filter:
                    text_filter = batch_dict['text_filter'].to(device).long()
                    feat_fusion = base.encode_filtered_fusion(text, text_filter, input)
                else:
                    feat_fusion = base.encode_fusion(text, input, 'ir')
                    
                feat_fusion = base.classifier(feat_fusion, "Fusion")
                query_feat_fusion[ptr:ptr + batch_num, :] = feat_fusion.detach().cpu().numpy()
                
                if config.CAT_EVAL or use_der:
                    feat_text = base.encode_text_feat(text)
                    feat_text = base.classifier(feat_text)
                    query_feat_text_f[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()
                    
                    feat_ir = base.encode_image_feat(input, 'ir')
                    feat_ir = base.classifier(feat_ir)
                    query_feat_ir_f[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()

            if 'Text' in config.test_modality:
                feat_text = base.encode_text_feat(text)
                feat_text = base.classifier(feat_text, 'Text')
                query_feat_text[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()

            torch.cuda.synchronize(device)
            end_t = time.perf_counter()
            total_extract_time += (end_t - start_t)

            ptr += batch_num

    print('Extracting Gallery Feature...')

    if loader.dataset == 'sysu':
        all_cmc_ir, all_mAP_ir, all_mINP_ir = 0, 0, 0
        all_cmc_fusion, all_mAP_fusion, all_mINP_fusion = 0, 0, 0
        all_cmc_text, all_mAP_text, all_mINP_text = 0, 0, 0
        
        for i in range(10):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            if 'IR' in config.test_modality and config.Fix_Visual:
                gall_feat_for_IR = np.zeros((loader.n_gallery, embed_dim))
            if any(m in config.test_modality for m in ['IR', 'Fusion', 'Text']):
                gall_feat = np.zeros((loader.n_gallery, embed_dim))
                
            with torch.no_grad():
                for batch_idx, batch_dict in enumerate(gall_loader):
                    input = batch_dict['img'].to(device)
                    batch_num = input.size(0)

                    torch.cuda.synchronize(device)
                    start_t = time.perf_counter()

                    feat_map = base.encode_image_featmap(input, 'rgb')
                    if 'IR' in config.test_modality and config.Fix_Visual:
                        feat_for_IR = base.backup_pool(feat_map).squeeze()
                        feat_for_IR = base.backup_classifier(feat_for_IR)
                        gall_feat_for_IR[ptr:ptr + batch_num, :] = feat_for_IR.detach().cpu().numpy()
                        
                    if any(m in config.test_modality for m in ['IR', 'Fusion', 'Text']):
                        feat = base.base_model.visual.__getattr__(config.pooling)(feat_map).squeeze()
                        feat = base.classifier(feat)
                        gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    
                    torch.cuda.synchronize(device)
                    end_t = time.perf_counter()
                    total_extract_time += (end_t - start_t)

                    ptr += batch_num

            if 'Fusion' in config.test_modality:
                torch.cuda.synchronize(device)
                m_start = time.perf_counter()

                if use_der:
                    query_feats_dict = {
                        'ir': query_feat_ir_f,
                        'text': query_feat_text_f,
                        'fusion': query_feat_fusion
                    }
                    distmat_fusion = adaptive_ensemble_similarity_der(query_feats_dict, gall_feat, device, config)
                    
                elif config.CAT_EVAL:
                    sim_ir = np.matmul(query_feat_ir_f, np.transpose(gall_feat))
                    sim_text = np.matmul(query_feat_text_f, np.transpose(gall_feat))
                    sim_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat))
                    distmat_fusion = -(sim_ir + sim_text + sim_fusion) / 3.0
                    
                else:
                    distmat_fusion = -np.matmul(query_feat_fusion, np.transpose(gall_feat))

                torch.cuda.synchronize(device)
                m_end = time.perf_counter()
                total_match_time += (m_end - m_start)

                cmc_fusion, mAP_fusion, mINP_fusion = eval_sysu(distmat_fusion, loader.query_label, loader.gall_label, loader.query_cam, loader.gall_cam)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion

        max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
        
        avg_extract_latency_ms = (total_extract_time / (loader.n_query + loader.n_gallery * 10)) * 1000.0
        avg_matching_latency_ms = (total_match_time / (loader.n_query * 10)) * 1000.0

        mode_str = "DER (Ours)" if use_der else ("CAT_EVAL" if config.CAT_EVAL else "Baseline")
        print("\n" + "="*50)
        print(f" [Inference Cost Report] Mode: {mode_str}")
        print(f" - Peak GPU Memory Usage : {max_mem_mb:.2f} MB")
        print(f" - Total Extraction Time  : {total_extract_time:.4f} s (Avg: {avg_extract_latency_ms:.3f} ms/img)")
        print(f" - Total Matching/DER Time: {total_match_time:.4f} s (Avg: {avg_matching_latency_ms:.3f} ms/query)")
        print("="*50 + "\n")

        if 'Fusion' in config.test_modality:
            result_dict['Fusion'] = (all_mINP_fusion / 10.0, all_mAP_fusion / 10.0, all_cmc_fusion / 10.0)

    return result_dict