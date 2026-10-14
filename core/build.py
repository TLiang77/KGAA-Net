import os
from tools import os_walk
import network.clip_model.objectives as objectives
from network.clip_model.clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import TripletLoss_WRT,kl_align_loss,TripletLoss_WRT_local,L_i2t,L_t2i

    
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Classifier(nn.Module):
    def __init__(self, pid_num, dim=512, Return_B4_BN=False, uni_BN=False, joint_mode='uni',modal='RGB,IR,Text,Fusion'):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        # self.GAP = GeneralizedMeanPoolingP()
        self.Return_B4_BN = Return_B4_BN
        self.modal = modal
        self.uni_BN = uni_BN
        self.joint_mode = joint_mode
        if uni_BN:
            assert joint_mode == 'uni'
            if joint_mode == 'uni':
                self.BN_RGB = nn.BatchNorm1d(dim)
                self.BN_RGB.apply(weights_init_kaiming)
                self.BN_IR = nn.BatchNorm1d(dim)
                self.BN_IR.apply(weights_init_kaiming)
                self.BN_Fusion = nn.BatchNorm1d(dim)
                self.BN_Fusion.apply(weights_init_kaiming)
                self.BN_Text = nn.BatchNorm1d(dim)
                self.BN_Text.apply(weights_init_kaiming)
        else:
            self.BN = nn.BatchNorm1d(dim)
            self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features, mode="RGB"): # IR, Fusion, Text, RGB
        # features = self.GAP(features_map)
        if self.uni_BN:
            if self.training:
                len_feat = len(features)
                b = len_feat // 5
                rgb_features = self.BN_RGB(features[:2*b].squeeze())
                ir_features = self.BN_IR(features[2*b:3*b].squeeze())
                fusion_features = self.BN_Fusion(features[3*b:4*b].squeeze())
                text_features = self.BN_Text(features[4*b:5*b].squeeze())
                bn_features = torch.cat((rgb_features, ir_features, fusion_features, text_features),dim=0)
                
            else:
                if mode == 'RGB':
                    bn_features = self.BN_RGB(features.squeeze())
                elif mode == 'IR':
                    bn_features = self.BN_IR(features.squeeze())
                elif mode == 'Fusion':
                    bn_features = self.BN_Fusion(features.squeeze())
                elif mode == 'Text':
                    bn_features = self.BN_Text(features.squeeze())
                else:
                    raise ValueError("mode must be in ['IR', 'Fusion', 'Text', 'RGB']")
        else:
            bn_features = self.BN(features.squeeze())

        cls_score = self.classifier(bn_features)

        if self.training:
            return features, cls_score
        else:
            # if self.Return_B4_BN:
            #     return features
            return self.l2_norm(bn_features)


class FM_cat(nn.Module):
    def __init__(self,in_channels):
        super(FM_cat, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.normal_(self.W[1].weight.data, 1.0, 0.01)
        nn.init.zeros_(self.W[1].bias.data)

    def forward(self,f):

        f = f.view(f.size(0),f.size(1),1,1)
        f = self.W(f)
        f = f.view(f.size(0),-1)
        # f = self.bottleneck(f+feat)

        return f

class CLIP2ReID(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.max_save_model_num = args.max_save_model_num
        self.output_path = args.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')
        self._init_device()

        self.num_classes = num_classes

        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size,download_root="pretrain_cache/clip",prj_output_dim=self.args.prj_output_dim,pooling=self.args.pooling)
        self.embed_dim = base_cfg['embed_dim']
        if args.pretrain_choice == 'RN50':
            # 复制conv1...的权重到conv1_...
            print("copy conv1 weight to conv1_")
            self.base_model.visual.conv1_.load_state_dict(self.base_model.visual.conv1.state_dict())
    

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature))  # 0.07

        if 'attention' in args.fusion_way:
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)
        if 'global' in args.fusion_way:
            # self.dim_conv = nn.Linear(512*2,512)
            self.global_attn_s = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.global_attn_t = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
             # init cross attn
            scale = 512**-0.5
            proj_std = scale * ((2 * 4)**-0.5)
            attn_std = scale
            # fc_std = (2 * 512)**-0.5
            nn.init.normal_(self.global_attn_s.in_proj_weight, std=attn_std)
            nn.init.normal_(self.global_attn_s.out_proj.weight, std=proj_std)
             # init cross attn
            nn.init.normal_(self.global_attn_t.in_proj_weight, std=attn_std)
            nn.init.normal_(self.global_attn_t.out_proj.weight, std=proj_std)

        if 'attention' in args.fusion_way and 'global' not in args.fusion_way:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)

            scale = self.embed_dim ** -0.5
            

            proj_std = scale * ((2 * args.cmt_depth)**-0.5)
            attn_std = scale

            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        # Loss definition
        self.classifier = Classifier(self.num_classes,self.embed_dim,args.Return_B4_BN,args.uni_BN,args.joint_mode)
        self.pid_criterion = nn.CrossEntropyLoss()
        self.tri_criterion = TripletLoss_WRT()
        self.wrt_local = TripletLoss_WRT_local()
        self.attr_value_map_rgb = getattr(args, 'attr_value_map_rgb', {})
        self.attr_value_map_ir = getattr(args, 'attr_value_map_ir', {})

        # 假设 self.embed_dim 已存在
        self.attr_rel_list_rgb = getattr(args, 'attr_rel_list_rgb', [])  # list of rel names
        self.attr_rel_list_ir  = getattr(args, 'attr_rel_list_ir', [])

        self.attr_heads_rgb = nn.ModuleDict()
        for rel in self.attr_rel_list_rgb:
            num_cls = len(args.attr_value_map_rgb[rel])
            self.attr_heads_rgb[rel] = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.embed_dim//2, num_cls)
            )

        self.attr_heads_ir = nn.ModuleDict()
        for rel in self.attr_rel_list_ir:
            num_cls = len(args.attr_value_map_ir[rel])
            self.attr_heads_ir[rel] = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.embed_dim//2, num_cls)
            )
        self.attr_class_weights_rgb = {}
        for rel, vm in args.attr_value_map_rgb.items():
            counts = [0]*len(vm)
            # if you saved counts earlier you can fill them; else leave None
            self.attr_class_weights_rgb[rel] = None

        self.attr_criterion = nn.CrossEntropyLoss(reduction='mean')

        # === Attr loss controls ===
        self.attr_kl_t_student = float(getattr(args, 'attr_kl_t_student', 0.7))
        self.attr_kl_t_teacher = float(getattr(args, 'attr_kl_t_teacher', 1.0))
        self.attr_ce_weight = float(getattr(args, 'attr_ce_weight', 0.1))
        # Provide a non-zero default warmup to avoid early optimization conflicts
        self.attr_warmup_iters = int(getattr(args, 'attr_warmup_iters', 1500))  # curriculum warmup (by steps)
        self._global_step = 0

    def _init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu_id) if torch.cuda.is_available() else 'cpu')
        print('Model is using device: {}'.format(self.device))

    def set_train(self):
        self.train()
        self.training = True

    def set_eval(self):
        self.eval()
        self.training = False
    
    def save_model(self, save_epoch, is_best, mode='Fusion'): # mode = ['IR', 'Fusion', 'Text'] or their composition
        if is_best:
            model_file_path = os.path.join(self.save_model_path, f'model_{mode}_{save_epoch}.pth')
            if self.args.DataParallel:
                torch.save(self.module.state_dict(), model_file_path)
            else:
                torch.save(self.state_dict(), model_file_path)

        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            if mode in ['Fusion', 'IR', 'Text']:
                valid_files = []
                for i,file in enumerate(files):
                    if ('.pth' in file) and (mode in file):
                        valid_files.append(file)
                if len(valid_files) > self.max_save_model_num:
                    file_iters = sorted([int(file.replace('.pth', '').split('_')[-1]) for file in valid_files], reverse=False)
                    model_file_path = os.path.join(root, f'model_{mode}_{file_iters[0]}.pth')
                    os.remove(model_file_path)
            else:
                raise ValueError("savinf mode must be in ['Fusion', 'IR', 'Text']")


    def resume_last_model(self,mode='Fusion'):
        root, _, files = os_walk(self.save_model_path)
        valid_files = []
        for file in files:
            if ('pth' in file) and (mode in file):
                valid_files.append(file)

        if len(files) > 0:
            indexes = []
            for file in valid_files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1],mode)
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch, mode='Fusion'):
        model_path = os.path.join(self.save_model_path, f'model_{mode}_{resume_epoch}.pth')
        print('Resume model from {}'.format(model_path))
        self.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume model from {}'.format(model_path))


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split(',')]
        print(f'Training Model with {self.current_task} tasks')
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = q + x 
        x = self.ln_post(x)
        return x
    
    def global_former_s(self, q, k, v):
        x = self.global_attn_s(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        
        x = q + x 

        x = self.ln_post(x)
        return x

    def global_former_t(self, q, k, v):
        x = self.global_attn_t(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = q + x 

        x = self.ln_post(x)
        return x

    def encode_image_featmap(self, image, mode=None): 
        x = self.base_model.encode_image(image,mode=mode)
        return x
        # return x.float() # for CLIP ResNet visual model

    def encode_text_featmap(self, text):
        x = self.base_model.encode_text(text)
        return x #[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def encode_image_feat(self, image, mode=None): # return [B, 512]
        x = self.base_model.encode_image(image,mode=mode)
        if 'RN' in self.args.pretrain_choice:
            return self.base_model.visual.__getattr__(self.args.pooling)(x).float().squeeze()
        elif "ViT" in self.args.pretrain_choice:
            return x[:, 0, :].float()
        else:
            raise ValueError("pretrain_choice must be in ['RN50', 'RN50_ORI', 'ViT-B/16']")

    def encode_text_feat(self, text): # return [B, 512]
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def encode_fusion(self, text, ir, mode='ir'):
        caption_ids = text 
        text = self.encode_text_featmap(text)
        ir = self.encode_image_featmap(ir,mode=mode)
        x = self.fusion_layer(text,ir,caption_ids,pa=self.args.pa, way=self.args.fusion_way)
        return x.float()
    
    def encode_filtered_fusion(self, text, filter, ir):
        caption_ids = text 
        filter_caption_ids = filter
        text_feat = self.encode_text_feat(caption_ids)
        filter_text_feat = self.encode_text_feat(filter_caption_ids)
        ir = self.encode_image_feat(ir,mode='ir')
        x = ir + text_feat - filter_text_feat
        return x.float()

    def text_fusion_layer(self, text_rgb_map, text_ir_map, caption_rgb_ids, caption_ir_ids, way='add'):
        if way == 'add':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = text_rgb_feats + text_ir_feats
        return f_text_feats

    def fusion_layer(self, text_map, ir_map, caption_ids, pa=0.1, way='add'):
        text_feats = text_map[torch.arange(text_map.shape[0]), caption_ids.argmax(dim=-1)]
        if 'RN' in self.args.pretrain_choice:
            ir_feats = self.base_model.visual.__getattr__(self.args.pooling)(ir_map).squeeze()
        if 'VIT' in self.args.pretrain_choice:
            ir_feats = ir_map[:, 0, :]
        if way == 'add':        
            f_feats = text_feats + ir_feats
            
        return f_feats.float()

    def forward(self, batch_dict, mode=None):
        # get data
        rgb_imgs0 = batch_dict['img_rgb_ori']
        rgb_imgs1 = batch_dict['img_rgb_aug']
        ir_imgs = batch_dict['img_ir']
        label_rgb = batch_dict['target_rgb']
        label_ir = batch_dict['target_ir']

        # init return dict
        ret = dict()

        # get feature map
        if self.args.Fix_Visual:
            with torch.no_grad():
                image_feats_map = self.base_model.encode_image(torch.cat((rgb_imgs0,rgb_imgs1,ir_imgs),dim=0), mode)
        else:
            image_feats_map = self.base_model.encode_image(torch.cat((rgb_imgs0,rgb_imgs1,ir_imgs),dim=0), mode)
        b = ir_imgs.size(0)


        if 'RN' in self.args.pretrain_choice:     # img: [64 + 32, 1024] text:[64, 77, 1024]
            rgb_feats_map = image_feats_map[:int(2*b)]
            ir_feats_map = image_feats_map[int(2*b):]
            if self.args.Fix_Visual:
                with torch.no_grad():
                    rgb_feats = self.base_model.visual.__getattr__(self.args.pooling)(rgb_feats_map).float().squeeze()
                    ir_feats = self.base_model.visual.__getattr__(self.args.pooling)(ir_feats_map).float().squeeze()
            else:
                rgb_feats = self.base_model.visual.__getattr__(self.args.pooling)(rgb_feats_map).float().squeeze()
                ir_feats = self.base_model.visual.__getattr__(self.args.pooling)(ir_feats_map).float().squeeze()

        elif "ViT" in self.args.pretrain_choice:  # img: [64 + 32, 193, 512] text:[64, 77, 512]
            rgb_feats_map = image_feats_map[:int(2*b),:,:]
            ir_feats_map = image_feats_map[int(2*b):,:,:] 
            rgb_feats = rgb_feats_map[:, 0, :].float()
            ir_feats = ir_feats_map[:, 0, :].float()
        else:
            raise ValueError("pretrain_choice must be in ['RN50', 'RN50_ORI', 'ViT-B/16']")

        
        logit_scale = self.logit_scale.exp()
        ret.update({'temperature': 1 / logit_scale})

           
        loss_list = self.args.loss_names.split(',')

        if self.args.training_mode == 'RGB_IR_Text': # 如果有文本信息辅助
            if self.args.fusion_way in ['norm_add', 'add', 'cross_attention', 'attention_rgb_text', 'attention_ir_text', 'global_attention', 'global_attention_rgb_text', 'global_attention_ir_text']:
                
                # 获取rgb图像特征
                ori_vi_feats = rgb_feats[:int(b)]
                aug_vi_feats = rgb_feats[int(b):]
                
                if self.args.joint_mode == 'uni':
                    text_rgb = batch_dict['text_rgb']
                    text_rgb_feats_map = self.base_model.encode_text(text_rgb)

                    # 获取融合后的特征
                    if self.args.Feat_Filter:
                        text_filter = batch_dict['text_ir']
                        text_filter_feats = self.encode_text_feat(text_filter)
                        t_feats = text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)]
                        f_feats = (ir_feats + t_feats - text_filter_feats).squeeze()
                        t_feats = t_feats.float()

                    else:
                        f_feats = self.fusion_layer(text_rgb_feats_map, ir_feats_map, text_rgb, pa=self.args.pa, way=self.args.fusion_way).squeeze()
                        # 获取文本特征
                        t_feats = text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)].float() #[64, 512]
                    # === 属性一致性 Loss ===
                    if 'attr_rgb_tensor' in batch_dict and 'attr_ir_tensor' in batch_dict:
                        attr_tensor_rgb = batch_dict['attr_rgb_tensor'].to(t_feats.device)  # [B, R_rgb]
                        attr_tensor_ir  = batch_dict['attr_ir_tensor'].to(ir_feats.device)  # [B, R_ir]

                        attr_losses = []
                        attr_ce_losses = []
                        for i, rel in enumerate(self.attr_rel_list_rgb):
                            if rel not in self.attr_heads_rgb or rel not in self.attr_heads_ir:
                                continue
                            head_rgb = self.attr_heads_rgb[rel]
                            head_ir  = self.attr_heads_ir[rel]

                            pred_rgb = head_rgb(t_feats)   # [B, C]
                            pred_ir  = head_ir(ir_feats)   # [B, C]

                            T_s = max(self.attr_kl_t_student, 1e-6)
                            T_t = max(self.attr_kl_t_teacher, 1e-6)

                            # 1. 正向蒸馏: Text <- IR
                            log_p_rgb_T = F.log_softmax(pred_rgb / T_s, dim=-1)
                            p_ir_T = F.softmax(pred_ir / T_t, dim=-1).detach()
                            kl_text_from_ir = F.kl_div(log_p_rgb_T, p_ir_T, reduction="batchmean") * (T_s * T_s)
                            kl_text_from_ir = torch.clamp(kl_text_from_ir, min=0.0)

                            # 2. 反向蒸馏: IR <- Text
                            log_p_ir_T = F.log_softmax(pred_ir / T_s, dim=-1)
                            p_rgb_T = F.softmax(pred_rgb / T_t, dim=-1).detach()
                            kl_ir_from_text = F.kl_div(log_p_ir_T, p_rgb_T, reduction="batchmean") * (T_s * T_s)
                            kl_ir_from_text = torch.clamp(kl_ir_from_text, min=0.0)

                            # 3. 动态权重 (示例：根据关系调整)
                            if rel in ['color', 'clothing_type']:  # 文本信息更可靠的属性
                                alpha, beta = 0.3, 0.7  # 侧重让红外学习文本
                            else:  # 轮廓、存在性等红外可能更稳定的属性
                                alpha, beta = 0.7, 0.3  # 侧重让文本学习红外

                            # 4. 合并双向损失
                            kl_bidirectional = alpha * kl_text_from_ir + beta * kl_ir_from_text
                            attr_losses.append(kl_bidirectional)

                            # 可选：属性 CE 监督（mask 掉 -1 和越界值）
                            if getattr(self.args, 'use_attr_ce', True):
                                # 来自 batch 的属性标签向量，长度与关系列表一致
                                target_rgb = attr_tensor_rgb[:, i] if i < attr_tensor_rgb.size(1) else None
                                target_ir  = attr_tensor_ir[:, i]  if i < attr_tensor_ir.size(1)  else None
                                num_classes = pred_rgb.size(1)  # 当前 relation 的类别数
                                
                                if target_rgb is not None:
                                    # 确保标签在有效范围内 [0, num_classes-1]
                                    mask_rgb = (target_rgb >= 0) & (target_rgb < num_classes)
                                    if mask_rgb.any():
                                        ce_rgb = self.attr_criterion(pred_rgb[mask_rgb], target_rgb[mask_rgb])
                                        attr_ce_losses.append(ce_rgb)
                                        
                                if target_ir is not None:
                                    # 确保标签在有效范围内 [0, num_classes-1]
                                    mask_ir = (target_ir >= 0) & (target_ir < num_classes)
                                    if mask_ir.any():
                                        ce_ir = self.attr_criterion(pred_ir[mask_ir], target_ir[mask_ir])
                                        attr_ce_losses.append(ce_ir)

                        # 课程式启用（基于 step）
                        attr_weight = float(getattr(self.args, 'attr_loss_weight', 0.3))
                        if self._global_step < self.attr_warmup_iters:
                            warm_ratio = float(self._global_step) / max(1, self.attr_warmup_iters)
                            attr_weight = attr_weight * warm_ratio

                        if len(attr_losses) > 0:
                            ret['attr_consistency_loss'] = torch.stack(attr_losses).mean() * attr_weight
                        else:
                            ret['attr_consistency_loss'] = torch.tensor(0.0).to(t_feats.device)

                        if len(attr_ce_losses) > 0 and self.attr_ce_weight > 0:
                            ret['attr_ce_loss'] = torch.stack(attr_ce_losses).mean() * self.attr_ce_weight
                        else:
                            ret['attr_ce_loss'] = torch.tensor(0.0).to(t_feats.device)
                    # # uni_id
                    if "id" in loss_list:
                        # uni_id
                        uni_pids = torch.cat([label_rgb,label_rgb,label_ir,label_ir,label_ir], dim=0)
                        all_feats = torch.cat((ori_vi_feats, aug_vi_feats, ir_feats, f_feats, t_feats), dim=0)
                        _, all_feat_scores = self.classifier(all_feats)
                        ret.update({'uni_id_loss':(self.pid_criterion(all_feat_scores, uni_pids))*self.args.id_loss_weight})
                        feat_acc = (all_feat_scores.max(1)[1] == uni_pids).float().mean()
                        ret.update({'acc': feat_acc})
                    
                    if "wrt" in loss_list:
                        # uni_wrt
                        ret.update({'uni_wrt_loss':(self.tri_criterion(all_feats, uni_pids))*self.args.wrt_loss_weight})
                

        elif self.args.training_mode == "RGB_Text":
            text_rgb = batch_dict['text_rgb']
            text_rgb_feats_map = self.base_model.encode_text(text_rgb)
            t_feats = text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)].float()
            img_text_feats = torch.cat((rgb_feats, t_feats), dim=0)
            pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
            if 'id' in loss_list:
                _, scores = self.classifier(img_text_feats)
                ret.update({'id_loss':(self.pid_criterion(scores, pids))*self.args.id_loss_weight})
                acc = (scores.max(1)[1] == pids).float().mean()
                ret.update({'acc': acc})

            if 'wrt' in loss_list:
                ret.update({'wrt_loss':(self.tri_criterion(img_text_feats, pids))*self.args.wrt_loss_weight})
        else:
            raise ValueError("training mode must be in ['RGB_IR_Text', 'RGB_IR', 'RGB_Text']")
        
        # step 计数（用于 warmup）
        self._global_step += 1
 
        return ret

def build_model(config):
    model = CLIP2ReID(config, num_classes=config.pid_num)
    return model


