
import torchvision.transforms as transforms
from data_loader.dataset import process_query_sysu, process_gallery_sysu, \
    process_test_regdb,process_gallery_llcm,process_query_llcm, SYSU_Tri_Data,RegDB_Tri_Data,LLCM_Tri_Data,Test_Tri_Data
from data_loader.processing import ChannelRandomErasing, ChannelAdapGray, ChannelExchange
from data_loader.sampler import GenIdx, IdentitySampler
import torch.utils.data as data
import os
import json
import torch
def custom_collate_fn(batch):
    """
    自定义的 collate 函数，用于处理字典中键不一致的情况
    """
    # 获取所有可能的键
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())
    
    # 为每个键创建列表
    result = {}
    for key in all_keys:
        values = []
        for item in batch:
            if key in item:
                values.append(item[key])
            else:
                # 如果某个样本没有这个键，添加 None 或默认值
                if key in ['attr_rgb_tensor', 'attr_ir_tensor']:
                    # 对于属性张量，创建一个全为-1的张量
                    if key == 'attr_rgb_tensor':
                        # 需要知道RGB属性的维度，这里先用一个合理的默认值
                        values.append(torch.full((20,), -1, dtype=torch.int64))  # 假设最多20个RGB属性
                    elif key == 'attr_ir_tensor':
                        values.append(torch.full((20,), -1, dtype=torch.int64))  # 假设最多20个IR属性
                else:
                    values.append(None)
        
        # 使用 PyTorch 的默认 collate 函数处理非 None 的值
        if all(v is not None for v in values):
            try:
                result[key] = data.dataloader.default_collate(values)
            except:
                # 如果默认 collate 失败，直接返回列表
                result[key] = values
        else:
            # 如果有 None 值，直接返回列表
            result[key] = values
    
    return result
class Loader:

    def __init__(self, config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.6)])

        self.transform_color2 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.6),
            ChannelExchange(gray = 2)])
            
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.6)])
        

        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.img_h, config.img_w)),
            transforms.ToTensor(),
            normalize])
        
        # dataset name and path
        self.dataset = config.dataset 
        self.sysu_data_path = config.sysu_data_path
        
        self.llcm_data_path = config.llcm_data_path

        if config.dataset == 'regdb':
            self.regdb_data_path = config.regdb_data_path
            self.trial = config.trial # only for RegDB dataset
            self.eval_num_regdb = config.eval_num_regdb # only for RegDB dataset

        # image size
        self.img_w = config.img_w
        self.img_h = config.img_h

        # number of positive identities
        self.num_pos = config.num_pos

        # batch size
        self.batch_size = config.batch_size
        
        # model setting
        self.mode = config.mode
        self.test_mode = config.test_mode
        self.gall_mode = config.gall_mode
        self.num_workers = config.num_workers
        self.joint_mode = config.joint_mode

        # nlp augmentation setting
        self.Feat_Filter = config.Feat_Filter
        self.captioner_name = config.captioner_name
        self.llm_aug = config.llm_aug
        self.llm_aug_prob = config.llm_aug_prob
        if "Text" in config.training_mode:
            print(f"Training With Text Generated From: {config.captioner_name}\n Traininig Mode: {config.training_mode}")

        # form dataloader
        self._loader()

        # === 将知识图谱属性映射传递给配置 ===
        if hasattr(self, 'samples') and hasattr(self.samples, 'attr_value_map_rgb'):
            config.attr_value_map_rgb = self.samples.attr_value_map_rgb
            config.attr_value_map_ir = self.samples.attr_value_map_ir
            # 同步关系列表（训练阶段也需要），保持与 dataset 中顺序一致
            if hasattr(self.samples, 'attr_rel_list_rgb'):
                config.attr_rel_list_rgb = list(self.samples.attr_rel_list_rgb)
            else:
                config.attr_rel_list_rgb = list(config.attr_value_map_rgb.keys())
            if hasattr(self.samples, 'attr_rel_list_ir'):
                config.attr_rel_list_ir = list(self.samples.attr_rel_list_ir)
            else:
                config.attr_rel_list_ir = list(config.attr_value_map_ir.keys())
            print(f"[INFO] 知识图谱属性映射已传递给配置:")
            print(f"  RGB 属性: {len(config.attr_value_map_rgb)} 个关系")
            print(f"  IR 属性: {len(config.attr_value_map_ir)} 个关系")
            for rel in config.attr_value_map_rgb.keys():
                print(f"    RGB {rel}: {len(config.attr_value_map_rgb[rel])} 个值")
            for rel in config.attr_value_map_ir.keys():
                print(f"    IR {rel}: {len(config.attr_value_map_ir[rel])} 个值")
        else:
            # 测试阶段：尝试从数据目录加载已保存的属性映射 JSON
            base_path = None
            if self.dataset == 'sysu':
                base_path = self.sysu_data_path
            elif self.dataset == 'regdb':
                base_path = self.regdb_data_path if hasattr(self, 'regdb_data_path') else None
            elif self.dataset == 'llcm':
                base_path = self.llcm_data_path
            if base_path is not None:
                kg_dir = os.path.join(base_path, 'KG')
                rgb_json = os.path.join(kg_dir, 'attr_value_map_rgb.json')
                ir_json  = os.path.join(kg_dir, 'attr_value_map_ir.json')
                try:
                    if os.path.isfile(rgb_json):
                        with open(rgb_json, 'r', encoding='utf-8') as f:
                            config.attr_value_map_rgb = json.load(f)
                    else:
                        config.attr_value_map_rgb = {}
                    if os.path.isfile(ir_json):
                        with open(ir_json, 'r', encoding='utf-8') as f:
                            config.attr_value_map_ir = json.load(f)
                    else:
                        config.attr_value_map_ir = {}
                    # 同步关系列表，若缺失
                    if not hasattr(config, 'attr_rel_list_rgb') or not config.attr_rel_list_rgb:
                        config.attr_rel_list_rgb = list(config.attr_value_map_rgb.keys())
                    if not hasattr(config, 'attr_rel_list_ir') or not config.attr_rel_list_ir:
                        config.attr_rel_list_ir = list(config.attr_value_map_ir.keys())
                    if len(config.attr_value_map_rgb) or len(config.attr_value_map_ir):
                        print(f"[INFO] 从 {kg_dir} 加载属性映射用于测试：RGB={len(config.attr_value_map_rgb)}，IR={len(config.attr_value_map_ir)}")
                        # 打印前若干关系名称及其类别数
                        preview_num = 3
                        rgb_rels = list(config.attr_value_map_rgb.keys())[:preview_num]
                        ir_rels  = list(config.attr_value_map_ir.keys())[:preview_num]
                        for rel in rgb_rels:
                            print(f"  [RGB] {rel}: {len(config.attr_value_map_rgb[rel])} 类")
                        for rel in ir_rels:
                            print(f"  [IR ] {rel}: {len(config.attr_value_map_ir[rel])} 类")
                        if len(config.attr_value_map_rgb) > preview_num:
                            print(f"  [RGB] ... 以及 {len(config.attr_value_map_rgb)-preview_num} 个其他关系")
                        if len(config.attr_value_map_ir) > preview_num:
                            print(f"  [IR ] ... 以及 {len(config.attr_value_map_ir)-preview_num} 个其他关系")
                except Exception as e:
                    print(f"[WARN] 加载属性映射失败：{e}")

    def _loader(self):
        if self.dataset == 'sysu':
            if self.mode == 'train':
                # train sysu data simples
                samples = SYSU_Tri_Data(self.sysu_data_path, transform1=self.transform_color1, transform2=self.transform_color2,
                                transform3=self.transform_thermal,\
                                        llm_aug_prob=self.llm_aug_prob,\
                                                llm_aug=self.llm_aug,captioner_name=self.captioner_name,\
                                                    joint_mode=self.joint_mode,\
                                                        Feat_Filter=self.Feat_Filter)
                self.color_pos, self.thermal_pos = GenIdx(samples.train_color_label, samples.train_thermal_label)
                self.samples = samples

            # test sysu data simples
            query_samples, gallery_samples_list = self._get_test_samples(self.dataset)
            query_loader = data.DataLoader(query_samples, batch_size=128, shuffle=False, drop_last=False,
                                                num_workers=self.num_workers)
            gallery_loaders = []
            for i in range(10):
                gallery_loader = data.DataLoader(gallery_samples_list[i], batch_size=128, shuffle=False,
                                                 drop_last=False, num_workers=self.num_workers)
                gallery_loaders.append(gallery_loader)
            self.query_loader = query_loader
            self.gallery_loaders = gallery_loaders

        elif self.dataset == 'regdb':
            if self.mode == 'train':
                samples = RegDB_Tri_Data(self.regdb_data_path, trial=self.trial, transform1=self.transform_color1, transform2=self.transform_color2,
                                transform3=self.transform_thermal,\
                                        llm_aug_prob=self.llm_aug_prob,\
                                                llm_aug=self.llm_aug,captioner_name=self.captioner_name,\
                                                    joint_mode=self.joint_mode,\
                                                        Feat_Filter=self.Feat_Filter)
                self.color_pos, self.thermal_pos = GenIdx(samples.train_color_label, samples.train_thermal_label)
                self.samples = samples


            query_samples_list, gallery_samples_list = self._get_test_samples(self.dataset)
            query_loaders = []
            for i in range(self.eval_num_regdb):
                query_loader = data.DataLoader(query_samples_list[i], batch_size=128, shuffle=False, drop_last=False,
                                                    num_workers=self.num_workers)
                query_loaders.append(query_loader)
            self.query_loaders = query_loaders
            
            gallery_loaders = []
            for i in range(self.eval_num_regdb):
                gallery_loader = data.DataLoader(gallery_samples_list[i], batch_size=128, shuffle=False, drop_last=False,
                                             num_workers=self.num_workers)
                gallery_loaders.append(gallery_loader)
            self.gallery_loaders = gallery_loaders
        
        elif self.dataset == 'llcm':
            if self.mode == 'train':
                samples = LLCM_Tri_Data(self.llcm_data_path,transform1=self.transform_color1, transform2=self.transform_color2,
                                transform3=self.transform_thermal,\
                                        llm_aug_prob=self.llm_aug_prob,\
                                                llm_aug=self.llm_aug,captioner_name=self.captioner_name,\
                                                    joint_mode=self.joint_mode,\
                                                        Feat_Filter=self.Feat_Filter)
                self.color_pos, self.thermal_pos = GenIdx(samples.train_color_label, samples.train_thermal_label)
                self.samples = samples
            
            query_samples, gallery_samples_list = self._get_test_samples(self.dataset)
            query_loader = data.DataLoader(query_samples, batch_size=128, shuffle=False, drop_last=False,
                                                num_workers=self.num_workers)
            gallery_loaders = []
            for i in range(10):
                gallery_loader = data.DataLoader(gallery_samples_list[i], batch_size=128, shuffle=False, drop_last=False,
                                             num_workers=self.num_workers)
                gallery_loaders.append(gallery_loader)
            self.query_loader = query_loader
            self.gallery_loaders = gallery_loaders

    def _get_test_samples(self, dataset):
        if dataset == 'sysu':
            query_img, query_label, query_cam = process_query_sysu(self.sysu_data_path, mode=self.test_mode)
            query_samples = Test_Tri_Data(query_img, query_label, transform=self.transform_test,
                                     img_size=(self.img_w, self.img_h), data_path=self.sysu_data_path,\
                                        captioner_name=self.captioner_name, joint_mode=self.joint_mode,gallorquery='query',\
                                            Feat_Filter=self.Feat_Filter)
            self.query_label = query_label
            self.query_cam = query_cam

            self.n_query = len(query_label)

            gallery_samples_list = []
            for i in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(self.sysu_data_path, mode=self.test_mode, trial=i,
                                                                      gall_mode=self.gall_mode)
                self.gall_cam = gall_cam
                self.gall_label = gall_label
                self.n_gallery = len(gall_label)

                gallery_samples = Test_Tri_Data(gall_img, gall_label,data_path=self.sysu_data_path,transform=self.transform_test,
                                        img_size=(self.img_w, self.img_h), joint_mode=self.joint_mode,gallorquery=f'gall[{i+1}]')
                gallery_samples_list.append(gallery_samples)
            return query_samples, gallery_samples_list
        elif self.dataset == 'regdb':
            query_samples_list = []
            for i in range(1,1+self.eval_num_regdb):
                query_img, query_label = process_test_regdb(self.regdb_data_path, trial=self.trial, modal='thermal')
                self.query_label = query_label
                self.n_query = len(query_label)
                query_samples = Test_Tri_Data(query_img, query_label, transform=self.transform_test,
                                        img_size=(self.img_w, self.img_h), data_path=self.regdb_data_path,\
                                            captioner_name=self.captioner_name, \
                                                joint_mode=self.joint_mode,gallorquery=f'query[{self.trial}]',\
                                                Feat_Filter=self.Feat_Filter)
                query_samples_list.append(query_samples)

            gallery_samples_list = []
            for i in range(1,1+self.eval_num_regdb):
                gall_img, gall_label = process_test_regdb(self.regdb_data_path, trial=self.trial, modal='visible')
                self.gall_label = gall_label
                self.n_gallery = len(gall_label)

                gallery_samples = Test_Tri_Data(gall_img, gall_label,data_path=self.regdb_data_path,transform=self.transform_test,
                                            img_size=(self.img_w, self.img_h), captioner_name=self.captioner_name,\
                                                joint_mode=self.joint_mode,gallorquery=f'gall[{self.trial}]')
                gallery_samples_list.append(gallery_samples)
            return query_samples_list, gallery_samples_list
        elif self.dataset == 'llcm':
            query_img, query_label, query_cam = process_query_llcm(self.llcm_data_path, mode=2) # nir
            query_samples = Test_Tri_Data(query_img, query_label, transform=self.transform_test,
                                     img_size=(self.img_w, self.img_h), data_path=self.llcm_data_path,\
                                        captioner_name=self.captioner_name, \
                                            joint_mode=self.joint_mode,gallorquery='query',\
                                            Feat_Filter=self.Feat_Filter)
            self.query_label = query_label
            self.query_cam = query_cam

            self.n_query = len(query_label)

            gallery_samples_list = []
            for i in range(10):
                gall_img, gall_label, gall_cam = process_gallery_llcm(self.llcm_data_path, mode=1, trial=i) # vis
                
                self.gall_cam = gall_cam
                self.gall_label = gall_label
                self.n_gallery = len(gall_label)

                gallery_samples = Test_Tri_Data(gall_img, gall_label,data_path=self.llcm_data_path,transform=self.transform_test,
                                            img_size=(self.img_w, self.img_h), captioner_name=self.captioner_name,\
                                                joint_mode=self.joint_mode,gallorquery=f'gall[{i+1}]')
                gallery_samples_list.append(gallery_samples)
            return query_samples, gallery_samples_list
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")


    def get_train_loader(self):
        sampler = IdentitySampler(self.samples.train_color_label, self.samples.train_thermal_label, self.color_pos,
                                  self.thermal_pos, self.num_pos, int(self.batch_size / self.num_pos))
        self.samples.cIndex = sampler.index1
        self.samples.tIndex = sampler.index2
        train_loader = data.DataLoader(self.samples, batch_size=self.batch_size,
                                       sampler=sampler, num_workers=self.num_workers, drop_last=True,
                                       collate_fn=custom_collate_fn)
        return train_loader