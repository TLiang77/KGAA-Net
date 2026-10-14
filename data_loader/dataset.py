import json
import os
import random
import regex as re
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
from .tokenizer import SimpleTokenizer
from tqdm import tqdm
import collections
#sysu:0.711485212	0.813491868	0.84293973
def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result

def parse_triples_to_attrs(path):
    """
    解析三元组文本（每行形如: Person_193 --[holds]--> bag 或 CSV: Person_193,holds,bag）
    返回:
      attr_dict: { pid_int: {rel: value_str_id} }  (value_str_id 先用字符串, 后面映射到整数)
      raw_value_map: { rel: {value_str: count} }
    """
    attr_dict = {}
    raw_value_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 兼容两种格式
            if "--[" in line and "]-->" in line:
                head, rest = line.split("--[", 1)
                rel, tail = rest.split("]-->", 1)
                head = head.strip(); rel = rel.strip(); tail = tail.strip()
            else:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 3:
                    continue
                head, rel, tail = parts[0], parts[1], ','.join(parts[2:])
            if not head.startswith("Person_"):
                continue
            try:
                pid = int(head.split("_")[1])
            except:
                continue
            raw_value_map.setdefault(rel, collections.Counter())
            raw_value_map[rel][tail] += 1
            if pid not in attr_dict:
                attr_dict[pid] = {}
            attr_dict[pid][rel] = tail
    return attr_dict, raw_value_map

def build_unified_value_map_and_filter(raw_value_map_rgb, raw_value_map_ir, min_samples=100, max_classes=20, topk_rels=10):
    """
    统一 RGB/IR 两个模态的类别空间：
      1) 先将两个模态的 raw_value_map 在同一 relation 维度做 Counter 合并
      2) 用合并后的频次筛选 relation（>= min_samples）
      3) 用合并后的频次选择每个 relation 的 Top-K 类别（其余归为 'other'）
      4) 用合并后的 relation 总频次选择全局 Top-K relations
    返回：统一的 attr_value_map（适用于 RGB 与 IR），以及统一的 attr_rel_list
    """
    unified_attr_value_map = {}
    rel_total_freq = {}

    # 统一关系键集合
    all_rels = set(list(raw_value_map_rgb.keys()) + list(raw_value_map_ir.keys()))

    for rel in all_rels:
        counter_rgb = raw_value_map_rgb.get(rel, collections.Counter())
        counter_ir  = raw_value_map_ir.get(rel,  collections.Counter())
        merged_counter = counter_rgb + counter_ir

        total = sum(merged_counter.values())
        if total < min_samples:
            continue
        rel_total_freq[rel] = total

        items = merged_counter.most_common()
        if len(items) > max_classes:
            items = items[:max_classes-1]
            values = [v for v, _ in items]
            values.append('other')
        else:
            values = [v for v, _ in items]

        unified_attr_value_map[rel] = {v: i for i, v in enumerate(values)}

    # 全局 Top-K relation
    sorted_rels = sorted(rel_total_freq.items(), key=lambda x: -x[1])
    filtered_rels = [rel for rel, _ in sorted_rels[:topk_rels] if rel in unified_attr_value_map]

    return unified_attr_value_map, filtered_rels

class SYSU_Tri_Data(data.Dataset):
    def __init__(self, data_dir, transform1=None, \
                 transform2=None, transform3=None, \
                 colorIndex=None, thermalIndex=None, \
                 text_length=77, llm_aug_prob=0.6,\
                 llm_aug=False, captioner_name='GIT', joint_mode="ir_crossfusion", \
                 Feat_Filter=False): # include: Feat_Filter=False
        # initialize text tokenizer
        self.tokenizer = SimpleTokenizer()

        # Load RGB data
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_image = train_color_image
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        # Load IR data
        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # Load text data
        self.Feat_Filter = Feat_Filter
        self.joint_mode = joint_mode
        self.text_length = text_length
        self.llm_aug = llm_aug
        self.llm_aug_prob = llm_aug_prob

        if joint_mode == "ir_crossfusion" or joint_mode == "uni":
            print("Loading RGB Text For Training...")
            self.text_dir_rgb = data_dir + f'Text/{captioner_name}_RGB/'
            self.train_text_rgb = np.load(self.text_dir_rgb + f'train_text_{captioner_name}_RGB.npy')
            self.train_text_rgb = [tokenize(caption, self.tokenizer) for caption in self.train_text_rgb]
            self.train_text_label_rgb = np.load(self.text_dir_rgb + f'train_text_label_{captioner_name}_RGB.npy')
            if llm_aug:
                self.llm_text_rgb = np.load(self.text_dir_rgb + f'train_llm_text_{captioner_name}_RGB.npy')
                self.llm_text_rgb = [tokenize(caption, self.tokenizer) for caption in self.llm_text_rgb]
            
            if Feat_Filter:
                print("Loading IR Text as Feat Filter For Training...")
                self.text_dir_ir = data_dir + f'Text/{captioner_name}_IR/'
                self.train_text_ir = np.load(self.text_dir_ir + f'train_text_{captioner_name}_IR.npy')
                self.train_text_ir = [tokenize(caption, self.tokenizer) for caption in self.train_text_ir]
                self.train_text_label_ir = np.load(self.text_dir_ir + f'train_text_label_{captioner_name}_IR.npy')
                if llm_aug:
                    self.llm_text_ir = np.load(self.text_dir_ir + f'train_llm_text_{captioner_name}_IR.npy')
                    self.llm_text_ir = [tokenize(caption, self.tokenizer) for caption in self.llm_text_ir]
        
        # parse triples
        triples_rgb_path = "/home/hdq/project/TVI-LFM-main-sxfz/datasets/sysu/KG/triples_RGB.csv"
        triples_ir_path  = "/home/hdq/project/TVI-LFM-main-sxfz/datasets/sysu/KG/triples_IR.csv"
        self.attr_dict_rgb_raw, self.raw_value_map_rgb = parse_triples_to_attrs(triples_rgb_path)
        self.attr_dict_ir_raw,  self.raw_value_map_ir  = parse_triples_to_attrs(triples_ir_path)

        # 统一构建 RGB/IR 的属性空间与关系列表
        unified_map, unified_rels = build_unified_value_map_and_filter(
            self.raw_value_map_rgb, self.raw_value_map_ir, min_samples=100, max_classes=20, topk_rels=10
        )
        self.attr_value_map_rgb = unified_map
        self.attr_value_map_ir  = unified_map
        self.attr_rel_list_rgb  = unified_rels
        self.attr_rel_list_ir   = unified_rels

        # 将每个 pid 的字符串 value -> id（如果 value 不在保留列表，映为 'other' 或 -1）
        self.attr_dict_rgb = {}
        for pid, rels in self.attr_dict_rgb_raw.items():
            mapped = {}
            for rel in self.attr_rel_list_rgb:
                v = rels.get(rel, None)
                if v is None:
                    mapped[rel] = -1
                else:
                    if v in self.attr_value_map_rgb[rel]:
                        mapped[rel] = self.attr_value_map_rgb[rel][v]
                    else:
                        # 如果 value 不在 top-K，则归为 'other'（若存在）或 -1
                        mapped[rel] = self.attr_value_map_rgb[rel].get('other', -1)
            self.attr_dict_rgb[pid] = mapped

        self.attr_dict_ir = {}
        for pid, rels in self.attr_dict_ir_raw.items():
            mapped = {}
            for rel in self.attr_rel_list_ir:
                v = rels.get(rel, None)
                if v is None:
                    mapped[rel] = -1
                else:
                    if v in self.attr_value_map_ir[rel]:
                        mapped[rel] = self.attr_value_map_ir[rel][v]
                    else:
                        mapped[rel] = self.attr_value_map_ir[rel].get('other', -1)
            self.attr_dict_ir[pid] = mapped

        # === Remap KG pid keys (orig pid) -> train pid (0..N-1) based on exp/train_id.txt ===
        try:
            train_id_file = os.path.join(data_dir, 'exp', 'train_id.txt')
            if os.path.isfile(train_id_file):
                with open(train_id_file, 'r', encoding='utf-8') as f:
                    line = f.read().strip()
                # 兼容与 test_id 相同的逗号分隔格式或换行分隔
                if ',' in line:
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                else:
                    parts = [p.strip() for p in line.splitlines() if p.strip()]
                # train 映射按出现顺序 0..N-1
                orig_pid_list = [int(p) for p in parts]
                orig_to_train = {pid: idx for idx, pid in enumerate(orig_pid_list)}

                def remap_attr_dict(orig_attr_dict):
                    remapped = {}
                    miss = 0
                    for orig_pid, rel_map in orig_attr_dict.items():
                        if orig_pid in orig_to_train:
                            remapped[orig_to_train[orig_pid]] = rel_map
                        else:
                            miss += 1
                    return remapped, miss

                self.attr_dict_rgb, miss_rgb = remap_attr_dict(self.attr_dict_rgb)
                self.attr_dict_ir, miss_ir = remap_attr_dict(self.attr_dict_ir)

                # 统计覆盖情况：训练ID中有多少在KG里
                kg_rgb_ids = set(self.attr_dict_rgb_raw.keys()) if hasattr(self, 'attr_dict_rgb_raw') else set()
                kg_ir_ids  = set(self.attr_dict_ir_raw.keys())  if hasattr(self, 'attr_dict_ir_raw') else set()
                train_ids_set = set(orig_to_train.keys())
                covered_rgb = len(train_ids_set & kg_rgb_ids)
                covered_ir  = len(train_ids_set & kg_ir_ids)
                not_in_kg_rgb = len(train_ids_set - kg_rgb_ids)
                not_in_kg_ir  = len(train_ids_set - kg_ir_ids)

                print(f"[KG] SYSU train id remap loaded: {len(orig_to_train)} ids")
                print(f"      RGB: covered(train∩KG)={covered_rgb}, not_in_KG(train-KG)={not_in_kg_rgb}, KG_only(miss)={miss_rgb}")
                print(f"      IR : covered(train∩KG)={covered_ir},  not_in_KG(train-KG)={not_in_kg_ir},  KG_only(miss)={miss_ir}")
            else:
                print(f"[KG] SYSU train id file not found: {train_id_file}; using original KG pid space (may cause -1)")
        except Exception as e:
            print(f"[KG] Remapping train ids failed: {e}")

        # Save attr maps to JSON for inspection (单独文件，不混到训练日志)
        save_dir = os.path.join(data_dir, "KG")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "attr_value_map_rgb.json"), "w", encoding="utf-8") as f:
            json.dump(self.attr_value_map_rgb, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, "attr_value_map_ir.json"), "w", encoding="utf-8") as f:
            json.dump(self.attr_value_map_ir, f, ensure_ascii=False, indent=2)
        # get transforms
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

        # initialize position indices (for simplers)
        self.cIndex = colorIndex
        self.tIndex = thermalIndex


    def __getitem__(self, index):
        # define batch_dict
        batch_dict = {}

        # get image and label
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        # apply img transforms
        img1_0 = self.transform1(img1) # color image
        img1_1 = self.transform2(img1) # color image with augmentation
        img2 = self.transform3(img2)  # thermal image

        # apply text transforms
        if self.joint_mode == "ir_crossfusion" or self.joint_mode == "uni":
            caption_id_rgb = self.train_text_rgb[self.cIndex[index]]
            if self.llm_aug and random.random() < self.llm_aug_prob:
                caption_id_rgb = self.llm_text_rgb[self.cIndex[index]]
            batch_dict['text_rgb'] = caption_id_rgb

            if self.Feat_Filter:
                caption_id_ir = self.train_text_ir[self.tIndex[index]]
                if self.llm_aug and random.random() < self.llm_aug_prob:
                    caption_id_ir = self.llm_text_ir[self.tIndex[index]]
                batch_dict['text_ir'] = caption_id_ir
        
        # add to batch_dict
        batch_dict['img_rgb_ori'] = img1_0
        batch_dict['img_rgb_aug'] = img1_1
        batch_dict['img_ir'] = img2
        batch_dict['target_rgb'] = target1
        batch_dict['target_ir'] = target2

        # 在 __getitem__
        pid_rgb = int(target1)
        pid_ir  = int(target2)

        # build fixed-length attr vectors (len = R)
        R_rgb = len(self.attr_rel_list_rgb)
        attr_vec_rgb = np.full((R_rgb,), -1, dtype=np.int64)
        if pid_rgb in self.attr_dict_rgb:
            for i, rel in enumerate(self.attr_rel_list_rgb):
                attr_vec_rgb[i] = self.attr_dict_rgb[pid_rgb].get(rel, -1)
        batch_dict['attr_rgb_tensor'] = torch.from_numpy(attr_vec_rgb)

        R_ir = len(self.attr_rel_list_ir)
        attr_vec_ir = np.full((R_ir,), -1, dtype=np.int64)
        if pid_ir in self.attr_dict_ir:
            for i, rel in enumerate(self.attr_rel_list_ir):
                attr_vec_ir[i] = self.attr_dict_ir[pid_ir].get(rel, -1)
        batch_dict['attr_ir_tensor'] = torch.from_numpy(attr_vec_ir)

        return batch_dict

    def __len__(self):
        return len(self.train_color_label)
    
    def get_bpe_tokens(self, word):
        token = ''.join(self.tokenizer.byte_encoder[b] for b in word.encode('utf-8'))
        bpe_tokens = [self.tokenizer.encoder[bpe_token] for bpe_token in self.tokenizer.bpe(token).split(' ')]
        return bpe_tokens

class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform1=None, transform2=None, transform3=None, colorIndex=None, thermalIndex=None):
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # RGB format
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform1(img1)
        img1_1 = self.transform2(img1)
        img2 = self.transform3(img2)

        return img1_0, img1_1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform1=None, transform2=None, transform3=None,
                 colorIndex=None, thermalIndex=None):
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # RGB format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        # img1_0 = self.transform1(img1)
        img1_0 = self.transform2(img1)
        img2 = self.transform3(img2)

        return img1_0, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
    

class RegDB_Tri_Data(data.Dataset):
    def __init__(self, data_dir, trial,transform1=None, \
                 transform2=None, transform3=None, \
                    colorIndex=None, thermalIndex=None, \
                            text_length=77, llm_aug_prob=0.5,\
                                    llm_aug=False, captioner_name='Blip', joint_mode="ir_crossfusion", \
                                        Feat_Filter=False):
        # initialize text tokenizer
        self.tokenizer = SimpleTokenizer()

        # init text option
        self.Feat_Filter = Feat_Filter
        self.joint_mode = joint_mode
        self.text_length = text_length
        self.llm_aug = llm_aug
        self.llm_aug_prob = llm_aug_prob

        # Load RGB&IR training data
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # IR format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        # Load text data
        if joint_mode == "ir_crossfusion" or joint_mode == "uni":
            print("Loading RGB Text For Training...")
            self.text_dir_rgb = data_dir + f'Text/{captioner_name}_RGB/'
            self.text_rgb_dict = json.load(open(self.text_dir_rgb + f'caption_llm_dict_{captioner_name}_RGB.json'))
            self.train_text_rgb = [tokenize(self.text_rgb_dict[data_dir + i_path]['description'], self.tokenizer) for i_path in color_img_file]
            if llm_aug:
                self.llm_text_rgb = [tokenize(self.text_rgb_dict[data_dir + i_path]['aug_description'], self.tokenizer) for i_path in color_img_file]
            
            if Feat_Filter:
                print("Loading IR Text as Feat Filter For Training...")
                self.text_dir_ir = data_dir + f'Text/{captioner_name}_IR/'
                self.text_ir_dict = json.load(open(self.text_dir_ir + f'caption_llm_dict_{captioner_name}_IR.json'))
                self.train_text_ir = [tokenize(self.text_ir_dict[data_dir + i_path]['description'], self.tokenizer) for i_path in thermal_img_file]
                if llm_aug:
                    self.llm_text_ir = [tokenize(self.text_ir_dict[data_dir + i_path]['aug_description'], self.tokenizer) for i_path in thermal_img_file]

        # parse triples
        triples_rgb_path = "/home/hdq/project/TVI-LFM-main-sxfz/datasets/regdb/KG/triples_RGB.csv"
        triples_ir_path  = "/home/hdq/project/TVI-LFM-main-sxfz/datasets/regdb/KG/triples_IR.csv"
        self.attr_dict_rgb_raw, self.raw_value_map_rgb = parse_triples_to_attrs(triples_rgb_path)
        self.attr_dict_ir_raw,  self.raw_value_map_ir  = parse_triples_to_attrs(triples_ir_path)

        # 统一构建 RGB/IR 的属性空间与关系列表
        unified_map, unified_rels = build_unified_value_map_and_filter(
            self.raw_value_map_rgb, self.raw_value_map_ir, min_samples=80, max_classes=15, topk_rels=10
        )
        self.attr_value_map_rgb = unified_map
        self.attr_value_map_ir  = unified_map
        self.attr_rel_list_rgb  = unified_rels
        self.attr_rel_list_ir   = unified_rels

        # 将每个 pid 的字符串 value -> id（如果 value 不在保留列表，映为 'other' 或 -1）
        self.attr_dict_rgb = {}
        for pid, rels in self.attr_dict_rgb_raw.items():
            mapped = {}
            for rel in self.attr_rel_list_rgb:
                v = rels.get(rel, None)
                if v is None:
                    mapped[rel] = -1
                else:
                    if v in self.attr_value_map_rgb[rel]:
                        mapped[rel] = self.attr_value_map_rgb[rel][v]
                    else:
                        # 如果 value 不在 top-K，则归为 'other'（若存在）或 -1
                        mapped[rel] = self.attr_value_map_rgb[rel].get('other', -1)
            self.attr_dict_rgb[pid] = mapped

        self.attr_dict_ir = {}
        for pid, rels in self.attr_dict_ir_raw.items():
            mapped = {}
            for rel in self.attr_rel_list_ir:
                v = rels.get(rel, None)
                if v is None:
                    mapped[rel] = -1
                else:
                    if v in self.attr_value_map_ir[rel]:
                        mapped[rel] = self.attr_value_map_ir[rel][v]
                    else:
                        mapped[rel] = self.attr_value_map_ir[rel].get('other', -1)
            self.attr_dict_ir[pid] = mapped

        # === Remap KG pid keys (orig pid) -> train pid (0..N-1) based on exp/train_id.txt ===
        try:
            train_id_file = os.path.join(data_dir, 'idx', f'train_id_{trial}.txt')
            if os.path.isfile(train_id_file):
                with open(train_id_file, 'r', encoding='utf-8') as f:
                    line = f.read().strip()
                # 兼容与 test_id 相同的逗号分隔格式或换行分隔
                if ',' in line:
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                else:
                    parts = [p.strip() for p in line.splitlines() if p.strip()]
                # train 映射按出现顺序 0..N-1
                orig_pid_list = [int(p) for p in parts]
                orig_to_train = {pid: idx for idx, pid in enumerate(orig_pid_list)}

                def remap_attr_dict(orig_attr_dict):
                    remapped = {}
                    miss = 0
                    for orig_pid, rel_map in orig_attr_dict.items():
                        if orig_pid in orig_to_train:
                            remapped[orig_to_train[orig_pid]] = rel_map
                        else:
                            miss += 1
                    return remapped, miss

                self.attr_dict_rgb, miss_rgb = remap_attr_dict(self.attr_dict_rgb)
                self.attr_dict_ir, miss_ir = remap_attr_dict(self.attr_dict_ir)

                # 统计覆盖情况：训练ID中有多少在KG里
                kg_rgb_ids = set(self.attr_dict_rgb_raw.keys()) if hasattr(self, 'attr_dict_rgb_raw') else set()
                kg_ir_ids  = set(self.attr_dict_ir_raw.keys())  if hasattr(self, 'attr_dict_ir_raw') else set()
                train_ids_set = set(orig_to_train.keys())
                covered_rgb = len(train_ids_set & kg_rgb_ids)
                covered_ir  = len(train_ids_set & kg_ir_ids)
                not_in_kg_rgb = len(train_ids_set - kg_rgb_ids)
                not_in_kg_ir  = len(train_ids_set - kg_ir_ids)

                print(f"[KG] REGDB train id remap loaded: {len(orig_to_train)} ids")
                print(f"      RGB: covered(train∩KG)={covered_rgb}, not_in_KG(train-KG)={not_in_kg_rgb}, KG_only(miss)={miss_rgb}")
                print(f"      IR : covered(train∩KG)={covered_ir},  not_in_KG(train-KG)={not_in_kg_ir},  KG_only(miss)={miss_ir}")
            else:
                print(f"[KG] REGDB train id file not found: {train_id_file}; using original KG pid space (may cause -1)")
        except Exception as e:
            print(f"[KG] Remapping train ids failed: {e}")

        # Save attr maps to JSON for inspection (单独文件，不混到训练日志)
        save_dir = os.path.join(data_dir, "KG")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "attr_value_map_rgb.json"), "w", encoding="utf-8") as f:
            json.dump(self.attr_value_map_rgb, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, "attr_value_map_ir.json"), "w", encoding="utf-8") as f:
            json.dump(self.attr_value_map_ir, f, ensure_ascii=False, indent=2)
                
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        batch_dict = {}

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform1(img1) # color image
        img1_1 = self.transform2(img1) # color image with augmentation
        img2 = self.transform3(img2)

        if self.joint_mode == "ir_crossfusion" or self.joint_mode == "uni":
            caption_id_rgb = self.train_text_rgb[self.cIndex[index]]
            if self.llm_aug and random.random() < self.llm_aug_prob:
                    caption_id_rgb = self.llm_text_rgb[self.cIndex[index]]
            batch_dict['text_rgb'] = caption_id_rgb

            if self.Feat_Filter:
                caption_id_ir = self.train_text_ir[self.tIndex[index]]
                if self.llm_aug and random.random() < self.llm_aug_prob:
                    caption_id_ir = self.llm_text_ir[self.tIndex[index]]
                batch_dict['text_ir'] = caption_id_ir

        batch_dict['img_rgb_ori'] = img1_0
        batch_dict['img_rgb_aug'] = img1_1
        batch_dict['img_ir'] = img2
        batch_dict['target_rgb'] = target1
        batch_dict['target_ir'] = target2

        # 在 __getitem__
        pid_rgb = int(target1)
        pid_ir  = int(target2)

        # build fixed-length attr vectors (len = R)
        R_rgb = len(self.attr_rel_list_rgb)
        attr_vec_rgb = np.full((R_rgb,), -1, dtype=np.int64)
        if pid_rgb in self.attr_dict_rgb:
            for i, rel in enumerate(self.attr_rel_list_rgb):
                attr_vec_rgb[i] = self.attr_dict_rgb[pid_rgb].get(rel, -1)
        batch_dict['attr_rgb_tensor'] = torch.from_numpy(attr_vec_rgb)

        R_ir = len(self.attr_rel_list_ir)
        attr_vec_ir = np.full((R_ir,), -1, dtype=np.int64)
        if pid_ir in self.attr_dict_ir:
            for i, rel in enumerate(self.attr_rel_list_ir):
                attr_vec_ir[i] = self.attr_dict_ir[pid_ir].get(rel, -1)
        batch_dict['attr_ir_tensor'] = torch.from_numpy(attr_vec_ir)

        return batch_dict

    def __len__(self):
        return len(self.train_color_label)
    

class LLCM_Tri_Data(data.Dataset):
    def __init__(self, data_dir, transform1=None, \
                 transform2=None, transform3=None, \
                    colorIndex=None, thermalIndex=None, \
                            text_length=77, llm_aug_prob=0.5,\
                                    llm_aug=False, captioner_name='Blip', joint_mode="ir_crossfusion", \
                                        Feat_Filter=False):
        # initialize text tokenizer
        self.tokenizer = SimpleTokenizer()

        # init text option
        self.Feat_Filter = Feat_Filter
        self.joint_mode = joint_mode
        self.text_length = text_length
        self.llm_aug = llm_aug
        self.llm_aug_prob = llm_aug_prob

        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            #print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)
        
        # RGB format
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # IR format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        # Load text data
        if joint_mode == "ir_crossfusion" or joint_mode == "uni":
            print("Loading RGB Text For Training...")
            self.text_dir_rgb = data_dir + '/' f'Text/{captioner_name}_RGB/'
            self.text_rgb_dict = json.load(open(self.text_dir_rgb + f'caption_llm_dict_{captioner_name}_RGB.json'))
            self.train_text_rgb = [tokenize(self.text_rgb_dict[data_dir + i_path]['description'], self.tokenizer) for i_path in color_img_file]
            if llm_aug:
                self.llm_text_rgb = [tokenize(self.text_rgb_dict[data_dir + i_path]['aug_description'], self.tokenizer) for i_path in color_img_file]
            
            if Feat_Filter:
                print("Loading IR Text as Feat Filter For Training...")
                self.text_dir_ir = data_dir + f'Text/{captioner_name}_IR/'
                self.text_ir_dict = json.load(open(self.text_dir_ir + f'caption_llm_dict_{captioner_name}_IR.json'))
                self.train_text_ir = [tokenize(self.text_ir_dict[data_dir + i_path]['description'], self.tokenizer) for i_path in thermal_img_file]
                if llm_aug:
                    self.llm_text_ir = [tokenize(self.text_ir_dict[data_dir + i_path]['aug_description'], self.tokenizer) for i_path in thermal_img_file]

         # parse triples
        triples_rgb_path = "/home/hdq/project/TVI-LFM-main-sxfz/datasets/llcm/KG/triples_RGB_llcm.csv"
        triples_ir_path  = "/home/hdq/project/TVI-LFM-main-sxfz/datasets/llcm/KG/triples_IR_llcm.csv"
        self.attr_dict_rgb_raw, self.raw_value_map_rgb = parse_triples_to_attrs(triples_rgb_path)
        self.attr_dict_ir_raw,  self.raw_value_map_ir  = parse_triples_to_attrs(triples_ir_path)

        # 过滤并构造 value id map（超参数可以调）
        MIN_SAMPLES = 100  # 提高阈值，只保留最稳定的属性
        MAX_CLASSES = 20   # 减少类别数，降低复杂度
        unified_map, unified_rels = build_unified_value_map_and_filter(
            self.raw_value_map_rgb, self.raw_value_map_ir, min_samples=MIN_SAMPLES, max_classes=MAX_CLASSES
        )
        self.attr_value_map_rgb = unified_map
        self.attr_value_map_ir  = unified_map
        self.attr_rel_list_rgb  = unified_rels
        self.attr_rel_list_ir   = unified_rels

        # 将每个 pid 的字符串 value -> id（如果 value 不在保留列表，映为 'other' 或 -1）
        self.attr_dict_rgb = {}
        for pid, rels in self.attr_dict_rgb_raw.items():
            mapped = {}
            for rel in self.attr_rel_list_rgb:
                v = rels.get(rel, None)
                if v is None:
                    mapped[rel] = -1
                else:
                    if v in self.attr_value_map_rgb[rel]:
                        mapped[rel] = self.attr_value_map_rgb[rel][v]
                    else:
                        # 如果 value 不在 top-K，则归为 'other'（若存在）或 -1
                        mapped[rel] = self.attr_value_map_rgb[rel].get('other', -1)
            self.attr_dict_rgb[pid] = mapped

        self.attr_dict_ir = {}
        for pid, rels in self.attr_dict_ir_raw.items():
            mapped = {}
            for rel in self.attr_rel_list_ir:
                v = rels.get(rel, None)
                if v is None:
                    mapped[rel] = -1
                else:
                    if v in self.attr_value_map_ir[rel]:
                        mapped[rel] = self.attr_value_map_ir[rel][v]
                    else:
                        mapped[rel] = self.attr_value_map_ir[rel].get('other', -1)
            self.attr_dict_ir[pid] = mapped

        # === Remap KG pid keys (orig pid) -> train pid (0..N-1) based on exp/train_id.txt ===
        try:
            train_id_file = os.path.join(data_dir, 'idx', 'train_id.txt')
            if os.path.isfile(train_id_file):
                with open(train_id_file, 'r', encoding='utf-8') as f:
                    line = f.read().strip()
                # 兼容与 test_id 相同的逗号分隔格式或换行分隔
                if ',' in line:
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                else:
                    parts = [p.strip() for p in line.splitlines() if p.strip()]
                # train 映射按出现顺序 0..N-1
                orig_pid_list = [int(p) for p in parts]
                orig_to_train = {pid: idx for idx, pid in enumerate(orig_pid_list)}

                def remap_attr_dict(orig_attr_dict):
                    remapped = {}
                    miss = 0
                    for orig_pid, rel_map in orig_attr_dict.items():
                        if orig_pid in orig_to_train:
                            remapped[orig_to_train[orig_pid]] = rel_map
                        else:
                            miss += 1
                    return remapped, miss

                self.attr_dict_rgb, miss_rgb = remap_attr_dict(self.attr_dict_rgb)
                self.attr_dict_ir, miss_ir = remap_attr_dict(self.attr_dict_ir)

                # 统计覆盖情况：训练ID中有多少在KG里
                kg_rgb_ids = set(self.attr_dict_rgb_raw.keys()) if hasattr(self, 'attr_dict_rgb_raw') else set()
                kg_ir_ids  = set(self.attr_dict_ir_raw.keys())  if hasattr(self, 'attr_dict_ir_raw') else set()
                train_ids_set = set(orig_to_train.keys())
                covered_rgb = len(train_ids_set & kg_rgb_ids)
                covered_ir  = len(train_ids_set & kg_ir_ids)
                not_in_kg_rgb = len(train_ids_set - kg_rgb_ids)
                not_in_kg_ir  = len(train_ids_set - kg_ir_ids)

                print(f"[KG] LLCM train id remap loaded: {len(orig_to_train)} ids")
                print(f"      RGB: covered(train∩KG)={covered_rgb}, not_in_KG(train-KG)={not_in_kg_rgb}, KG_only(miss)={miss_rgb}")
                print(f"      IR : covered(train∩KG)={covered_ir},  not_in_KG(train-KG)={not_in_kg_ir},  KG_only(miss)={miss_ir}")
            else:
                print(f"[KG] LLCM train id file not found: {train_id_file}; using original KG pid space (may cause -1)")
        except Exception as e:
            print(f"[KG] Remapping train ids failed: {e}")

        # Save attr maps to JSON for inspection (单独文件，不混到训练日志)
        save_dir = os.path.join(data_dir, "KG")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "attr_value_map_rgb.json"), "w", encoding="utf-8") as f:
            json.dump(self.attr_value_map_rgb, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, "attr_value_map_ir.json"), "w", encoding="utf-8") as f:
            json.dump(self.attr_value_map_ir, f, ensure_ascii=False, indent=2)
        
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.cIndex = colorIndex
        self.tIndex = thermalIndex


    def __getitem__(self, index):
        batch_dict = {}

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform1(img1) # color image
        img1_1 = self.transform2(img1)
        img2 = self.transform3(img2)

        if self.joint_mode == "ir_crossfusion" or self.joint_mode == "uni":
            caption_id_rgb = self.train_text_rgb[self.cIndex[index]]
            if self.llm_aug and random.random() < self.llm_aug_prob:
                    caption_id_rgb = self.llm_text_rgb[self.cIndex[index]]
            batch_dict['text_rgb'] = caption_id_rgb

            if self.Feat_Filter:
                caption_id_ir = self.train_text_ir[self.tIndex[index]]
                if self.llm_aug and random.random() < self.llm_aug_prob:
                    caption_id_ir = self.llm_text_ir[self.tIndex[index]]
                batch_dict['text_ir'] = caption_id_ir

        # 在 __getitem__
        pid_rgb = int(target1)
        pid_ir  = int(target2)

        # build fixed-length attr vectors (len = R)
        R_rgb = len(self.attr_rel_list_rgb)
        attr_vec_rgb = np.full((R_rgb,), -1, dtype=np.int64)
        if pid_rgb in self.attr_dict_rgb:
            for i, rel in enumerate(self.attr_rel_list_rgb):
                attr_vec_rgb[i] = self.attr_dict_rgb[pid_rgb].get(rel, -1)
        batch_dict['attr_rgb_tensor'] = torch.from_numpy(attr_vec_rgb)

        R_ir = len(self.attr_rel_list_ir)
        attr_vec_ir = np.full((R_ir,), -1, dtype=np.int64)
        if pid_ir in self.attr_dict_ir:
            for i, rel in enumerate(self.attr_rel_list_ir):
                attr_vec_ir[i] = self.attr_dict_ir[pid_ir].get(rel, -1)
        batch_dict['attr_ir_tensor'] = torch.from_numpy(attr_vec_ir)

        batch_dict['img_rgb_ori'] = img1_0
        batch_dict['img_rgb_aug'] = img1_1
        batch_dict['img_ir'] = img2
        batch_dict['target_rgb'] = target1
        batch_dict['target_ir'] = target2

        return batch_dict

    def __len__(self):
        return len(self.train_color_label)
    


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(224, 224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
    
class Test_Tri_Data(data.Dataset):
    def __init__(self, test_img_file, test_label, data_path, transform=None, \
                 img_size=(144, 288), captioner_name='GIT', \
                    joint_mode="ir_crossfusion", gallorquery='query', \
                            Feat_Filter=False): # include Feat_Filter=False
        self.tokenizer = SimpleTokenizer()
        self.Feat_Filter = Feat_Filter
        self.type = gallorquery
        assert 'query' in gallorquery or 'gall' in gallorquery, "gallorquery must be 'query[i]' or 'gall[i]'"


        # Load ir test img data and text

        with open(data_path + f'Text/{captioner_name}_RGB/id_caption_map_{captioner_name}_RGB.json','r') as f:
            text_dict_rgb = json.load(f)
        if Feat_Filter:
            with open(data_path + f'Text/{captioner_name}_IR/caption_dict_{captioner_name}_IR.json','r') as f:
                text_dict_ir = json.load(f)
        

        test_image = []
        test_text_ir = []
        test_text_rgb = []
        self.joint_mode = joint_mode
        print(f"Loading Test {self.type} Data...")
        for i in range(len(test_img_file)):
            # load img from the test_img_file
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
            
            # load text from the test_label
            test_text_rgb.append(tokenize(np.random.choice(text_dict_rgb[str(test_label[i])]), self.tokenizer))
            if Feat_Filter:
                img_path = test_img_file[i]
                # Normalize path for Windows compatibility
                img_path = img_path.replace('\\', '/')
                if 'test_nir' in img_path: # replace as nir
                    nir_path = data_path + 'nir/' + img_path.replace('test_nir','nir').split('cam')[1][2:]
                    test_text_ir.append(tokenize(text_dict_ir[nir_path]['description'], self.tokenizer))
                else:
                    test_text_ir.append(tokenize(text_dict_ir[img_path]['description'], self.tokenizer))

        test_image = np.array(test_image)
        test_text_ir = np.array(test_text_ir)
        test_text_rgb = np.array(test_text_rgb)

        self.test_image = test_image
        self.test_text_rgb = test_text_rgb
        self.test_text_ir = test_text_ir
        self.test_label = test_label

        self.transform = transform

    def __getitem__(self, index):
        # define batch_dict
        batch_dict = {}

        if len(self.test_text_rgb):
            text = self.test_text_rgb[index]
            batch_dict['text'] = text
        if len(self.test_text_ir):
            text = self.test_text_ir[index]
            if self.Feat_Filter:
                batch_dict['text_filter'] = text
            else:
                batch_dict['text'] = text

        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)

        # add to batch_dict
        batch_dict['img'] = img1
        batch_dict['target'] = target1
        return batch_dict

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label

def process_query_sysu(data_path, mode='all', relabel=False):

    # mode selection
    if mode == 'all':
        ir_cameras = ['cam3', 'cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)

    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode='all', trial=0, relabel=False, gall_mode='single'):

    random.seed(trial)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                if gall_mode == 'single':
                    files_rgb.append(random.choice(new_files))
                if gall_mode == 'multi':
                    files_rgb.append(np.random.choice(new_files, 10, replace=False))
    gall_img = []
    gall_id = []
    gall_cam = []

    for img_path in files_rgb:
        if gall_mode == 'single':
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            gall_img.append(img_path)
            gall_id.append(pid)
            gall_cam.append(camid)

        if gall_mode == 'multi':
            for i in img_path:
                camid, pid = int(i[-15]), int(i[-13:-9])
                gall_img.append(i)
                gall_id.append(pid)
                gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split('/')[1]) for s in data_file_list]

    return file_image, np.array(file_label)


def process_query_llcm(data_path, mode = 1):
    if mode== 1:
        cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
    elif mode ==2:
        cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']
    
    file_path = os.path.join(data_path,'idx/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_llcm(data_path, mode = 1, trial = 0):
    
    random.seed(trial)
    
    if mode== 1:
        cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
    elif mode ==2:
        cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']
        
    file_path = os.path.join(data_path,'idx/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)