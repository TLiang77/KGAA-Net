base_model_path="stage1_model/llcm.pth"

# TVI-LFM training

python main.py --mode train --training_mode RGB_IR_Text --captioner_name Blip --llm_aug --joint_mode uni --Feat_Filter \
    --lr_txt 0.00035 --text_weight_decay 0.0005 --text_weight_decay_bias 0.0005 \
    --fusion_way add --Fix_Visual --training_weight_init $base_model_path\
    --output_path logs/llcm1 \
    --dataset llcm --loss_names id,wrt --test_modality Fusion --eval_start_epoch 0\
