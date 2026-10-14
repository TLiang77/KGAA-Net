base_model_path="stage1/sysu.pth"
test_model_path="logs/sysu/FV_Filter/Baseline_train[RGB_IR_Text]_joint[uni]_Blip_add_LLM_0.5_id,wrt_Fix_Visual_Filtered/models/model_Fusion_3.pth"

python main.py --training_mode RGB_IR_Text --mode test --captioner_name Blip --joint_mode uni \
    --lr_txt 0.00035 --text_weight_decay 0.0005 --text_weight_decay_bias 0.0005 \
    --fusion_way add --Fix_Visual --Feat_Filter --llm_aug --test_model_path $test_model_path\
    --training_weight_init $base_model_path --output_path logs/sysu \
    --dataset sysu --loss_names id,wrt --test_modality Fusion --LOG4TEST --eval_start_epoch 0 \
    --use_der \