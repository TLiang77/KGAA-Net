# 循环 trial 从 1 到 10
for trial in 8
do
    echo "========================================="
    echo "Starting Trial: $trial"
    echo "========================================="

    test_model_path="logs/3regdb/FV_Filter/Baseline_${trial}_train[RGB_IR_Text]_joint[uni]_Blip_add_LLM_0.5_id,wrt_Fix_Visual_Filtered/models/model_Fusion.pth"
    base_model_path="stage1/regdb_${trial}.pth" # 这个路径在每次循环中保持不变

    # 执行 Python 命令
    python main.py --mode test \
        --training_mode RGB_IR_Text --captioner_name Blip --joint_mode uni --lr_visual 0.002 \
        --fusion_way add --Fix_Visual --Feat_Filter --llm_aug --training_weight_init $base_model_path \
        --dataset regdb --loss_names id,wrt --test_modality Fusion --LOG4TEST \
         --trial $trial --test_model_path $test_model_path \
        --output_path logs/

    echo "Finished Trial: $trial"
    echo ""
done

echo "All trials completed."

