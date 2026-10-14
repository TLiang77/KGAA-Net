for trial in {1..10}
do
    echo "Trial ${trial}: 准备检查模型文件是否存在..."

    python main.py --mode train \
        --training_mode RGB_IR_Text --captioner_name Blip --joint_mode uni --lr_visual 0.002 \
        --fusion_way add --Fix_Visual --Feat_Filter --llm_aug --training_weight_init stge1_model/regdb \
        --dataset regdb --loss_names id,wrt --test_modality Fusion \
        --trial $trial  --eval_start_epoch 0 \
        --output_path logs/
done
echo 'Done!'