trial=1
python main.py --training_mode RGB_IR --mode train --lr_visual 0.002\
    --dataset regdb --loss_names id,wrt --test_modality IR \
    --output_path logs/regdb/base --trial $trial

