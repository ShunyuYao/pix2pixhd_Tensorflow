python ./train_test.py \
--phase="train" --epoch=1 --batch=1 --n_class=18 \
--num_d=2 --decay_ep=10 --im_high=128 \
--im_width=128 --feat_weight=10 --old_lr=0.0002 \
--decay_weight=20 \
--data_dir="../deep-imit-train-bak/target/gray_maskgirl/train_img" \
--tf_record_dir="./datasets/tf_train/train.tfrecords" \
--save_path="./datasets/train/Logs" \
--save_im_dir="./datasets/train/Logs" \
--ckpt_dir="./datasets/train/Logs" \
--label_dir="./datasets/train/Logs" \
--inst_dir="./datasets/train/Logs" \
--debug
# --saved_model \
