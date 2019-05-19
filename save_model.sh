python ./train_test.py \
--phase="test" --epoch=50 --batch=1 --n_class=18 \
--num_d=2 --decay_ep=50 --im_high=128 \
--im_width=128 --feat_weight=10 --old_lr=0.0002 \
--decay_weight=100 \
--data_dir="../deep-imit-train-bak/target/gray_maskgirl/train_img" \
--tf_record_dir="./datasets/tf_train/train.tfrecords" \
--save_path="./datasets/train/Logs" \
--save_im_dir="./datasets/train/Logs" \
--ckpt_dir="./datasets/train/Logs/model.ckpt-232392" \
--label_dir="./datasets/train/Logs" \
--inst_dir="./datasets/train/Logs" \
--restore \
--saved_model
# --debug
# --saved_model \
