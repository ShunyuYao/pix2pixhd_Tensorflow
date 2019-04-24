# pix2pixhd_Tensorflow
Pix2pix primary architecture based solely on tensorflow
## Create tfrecord
1.&#8195;In order to speed up the reading of data, the data is first formatted into tfrecords format. In this case, Cityscapes is used.First extract the label of the dataset, take a picture, and generate a boundary map:<br>
&#8195;&#8195;&#8195;python ./data/pix2pixHD/get_data.py --data_dir="./data/dataset/cityscapes/train" -- save_dir="./data/pix2pix/data" --phase="train"<br>
2.&#8195;Then generating the tfrecord file:&#8195;python --file_label_dir="./data/pix2pix/data/train/label" --TFRECORD_DIR="./data/pix2pix/data/train/" --phase="train" --epoch=1000<br>
3.&#8195;The batch of training was assigned in advance because when the data was read with 'tf.train.shuffle_batch', although the sampling order of the samples was random, there was no guarantee that each sample would appear in a training period, so use 'tf.train. Batch' reads data. If you are unsure of the trained batch, you can set a larger value (but not too large).<br>
## The difference between training and inference
&#8195;During training, the code of the real picture is used as a feature input to the global generator; after the training is finished, the output feature space of the encoder can be separately clustered to obtain a specific code of a certain feature (such as the texture of the road). : asphalt road or stone road, as described in the paper). In the test, you need to specify the feature information manually. This function is still in the process of perfection, but you can enter the 'b_fed' in ‘Load_model’ by entering pix2pixhd to implement manual input.<br>
## Implementation of feature selection:
&#8195;&#8195;For the output of the encoder, add two control quantities, k, b. Output = output(encoder) * k + b. When training, k=1, b=0; when inference, k=0, b is a manually added feature value.<br>
## Train:
&#8195;&#8195;python ./train_test.py --phase="train" --epoch=1 --batch=1 --n_class=18 --num_d=2 --save_iter=5 --decay_ep=10 --im_high=128 --im_width=128 --feat_weight=10 --old_lr=0.002 --decay_weight=20 --sace_ckpt_iter=2 --data_dir="./datasets/pix2pix/data" --tf_record_dir="./datasets/tf_train" --save_path="./datasets/train/Logs" --save_im_dir="./datasets/train/Logs" --ckpt_dir="./datasets/train/Logs" --label_dir="./datasets/train/Logs" --inst_dir="./datasets/train/Logs"<br>
&#8195;At training time,the input of ckpt, label_dir, and ins_dir is not required during training, just for the setting of argparse.

python ./train_test.py \
--phase="train" --epoch=1 --batch=1 --n_class=18 \
--num_d=2 --decay_ep=10 --im_high=128 \
--im_width=128 --feat_weight=10 --old_lr=0.002 \
--decay_weight=20 \
--data_dir="../deep-imit-train-bak/target/gray_maskgirl/train_img" \
--tf_record_dir="./datasets/tf_train/train.tfrecords" \
--save_path="./datasets/train/Logs" \
--save_im_dir="./datasets/train/Logs" \
--ckpt_dir="./datasets/train/Logs" \
--label_dir="./datasets/train/Logs" \
--inst_dir="./datasets/train/Logs" \
--saved_model \
--debug

python make_tfrecord.py --file_label_dir /home/projects/deep-imit-train-bak/target/gray_maskgirl/train_label \
--file_img_dir /home/projects/deep-imit-train-bak/target/gray_maskgirl/train_img \
--TFRECORD_DIR /home/projects/pix2pixhd_Tensorflow/datasets \
--phase train \
--epoch 1
