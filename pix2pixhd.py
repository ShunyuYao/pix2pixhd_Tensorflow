from activations import *
from conv_base import *
from blocks import *
import sys

class pix2pixHD:
    def __init__(self,opt):
        self.epoch = opt.epoch
        self.batch = opt.batch
        self.n_class = opt.n_class
        self.d_weight = 1/opt.num_d
        self.feat_weight = opt.feat_weight
        self.old_lr = opt.old_lr
        self.save_iter = opt.save_iter
        self.decay_ep = opt.decay_ep
        self.decay_weight = opt.decay_weight
        self.im_width = opt.im_width
        self.im_height = opt.im_high
        self.save_ckpt_ep = opt.save_ckpt_ep
        self.n_im = len(os.listdir(opt.data_dir))
        print('total images: ', self.n_im)
        sys.stdout.flush()

        self.tf_record_dir = opt.tf_record_dir
        self.save_path = opt.save_path    #'./data/pix2pixhd/Logs'
        self.save_im_dir = opt.save_im_dir
        self.ckpt_dir = opt.ckpt_dir    #  './data/pix2pixhd/Logs/model.ckpt-10'
        self.label_dir = opt.label_dir
        self.inst_dir = opt.inst_dir

        self.label = tf.placeholder(tf.int32,[None,self.im_width,self.im_height])
        # self.bound = tf.placeholder(tf.float32,[None,self.im_width,self.im_height])
        self.real_im = tf.placeholder(tf.float32,[None,self.im_width,self.im_height,3])
        # self.k = tf.placeholder(tf.float32,[1])
        # self.b = tf.placeholder(tf.float32,[None,self.im_width,self.im_height,3])
        # process
        step_per_epoch = self.n_im // self.batch
        self.total_steps = self.epoch * step_per_epoch
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.polynomial_decay(self.old_lr,
                                            self.global_step,
                                            self.total_steps,
                                            end_learning_rate = 1e-8
                                            )
        self.onehot = tf.one_hot(self.label, self.n_class)
        # self.bound_ = tf.expand_dims(self.bound,3)
        # self.real_im = ((self.real_im / 255) - 0.5) / 0.5

        self.vggloss = VGGLoss()
        self.lambda_feat = opt.lambda_feat
        self.lr_decay = opt.lr_decay
        self.debug = opt.debug
        self.saved_model = opt.saved_model
        self.restore = opt.restore
        if self.debug:
            self.save_iter = 5

    #############################  data_loader ##################################
    def _extract_fn(self, tf_record):
        features={
            'Label':tf.FixedLenFeature([], tf.string),
            'Real':tf.FixedLenFeature([], tf.string),
        }
        sample = tf.parse_single_example(tf_record, features)

        image_label = tf.decode_raw(sample['Label'], tf.uint8)
        image_label = tf.reshape(image_label, [480, 480, 1])
        image_label = tf.image.resize_images(image_label, [self.im_height, self.im_width])
        image_label = tf.squeeze(image_label)

        image_real = tf.decode_raw(sample['Real'], tf.uint8)
        image_real = tf.reshape(image_real, [480, 480, 3])
        image_real = tf.image.resize_images(image_real, [self.im_height, self.im_width])
        image_real = ((image_real / 255) - 0.5) / 0.5

        return [image_label, image_real]
    # def read_and_decode(self,filename):
    #     filename_queue = tf.train.string_input_producer([filename])
    #     reader = tf.TFRecordReader()
    #     _, serialized_example = reader.read(filename_queue)
    #     features = tf.parse_single_example(serialized_example,
    #                                        features={
    #                                            'Label':tf.FixedLenFeature([], tf.string),
    #                                            'Real':tf.FixedLenFeature([], tf.string),
    #                                        })
    #     image_label = tf.decode_raw(features['Label'], tf.uint8)
    #     image_label = tf.reshape(image_label, [self.im_height, self.im_width])
    #
    #     image_real = tf.decode_raw(features['Real'], tf.uint8)
    #     image_real = tf.reshape(image_real, [self.im_height, self.im_width, 3])
    #
    #     return image_label, image_real
     ###############################################################################

    def build_G(self, x_label, netG='local'):
        with tf.variable_scope('G'):
            # x_concat = tf.concat([x_bound, x_label],3)
            if netG == 'global':
                out = G_base('global_generator', x_label, self.batch)
                out = conv('conv_end',out,7*7,3,1,3,False)
                out = tanh('tanh_end',out)
                return out
            else:
                conv1 = conv('conv1',x_label,7*7,32,1,3,False)
                in1 = tf.nn.relu(ins_norm('ins1', conv1))
                conv2 = conv('conv2',in1,3*3,64,2,0,True)
                ins2 = tf.nn.relu(ins_norm('ins2', conv2))

                x_pool = pool('pool_x', x_label)
                G1 = G_base('G1', x_pool, self.batch)
                G_add = tf.add(G1, ins2, name='G_Add')

                res_1 = res_block('res_1',G_add, dim=64)
                res_2 = res_block('res_2',res_1, dim=64)
                res_3 = res_block('res_3',res_2, dim=64)
                trans1 = conv_trans('trans1',res_3,3*3,32,2,self.batch,True)
                ins_trans1 = tf.nn.relu(ins_norm('ins_trans1', trans1))
                conv_fin = conv('conv_fin',ins_trans1,7*7,3,1,3,False)
                trans_tanh = tanh('trans_tanh',conv_fin)
            return trans_tanh

    def build_D1(self,im,label,reuse):
        with tf.variable_scope('D1',reuse=reuse):
            x_ = tf.concat([im,label],3)
            D = D_base('D',x_)
            return D

    def build_D2(self,im,label,reuse):
        with tf.variable_scope('D2',reuse=reuse):
            x_ = tf.concat([im,label],3)
            x_pool = pool('pool_D',x_)
            D = D_base('D',x_pool)
            return D

        # with tf.variable_scope('Encoder'):
        #     x_encode = G_base('encode',x,self.batch)
        #     return x_encode

    def forward(self):
        # self.x_feat = self.encoder(self.real_im_)

        self.fake_im = self.build_G(self.onehot)
        self.real_D1_out = self.build_D1(self.real_im,self.onehot,False)
        self.fake_D1_out = self.build_D1(self.fake_im,self.onehot,True)

        self.real_D2_out = self.build_D2(self.real_im,self.onehot,False)
        self.fake_D2_out = self.build_D2(self.fake_im,self.onehot,True)
        tf.summary.image('gen_img', (self.fake_im * 0.5 + 0.5)*255)
        tf.summary.image('real_img', (self.real_im * 0.5 + 0.5)*255)

    def cacu_loss(self):
        self.lsgan_d1 = 0.5 * (disc_loss(self.real_D1_out, True) + disc_loss(self.fake_D1_out, False))
        self.lsgan_d2 = 0.5 * (disc_loss(self.real_D2_out, True) + disc_loss(self.fake_D2_out, False))
        # tf.reduce_mean(0.5*tf.square(self.real_D1_out[-1]-1) + 0.5*tf.square(self.fake_D1_out[-1]))
        # self.lsgan_d2 = tf.reduce_mean(0.5*tf.square(self.real_D2_out[-1]-1) + 0.5*tf.square(self.fake_D2_out[-1]))
        self.lsgan_g = disc_loss(self.fake_D1_out, True) + disc_loss(self.fake_D2_out, True)
        # 0.5*tf.reduce_mean(tf.square(self.fake_D2_out[-1]-1)) + 0.5*tf.reduce_mean(tf.square(self.fake_D1_out[-1]-1))
        self.feat_loss = feat_loss(self.real_D1_out, self.fake_D1_out, self.real_D2_out, self.fake_D2_out, self.feat_weight, self.d_weight) * self.lambda_feat
        self.vgg_loss = self.vggloss(self.real_im, self.fake_im) * self.lambda_feat

        tf.summary.scalar('d1_loss',self.lsgan_d1)
        tf.summary.scalar('d2_loss',self.lsgan_d2)
        tf.summary.scalar('g_loss',self.lsgan_g)
        tf.summary.scalar('feat_loss',self.feat_loss)
        tf.summary.scalar('vgg_loss', self.vgg_loss)

    def train(self):
        lr = self.old_lr
        self.forward()
        self.cacu_loss()
        D1_vars = [var for var in tf.all_variables() if 'D1' in var.name]
        D2_vars = [var for var in tf.all_variables() if 'D2' in var.name]
        G_vars = [var for var in tf.all_variables() if 'G' in var.name]
        # encoder_vars = [var for var in tf.all_variables() if 'Encoder' in var.name]
        if not self.lr_decay:
            optim_D1 = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.lsgan_d1, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D1'))
            optim_D2 = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.lsgan_d2, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D2'))
            optim_G_ALL = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.lsgan_g + self.feat_loss + self.vgg_loss,
                                                              var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))
        else:
            optim_D1 = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.lsgan_d1,
                                                                           var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D1'),
                                                                           global_step=self.global_step)
            optim_D2 = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.lsgan_d2,
                                                                           var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D2'),
                                                                           global_step=self.global_step)
            optim_G_ALL = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.lsgan_g + self.feat_loss + self.vgg_loss,
                                                              var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'),
                                                              global_step=self.global_step)
            tf.summary.scalar('lr', self.lr)


        dataset = tf.data.TFRecordDataset([self.tf_record_dir])
        # if self.debug:
        #     dataset = dataset.take(100)
        dataset = dataset.map(self._extract_fn)
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.batch(self.batch)
        iterator = dataset.make_initializable_iterator()
        next_data = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            merge = tf.summary.merge_all()
            graph = tf.summary.FileWriter(self.save_path, sess.graph)
            Saver = tf.train.Saver(max_to_keep=5)
            if self.restore:
                restore_vars = []
                for var in tf.all_variables():
                    if 'G' in var.name:
                        restore_vars.append(var)
                    elif 'D1' in var.name:
                        restore_vars.append(var)
                    elif 'D2' in var.name:
                        restore_vars.append(var)
                restore_saver = tf.train.Saver(var_list=restore_vars)
                restore_saver.restore(sess, self.ckpt_dir)
                print('-----restore sucess-----')
                sys.stdout.flush()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for ep in range(self.epoch):
                print('epoch: ', ep+1)
                sys.stdout.flush()
                for j in range(self.n_im//self.batch):
                    print('\t step: ', j+1)
                    sys.stdout.flush()
                    dataset = sess.run(next_data)
                    label_fed, real_im_fed = dataset[0], dataset[1]
                    dict_ = {self.label: label_fed,
                             self.real_im: real_im_fed}

                    step = int(ep*(self.n_im // self.batch) + j)
                    total_step = j*self.batch + ep*self.n_im
                    _ = sess.run([optim_G_ALL], feed_dict=dict_)
                    _, _ = sess.run([optim_D1, optim_D2], feed_dict=dict_)
                    # print('real:', sess.run(self.real_im, feed_dict={self.real_im: real_im_fed})[0,:32,:32,:])
                    # print('fake:', (fake_im[0, :, :, :] * 0.5 + 0.5) * 255)

                    if self.saved_model and self.debug and ep == 0 and j == 0:
                        tf.saved_model.simple_save(sess,
                                                   export_dir=os.path.join(self.save_path, 'net_all'),
                                                   inputs={'input': self.label},
                                                   outputs={'output': self.fake_im})

                    if (j+1) % self.save_iter == 0:
                        print('save iter')
                        sys.stdout.flush()
                        Merge = sess.run(merge, feed_dict=dict_)
                        graph.add_summary(Merge, step)
                    #     Save_im((fake_im * 0.5 + 0.5) * 255, self.save_im_dir, ep, j)

                if (ep+1) % self.save_ckpt_ep == 0:
                    print('save epoch')
                    sys.stdout.flush()
                    num_trained = total_step
                    Saver.save(sess, self.save_path + '/' + 'model.ckpt', num_trained)
                    print('save success at num images trained: ',num_trained)
                    sys.stdout.flush()
                if ep > self.decay_ep:
                    lr = self.old_lr - ep / self.decay_weight

            if self.saved_model:
                tf.saved_model.simple_save(sess,
                                           export_dir=os.path.join(self.save_path, 'net_all'),
                                           inputs={'input': self.label},
                                           outputs={'output': self.fake_im})
            coord.request_stop()
            coord.join(threads)
            return True

    def Load_model(self):
        #  b_fed is a feature vector extracted from the encoder's encoding and needs to be specified
        #   by human (by clustering the results of the trained encoder).
        # self.x_feat = self.encoder(self.real_im_)
        self.fake_im = self.build_G(self.onehot)
        G_vars = [var for var in tf.all_variables() if 'G' in var.name]

        dataset = tf.data.TFRecordDataset([self.tf_record_dir])
        # if self.debug:
        #     dataset = dataset.take(100)
        dataset = dataset.map(self._extract_fn)
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.batch(self.batch)
        iterator = dataset.make_initializable_iterator()
        next_data = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            sess.run(iterator.initializer)

            graph = tf.summary.FileWriter(self.save_path, sess.graph)
            Saver = tf.train.Saver(var_list=G_vars)
            Saver.restore(sess, self.ckpt_dir)

            dataset = sess.run(next_data)
            label_fed, real_im_fed = dataset[0], dataset[1]
            dict_ = {self.label: label_fed,
                     self.real_im: real_im_fed}
            ims = sess.run(self.fake_im, feed_dict=dict_)

            if self.saved_model:
                tf.saved_model.simple_save(sess,
                                           export_dir=os.path.join(self.save_path, 'net_G'),
                                           inputs={'input': self.label},
                                           outputs={'output': self.fake_im})

            # label_fed, bound_fed = load_data(self.label_dir,self.inst_dir)
            # #  k_fed must be zero, which means that the actual output of the encoder is not considered, because there is no ideal result color map when used.
            # #      (The characteristic input of G is: output(encoder)*k+b, k=1 during training, b=0)
            # k_fed = np.zeros([1],np.float32)
            #
            # real_im_fed = np.zeros([np.shape(label_fed)[0],self.im_width,self.im_height,3],np.float32)
            #
            # dict_ = {self.label:label_fed,self.bound:bound_fed,self.real_im:real_im_fed,self.k:k_fed,self.b:b_fed}


            # Save_im(ims, self.save_im_dir, 0, 0)
            print(np.shape(ims))


class VGGLoss(tf.keras.Model):
    def __init__(self):
        super(VGGLoss, self).__init__(name='VGGLoss')
        self.vgg = Vgg19()
        # self.criterion = tf.keras.losses.mean_absolute_error  # lambda ta, tb: tf.reduce_mean(tf.abs(ta - tb))
        self.layer_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def call(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            y_vgg_temp = tf.stop_gradient(y_vgg[i])
            loss += self.layer_weights[i] * tf.reduce_mean(tf.abs(x_vgg[i] - y_vgg_temp))
        return loss


# the keras vgg19 do not count ReLU layer
# the index is a little different
class Vgg19(tf.keras.Model):
    def __init__(self, trainable=False):
        super(Vgg19, self).__init__(name='Vgg19')
        vgg_pretrained_features = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False)
        if trainable is False:
            vgg_pretrained_features.trainable = False
        vgg_pretrained_features = vgg_pretrained_features.layers
        self.slice1 = tf.keras.Sequential()
        self.slice2 = tf.keras.Sequential()
        self.slice3 = tf.keras.Sequential()
        self.slice4 = tf.keras.Sequential()
        self.slice5 = tf.keras.Sequential()
        for x in range(1, 2):
            self.slice1.add(vgg_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add(vgg_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add(vgg_pretrained_features[x])
        for x in range(8, 13):
            self.slice4.add(vgg_pretrained_features[x])
        for x in range(13, 18):
            self.slice5.add(vgg_pretrained_features[x])

    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
