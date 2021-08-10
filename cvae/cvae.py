import tensorflow as tf
import numpy as np

from cvae.util import *


class cfr_net(object):
    """
    cvae implements the CECVAE-ORTH proposed by "Disentangled Representation Learning with Variational\
     Inference: A Method for Treatment Effect Estimation" (NeurIPS 2021 Conference Paper2148)

    The code of CECVAE-ORTH is built upon the Counterfactual regression (CFR) work of Johansson, Shalit & Sontag (2016) and \
    Shalit, Johansson & Sontag (2016), https://github.com/clinicalml/cfrnet.
    The parameter searching, network training and evaluation follow the procedures of CFR to ensure fair comparison.

    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, dims):
        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        ''' Start Session '''
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        ''' Initialize input placeholders '''
        self.x = tf.placeholder("float", shape=[None, dims], name='x')  # Features
        self.t = tf.placeholder("float", shape=[None, 1], name='t')  # Treatent
        self.y = tf.placeholder("float", shape=[None, 1], name='y')  # Outcome
        self.p = tf.placeholder("float", name='p_treated')

        ''' Parameter placeholders '''
        self.dropout_in = tf.placeholder("float", name='dropout_in')
        self.dropout_out = tf.placeholder("float", name='dropout_out')

        self._build_graph(dims)

    def _build_fully_connected_layers(self, input, n_layers, n_size, keep_prob, var_scope, reuse=False):
        with tf.variable_scope(var_scope, reuse=reuse):
            h = [input]
            for i in range(0, n_layers):
                h.append(tf.nn.dropout(
                    tf.layers.dense(h[i], n_size, self.nonlin,
                                    kernel_initializer=tf.random_normal_initializer(
                                        stddev=0.1 / np.sqrt(h[i].shape[-1].value)))
                    , keep_prob))
            return h[-1]

    @staticmethod
    def _get_sample_from_dist(mu, log_square_sigma):
        sigma = tf.exp(0.5 * log_square_sigma)
        sample = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        return sample

    @staticmethod
    def _KL_distance(mu1, log_square_sigma1, mu2, log_square_sigma2):
        KL = 0.5 * (tf.divide(tf.exp(log_square_sigma1) + tf.square(mu1 - mu2),
                              tf.exp(log_square_sigma2)) - 1 - log_square_sigma1 + log_square_sigma2)
        return tf.reduce_mean(tf.reduce_sum(KL, -1))

    def _cal_orth(self, rep1, rep2):
        orth_loss = tf.reduce_mean(tf.abs(tf.divide(tf.reduce_sum(tf.multiply(rep1, rep2), 1),
                                                    tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(rep1), 1)),
                                                                tf.sqrt(tf.reduce_sum(tf.square(rep2), 1))
                                                                )
                                                    )
                                          )
                                   )
        return orth_loss

    def _build_graph(self, dims):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """
        z_t_en_mu = self._build_fully_connected_layers(self.x, FLAGS.n_in, FLAGS.dim_in, self.dropout_in,
                                                       "z_t_en_mu")
        z_t_en_sigma = self._build_fully_connected_layers(self.x, FLAGS.n_in, FLAGS.dim_in, self.dropout_in,
                                                          "z_t_en_sigma")
        z_c_en_mu = self._build_fully_connected_layers(self.x, FLAGS.n_in, FLAGS.dim_in, self.dropout_in,
                                                       "z_c_en_mu")
        z_c_en_sigma = self._build_fully_connected_layers(self.x, FLAGS.n_in, FLAGS.dim_in, self.dropout_in,
                                                          "z_c_en_sigma")
        z_y_en_mu = self._build_fully_connected_layers(self.x, FLAGS.n_in, FLAGS.dim_in, self.dropout_in,
                                                       "z_y_en_mu")
        z_y_en_sigma = self._build_fully_connected_layers(self.x, FLAGS.n_in, FLAGS.dim_in, self.dropout_in,
                                                          "z_y_en_sigma")

        z_t_de_mu = self._build_fully_connected_layers(tf.concat([self.x, self.t], -1), FLAGS.n_in,
                                                       FLAGS.dim_in, self.dropout_in, "z_t_de_mu")
        z_t_de_sigma = self._build_fully_connected_layers(tf.concat([self.x, self.t], -1), FLAGS.n_in,
                                                          FLAGS.dim_in, self.dropout_in, "z_t_de_sigma")
        z_c_de_mu = self._build_fully_connected_layers(tf.concat([self.x, self.t, self.y], -1), FLAGS.n_in,
                                                       FLAGS.dim_in, self.dropout_in, "z_c_de_mu")
        z_c_de_sigma = self._build_fully_connected_layers(tf.concat([self.x, self.t, self.y], -1), FLAGS.n_in,
                                                          FLAGS.dim_in, self.dropout_in, "z_c_de_sigma")
        z_y_de_mu = self._build_fully_connected_layers(tf.concat([self.x, self.y], -1), FLAGS.n_in,
                                                       FLAGS.dim_in, self.dropout_in, "z_y_de_mu")
        z_y_de_sigma = self._build_fully_connected_layers(tf.concat([self.x, self.y], -1), FLAGS.n_in,
                                                          FLAGS.dim_in, self.dropout_in, "z_y_de_sigma")

        z_t_sample_en = self._get_sample_from_dist(z_t_en_mu, z_t_en_sigma)
        z_c_sample_en = self._get_sample_from_dist(z_c_en_mu, z_c_en_sigma)
        z_y_sample_en = self._get_sample_from_dist(z_y_en_mu, z_y_en_sigma)
        z_t_sample_de = self._get_sample_from_dist(z_t_de_mu, z_t_de_sigma)
        z_c_sample_de = self._get_sample_from_dist(z_c_de_mu, z_c_de_sigma)
        z_y_sample_de = self._get_sample_from_dist(z_y_de_mu, z_y_de_sigma)

        zt_zc_concat_de = tf.concat([z_t_sample_de, z_c_sample_de], -1)
        # zt_zc_concat_en = tf.concat([z_t_sample_en, z_c_sample_en], -1)
        pred_t_de = tf.layers.dense(
            self._build_fully_connected_layers(zt_zc_concat_de, FLAGS.n_out, FLAGS.dim_out, self.dropout_out,
                                               't_out_net'), 1, name='pred_t_logit')
        pred_t_en = tf.layers.dense(
            self._build_fully_connected_layers(z_t_sample_en, FLAGS.n_out, FLAGS.dim_out, self.dropout_out,
                                               't_out_net_test'), 1, name='pred_t_logit_test')

        i0 = tf.to_int32(tf.where(self.t < 1)[:, 0])
        i1 = tf.to_int32(tf.where(self.t > 0)[:, 0])

        zc_zy_concat_de = tf.concat([z_c_sample_de, z_y_sample_de], -1)
        zc_zy_concat0_de = tf.gather(zc_zy_concat_de, i0)
        zc_zy_concat1_de = tf.gather(zc_zy_concat_de, i1)

        z_y_sample0_en = tf.gather(z_y_sample_en, i0)
        z_y_sample1_en = tf.gather(z_y_sample_en, i1)

        pred_y0_de = tf.layers.dense(
            self._build_fully_connected_layers(zc_zy_concat0_de, FLAGS.n_out, FLAGS.dim_out, self.dropout_out,
                                               'y0_out_net'), 1, name='pred_y0_logit')
        pred_y1_de = tf.layers.dense(
            self._build_fully_connected_layers(zc_zy_concat1_de, FLAGS.n_out, FLAGS.dim_out, self.dropout_out,
                                               'y1_out_net'), 1, name='pred_y1_logit')

        pred_y0_en = tf.layers.dense(
            self._build_fully_connected_layers(z_y_sample0_en, FLAGS.n_out, FLAGS.dim_out, self.dropout_out,
                                               'y0_out_net_test'), 1, name='pred_y0_logit_test')
        pred_y1_en = tf.layers.dense(
            self._build_fully_connected_layers(z_y_sample1_en, FLAGS.n_out, FLAGS.dim_out, self.dropout_out,
                                               'y1_out_net_test'), 1, name='pred_y1_logit_test')

        pred_y_de = tf.dynamic_stitch([i0, i1], [pred_y0_de, pred_y1_de])
        pred_y_en = tf.dynamic_stitch([i0, i1], [pred_y0_en, pred_y1_en])

        self.t_classif_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_t_de, labels=self.t)) + \
                              FLAGS.coef_t_pred*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_t_en, labels=self.t))


        if FLAGS.loss == "log":
            self.y_predict_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_y_de, labels=self.y)) + \
                                  FLAGS.coef_y_pred*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_y_en, labels=self.y))
        else:
            # self.y_predict_loss = tf.reduce_mean(self.sample_weight * tf.square(self.y - pred_y_de))
            self.y_predict_loss = tf.reduce_mean(tf.square(self.y - pred_y_de)) + \
                                  FLAGS.coef_y_pred*tf.reduce_mean(tf.square(self.y - pred_y_en))
            # self.y_predict_loss = tf.reduce_mean(tf.abs(self.y - pred_y_de))

        KL_zt = self._KL_distance(z_t_de_mu, z_t_de_sigma, z_t_en_mu, z_t_en_sigma)
        KL_zc = self._KL_distance(z_c_de_mu, z_c_de_sigma, z_c_en_mu, z_c_en_sigma)
        KL_zy = self._KL_distance(z_y_de_mu, z_y_de_sigma, z_y_en_mu, z_y_en_sigma)

        orth_loss_t_y = self._cal_orth(z_t_sample_en, z_y_sample_en)
        orth_loss_t_c = self._cal_orth(z_t_sample_en, z_c_sample_en)
        orth_loss_y_c = self._cal_orth(z_y_sample_en, z_c_sample_en)

        orth_loss = orth_loss_t_y + orth_loss_t_c + orth_loss_y_c

        self.tot_loss = self.t_classif_loss + self.y_predict_loss
        self.tot_loss = self.tot_loss + KL_zt + KL_zc + KL_zy
        self.tot_loss = self.tot_loss + FLAGS.coef_orth_loss * orth_loss

        if FLAGS.loss == "log":
            self.pred_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_y_en, labels=self.y))
        else:
            self.pred_loss = tf.sqrt(tf.reduce_mean(tf.square(self.y - pred_y_en)))

        self.imb_dist, imb_mat = wasserstein(z_t_sample_en, self.t, 0.5, lam=FLAGS.wass_lambda,
                                             its=FLAGS.wass_iterations,
                                             sq=False, backpropT=FLAGS.wass_bpt)
        if FLAGS.loss == "log":
            self.output = tf.nn.sigmoid(pred_y_en)
        else:
            self.output = pred_y_en
