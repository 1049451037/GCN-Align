import tensorflow as tf


class Config:
	language = 'zh_en' # zh_en | ja_en | fr_en
	e1 = 'data/' + language + '/ent_ids_1'
	e2 = 'data/' + language + '/ent_ids_2'
	r1 = 'data/' + language + '/rel_ids_1'
	r2 = 'data/' + language + '/rel_ids_2'
	a1 = 'data/' + language + '/training_attrs_1'
	a2 = 'data/' + language + '/training_attrs_2'
	ill = 'data/' + language + '/ref_ent_ids'
	kg1 = 'data/' + language + '/triples_1'
	kg2 = 'data/' + language + '/triples_2'
	epochs_se = 300
	epochs_ae = 600
	se_dim = 300
	ae_dim = 100
	act_func = tf.nn.relu
	gamma = 3.0  # margin based loss
	k = 25  # number of negative samples for each positive one
	seed = 3  # 30% of seeds
	beta = 0.9  # weight of SE
