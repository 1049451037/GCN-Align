from include.Config import Config
import tensorflow as tf
from include.Model import build_SE, build_AE, training
from include.Test import get_hits, get_combine_hits
from include.Load import *

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

if __name__ == '__main__':
	e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
	print(e)
	ILL = loadfile(Config.ill, 2)
	illL = len(ILL)
	np.random.shuffle(ILL)
	train = np.array(ILL[:illL // 10 * Config.seed])
	test = ILL[illL // 10 * Config.seed:]
	KG1 = loadfile(Config.kg1, 3)
	KG2 = loadfile(Config.kg2, 3)
	# build SE
	output_layer, loss = build_SE(Config.se_dim, Config.act_func, Config.gamma, Config.k, e, train, KG1 + KG2)
	se_vec, J = training(output_layer, loss, 25, Config.epochs_se, train, e, Config.k)
	print('loss:', J)
	print('Result of SE:')
	get_hits(se_vec, test)
	ent2id = get_ent2id([Config.e1, Config.e2])
	attr = loadattr([Config.a1, Config.a2], e, ent2id)
	output_layer, loss = build_AE(attr, Config.ae_dim, Config.act_func, Config.gamma, Config.k, e, train, KG1 + KG2)
	ae_vec, J = training(output_layer, loss, 5, Config.epochs_ae, train, e, Config.k)
	print('loss:', J)
	# print('Result of AE:')
	# get_hits(ae_vec, test)
	# np.save('se_vec.npy', se_vec) # save embeddings
	# np.save('ae_vec.npy', ae_vec)
	# print(se_vec.shape, ae_vec.shape)
	# print('embeddings are saved.')
	print('Result of SE+AE:')
	get_combine_hits(se_vec, ae_vec, Config.beta, test)
