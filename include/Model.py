import math
from .Init import *


def func(KG):
	head = {}
	cnt = {}
	for tri in KG:
		if tri[1] not in cnt:
			cnt[tri[1]] = 1
			head[tri[1]] = set([tri[0]])
		else:
			cnt[tri[1]] += 1
			head[tri[1]].add(tri[0])
	r2f = {}
	for r in cnt:
		r2f[r] = len(head[r]) / cnt[r]
	return r2f


def ifunc(KG):
	tail = {}
	cnt = {}
	for tri in KG:
		if tri[1] not in cnt:
			cnt[tri[1]] = 1
			tail[tri[1]] = set([tri[2]])
		else:
			cnt[tri[1]] += 1
			tail[tri[1]].add(tri[2])
	r2if = {}
	for r in cnt:
		r2if[r] = len(tail[r]) / cnt[r]
	return r2if


def get_mat(e, KG):
	r2f = func(KG)
	r2if = ifunc(KG)
	du = [1] * e
	for tri in KG:
		if tri[0] != tri[2]:
			du[tri[0]] += 1
			du[tri[2]] += 1
	M = {}
	for tri in KG:
		if tri[0] == tri[2]:
			continue
		if (tri[0], tri[2]) not in M:
			M[(tri[0], tri[2])] = math.sqrt(math.sqrt(r2if[tri[1]]))
		else:
			M[(tri[0], tri[2])] += math.sqrt(math.sqrt(r2if[tri[1]]))
		if (tri[2], tri[0]) not in M:
			M[(tri[2], tri[0])] = math.sqrt(math.sqrt(r2f[tri[1]]))
		else:
			M[(tri[2], tri[0])] += math.sqrt(math.sqrt(r2f[tri[1]]))
	for i in range(e):
		M[(i, i)] = 1
	return M, du


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
	print('getting a sparse tensor...')
	M, du = get_mat(e, KG)
	ind = []
	val = []
	for fir, sec in M:
		ind.append((sec, fir))
		val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
	M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])
	return M


# add a layer
def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
	inlayer = tf.nn.dropout(inlayer, 1 - dropout)
	print('adding a layer...')
	w0 = init([1, dimension])
	tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
	if act_func is None:
		return tosum
	else:
		return act_func(tosum)


def add_full_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
	inlayer = tf.nn.dropout(inlayer, 1 - dropout)
	print('adding a layer...')
	w0 = init([dimension_in, dimension_out])
	tosum = tf.sparse_tensor_dense_matmul(M, tf.matmul(inlayer, w0))
	if act_func is None:
		return tosum
	else:
		return act_func(tosum)


# se input layer
def get_se_input_layer(e, dimension):
	print('adding the se input layer...')
	ent_embeddings = tf.Variable(tf.truncated_normal([e, dimension], stddev=1.0 / math.sqrt(e)))
	return tf.nn.l2_normalize(ent_embeddings, 1)


# ae input layer
def get_ae_input_layer(attr):
	print('adding the ae input layer...')
	return tf.constant(attr)


# get loss node
def get_loss(outlayer, ILL, gamma, k):
	print('getting loss...')
	left = ILL[:, 0]
	right = ILL[:, 1]
	t = len(ILL)
	left_x = tf.nn.embedding_lookup(outlayer, left)
	right_x = tf.nn.embedding_lookup(outlayer, right)
	A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
	neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
	neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
	neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
	neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
	B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
	C = - tf.reshape(B, [t, k])
	D = A + gamma
	L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
	neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
	neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
	neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
	neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
	B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
	C = - tf.reshape(B, [t, k])
	L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
	return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


def build_SE(dimension, act_func, gamma, k, e, ILL, KG):
	tf.reset_default_graph()
	input_layer = get_se_input_layer(e, dimension)
	M = get_sparse_tensor(e, KG)
	hidden_layer = add_diag_layer(input_layer, dimension, M, act_func, dropout=0.0)
	output_layer = add_diag_layer(hidden_layer, dimension, M, None, dropout=0.0)
	loss = get_loss(output_layer, ILL, gamma, k)
	return output_layer, loss


def build_AE(attr, dimension, act_func, gamma, k, e, ILL, KG):
	tf.reset_default_graph()
	input_layer = get_ae_input_layer(attr)
	M = get_sparse_tensor(e, KG)
	hidden_layer = add_full_layer(input_layer, attr.shape[1], dimension, M, act_func, dropout=0.0)
	output_layer = add_diag_layer(hidden_layer, dimension, M, None, dropout=0.0)
	loss = get_loss(output_layer, ILL, gamma, k)
	return output_layer, loss


def training(output_layer, loss, learning_rate, epochs, ILL, e, k):
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # optimizer can be changed
	print('initializing...')
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	print('running...')
	J = []
	t = len(ILL)
	ILL = np.array(ILL)
	L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
	neg_left = L.reshape((t * k,))
	L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
	neg2_right = L.reshape((t * k,))
	for i in range(epochs):
		if i % 10 == 0:
			neg2_left = np.random.choice(e, t * k)
			neg_right = np.random.choice(e, t * k)
		sess.run(train_step, feed_dict={"neg_left:0": neg_left,
										"neg_right:0": neg_right,
										"neg2_left:0": neg2_left,
										"neg2_right:0": neg2_right})
		if (i + 1) % 20 == 0:
			th = sess.run(loss, feed_dict={"neg_left:0": neg_left,
										   "neg_right:0": neg_right,
										   "neg2_left:0": neg2_left,
										   "neg2_right:0": neg2_right})
			J.append(th)
			print('%d/%d' % (i + 1, epochs), 'epochs...')
	outvec = sess.run(output_layer)
	sess.close()
	return outvec, J
