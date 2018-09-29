import numpy as np
import scipy


def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
	Lvec = np.array([vec[e1] for e1, e2 in test_pair])
	Rvec = np.array([vec[e2] for e1, e2 in test_pair])
	sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
	top_lr = [0] * len(top_k)
	for i in range(Lvec.shape[0]):
		rank = sim[i, :].argsort()
		rank_index = np.where(rank == i)[0][0]
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_lr[j] += 1
	top_rl = [0] * len(top_k)
	for i in range(Rvec.shape[0]):
		rank = sim[:, i].argsort()
		rank_index = np.where(rank == i)[0][0]
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_rl[j] += 1
	print('For each left:')
	for i in range(len(top_lr)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
	print('For each right:')
	for i in range(len(top_rl)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))


def get_combine_hits(se_vec, ae_vec, beta, test_pair, top_k=(1, 10, 50, 100)):
	Lvec_se = np.array([se_vec[e1] for e1, e2 in test_pair])
	Rvec_se = np.array([se_vec[e2] for e1, e2 in test_pair])
	sim_se = scipy.spatial.distance.cdist(Lvec_se, Rvec_se, metric='cityblock')
	Lvec_ae = np.array([ae_vec[e1] for e1, e2 in test_pair])
	Rvec_ae = np.array([ae_vec[e2] for e1, e2 in test_pair])
	sim_ae = scipy.spatial.distance.cdist(Lvec_ae, Rvec_ae, metric='cityblock')
	LL = len(test_pair)
	top_lr = [0] * len(top_k)
	for i in range(LL):
		sim = sim_se[i, :] * beta + sim_ae[i, :] * (1.0 - beta)
		rank = sim.argsort()
		rank_index = np.where(rank == i)[0][0]
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_lr[j] += 1
	top_rl = [0] * len(top_k)
	for i in range(LL):
		sim = sim_se[:, i] * beta + sim_ae[:, i] * (1.0 - beta)
		rank = sim.argsort()
		rank_index = np.where(rank == i)[0][0]
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_rl[j] += 1
	print('For each left:')
	for i in range(len(top_lr)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / LL * 100))
	print('For each right:')
	for i in range(len(top_rl)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / LL * 100))
