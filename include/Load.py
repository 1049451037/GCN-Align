import numpy as np


# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1):
	print('loading a file...' + fn)
	ret = []
	with open(fn, encoding='utf-8') as f:
		for line in f:
			th = line[:-1].split('\t')
			x = []
			for i in range(num):
				x.append(int(th[i]))
			ret.append(tuple(x))
	return ret


def get_ent2id(fns):
	ent2id = {}
	for fn in fns:
		with open(fn, 'r', encoding='utf-8') as f:
			for line in f:
				th = line[:-1].split('\t')
				ent2id[th[1]] = int(th[0])
	return ent2id


# The most frequent attributes are selected to save space
def loadattr(fns, e, ent2id):
	cnt = {}
	for fn in fns:
		with open(fn, 'r', encoding='utf-8') as f:
			for line in f:
				th = line[:-1].split('\t')
				if th[0] not in ent2id:
					continue
				for i in range(1, len(th)):
					if th[i] not in cnt:
						cnt[th[i]] = 1
					else:
						cnt[th[i]] += 1
	fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
	attr2id = {}
	for i in range(1000):
		attr2id[fre[i][0]] = i
	attr = np.zeros((e, 1000), dtype=np.float32)
	for fn in fns:
		with open(fn, 'r', encoding='utf-8') as f:
			for line in f:
				th = line[:-1].split('\t')
				if th[0] in ent2id:
					for i in range(1, len(th)):
						if th[i] in attr2id:
							attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
	return attr
