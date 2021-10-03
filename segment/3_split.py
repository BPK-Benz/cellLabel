import os
import random

if __name__ == "__main__":

	# search

	segment_root = 'data/output_segment'
	data = []

	replicates = os.listdir(segment_root)
	replicates = sorted(replicates)
	for r in replicates:
		plates = os.listdir(os.path.join(segment_root, r))
		plates = sorted(plates)
		for p in plates:
			sets = os.listdir(os.path.join(segment_root, r, p))
			sets = sorted(sets)
			for s in sets:
				files = os.listdir(os.path.join(segment_root, r, p, s))
				files = sorted(files)
				for f in files:
					sample = os.path.join(r, p, s, f).replace('.png', '')
					data.append(sample)

	# split

	ratio = .8
	train = random.sample(data, k=int(len(data)*ratio))
	test = [s for s in data if not s in train]

	print('Training sample =', len(train))
	print('Testing sample  =', len(test))

	# save

	train_path = 'source/train.txt'
	with open(train_path, 'w') as f:
	    f.write('\n'.join(train))

	test_path = 'source/test.txt'
	with open(test_path, 'w') as f:
	    f.write('\n'.join(test))
