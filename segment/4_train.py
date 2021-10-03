import os
import cv2
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

from utils.unet import UNet
from utils.cell_dataset2 import cell_dataset

if __name__ == "__main__":
	
	#@title load datasets

	train_split_path = 'source/train.txt'
	train_dataset = cell_dataset(
		split = train_split_path, 
		augment = True
	)
	train_loader = DataLoader(
		train_dataset, batch_size=2,
		shuffle=True, num_workers=4
	)

	test_split_path = 'source/test.txt'
	test_dataset = cell_dataset(
		split = test_split_path, 
		augment = False
	)
	test_loader = DataLoader(
		test_dataset, batch_size=2,
		shuffle=False, num_workers=4
	)

	#@title load model

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net = UNet(n_channels=3, n_classes=3)
	net.to(device=device)

	# load pre-trained
	net1_path = 'source/latest.pth'
	if os.path.exists(net1_path):
		net.load_state_dict(torch.load(net1_path))
		print('continue from', net1_path)

	#@title prepare for training

	# learning_rate = 0.001     # epoch 0
	# learning_rate = 0.00001   # epoch 100
	learning_rate = 0.0000001   # epoch 100
	momentum = 0.9
	momentum2 = 0.99
	eps = 1e-8
	weight_decay = 0.00001
	optimizer = optim.RMSprop(net.parameters(), lr = learning_rate, weight_decay=1e-8)
	# criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
	criterion = nn.BCEWithLogitsLoss()

	#@title training

	epochs = 300
	test_every = 5
	save_every = 5

	for epoch in range(epochs):

		net.train()
		epoch_loss = 0

		for images, labels in train_loader:

			images = images.to(device=device, dtype=torch.float32)
			labels = labels.to(device=device, dtype=torch.float32)

			pred_masks = net(images)

			loss = criterion(pred_masks, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()

		epoch_loss /= len(train_loader)

		print("[ Training: epoch: %3d, loss: %6.4f ]"%(epoch, epoch_loss))

		# Test model
		if epoch % test_every == 0:

			net.eval()
			test_loss = 0

			with torch.no_grad():

				for images_test, masks_test in test_loader:

					images_test = images_test.to(device=device, dtype=torch.float32)
					masks_test = masks_test.to(device=device, dtype=torch.float32)

					pred_masks_test = net(images_test)
		
					loss = criterion(pred_masks_test, masks_test)
					test_loss += loss.item()

			test_loss /= len(test_loader)

			print("[ Testing: epoch: %3d, loss: %6.4f ]"%(epoch, test_loss))

		# Save model
		if epoch % save_every == 0:

			net1_path = 'source/latest.pth'
			net2_path = 'source/epoch_' + str(epoch).zfill(3) + '.pth'
			torch.save(net.state_dict(), net1_path)
			torch.save(net.state_dict(), net2_path)

			print("[ saved ]")