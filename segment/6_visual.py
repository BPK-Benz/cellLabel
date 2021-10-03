import os
import cv2
import numpy as np
import torch
from utils.unet import UNet

if __name__ == "__main__":


    # load model
    net_path = 'source/latest.pth'
    net = UNet(n_channels=3, n_classes=3)
    net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
    net.eval();

    # load data
    split = 'source/test.txt'
    with open(split) as f:
    	samples = f.readlines()
    total = len(samples)

    # training resolution
    res_x = 1360 // 2
    res_y = 1024 // 2
    threshold = 0.1

    for i in range(total):

        # show progress
        print('[ {} in {} | {} ]'.format(i, total, samples[i]))

        # get data
        name = samples[i].replace('\n', '') + '.png'
        image_path = os.path.join('data/output_export', name)

        image = cv2.imread(image_path)
        h, w, c = image.shape

        # pre-processing
        x = image.copy()
        # x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        # x = cv2.equalizeHist(x)
        # kernel = np.ones((2, 2), np.float32) / 4
        # x = cv2.filter2D(x, -1, kernel)
        # ret, x = cv2.threshold(x, 150, 255, cv2.THRESH_BINARY)
        x = cv2.resize(x, (res_x, res_y))
        # xx = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        x = x.astype(np.float32)
        x /= 255
        x = np.expand_dims(x, axis=0)

        # mask detection
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.tensor(x).float()
        out = net(x)

        # post-processing
        out = out.detach().numpy()
        out = out[0]

        # to opencv format
        channel1 = out[0].copy()
        channel1[ channel1 < threshold  ] = 0
        channel1[ channel1 >= threshold ] = 255
        channel1 = np.asarray(channel1, dtype=np.uint8)

        channel2 = out[1].copy()
        channel2[ channel2 < threshold  ] = 0
        channel2[ channel2 >= threshold ] = 255
        channel2 = np.asarray(channel2, dtype=np.uint8)

        channel3 = out[2].copy()
        channel3[ channel3 < threshold  ] = 0
        channel3[ channel3 >= threshold ] = 255
        channel3 = np.asarray(channel3, dtype=np.uint8)

        mask_image = cv2.merge([channel1, channel2, channel3])
        mask_image = cv2.resize(mask_image, (w, h))

        mask_path = image_path.replace('output_export', 'output_predicted_segment')
        mask_folder = os.path.abspath(os.path.join(mask_path, os.pardir))
        if not os.path.exists(mask_folder): os.makedirs(mask_folder)
        cv2.imwrite(mask_path, mask_image)


        # cv2.imshow('mask', mask)
        # key = cv2.waitKey(0)
        # if key == ord('q'): exit()
