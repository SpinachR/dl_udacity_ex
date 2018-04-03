import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


DIMS = (400, 400)
CAT1 = 'cat1.jpg'
im = ndimage.imread('images/cat1.jpg') / 127.5-1
im = im.reshape(1, 400, 400, 3)

input_img = np.concatenate([im, im], axis=0)
print("Input Img Shape: {}".format(input_img.shape))

num_batch, H, W, C = input_img.shape

# theta
M = np.array([[1, 0., 0.], [0., 1., 0.]])
M = np.resize(M, (num_batch, 2, 3))

x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
x_t, y_t = np.meshgrid(x, y)

# remember: theta * [x, y, 1] (we need to augment the dimensions to create homogeneous coordinates.)
ones = np.ones(np.prod(x_t.shape))
sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

# here, we only have one grid. we need 'num_batch' grids
sampling_grid = np.resize(sampling_grid, (num_batch, 3, H*W))


#### transform the sampling grid - batch multiple
batch_grids = np.matmul(M, sampling_grid)  # theta * sampling_grid: [num_batch, 2, H*W]
# reshape to (num_batch, height, width, 2)
batch_grids = batch_grids.reshape(num_batch, 2, H, W)
batch_grids = np.moveaxis(batch_grids, 1, -1)   # [num_batch, H, W, 2]: the last 2 items give us coordinates x and y.
# print(batch_grids[0,:,:,:].shape)
# print(batch_grids[0,:,:,:])

### Given our coordinates x and y in the sampling grid, we want interpolate the pixel value in the original image.
x_s = batch_grids[:, :, :, 0:1].squeeze()
y_s = batch_grids[:, :, :, 1:2].squeeze()

# rescale x and y to [0, W/H]
x = ((x_s + 1.) * W) * 0.5
y = ((y_s + 1.) * H) * 0.5
print('x max value before clip', np.amax(x))

# grab 4 nearest corner points for each (x_i, y_i)
x0 = np.floor(x).astype(np.int64)
x1 = x0 + 1
y0 = np.floor(y).astype(np.int64)
y1 = y0 + 1
#print(x0)
print('x0 max value before clip', np.amax(x0))

# Now we must make sure that no value goes beyond the image boundaries
# make sure it's inside img range [0, H] or [0, W]
x0 = np.clip(x0, 0, W-1)
x1 = np.clip(x1, 0, W-1)
y0 = np.clip(y0, 0, H-1)
y1 = np.clip(y1, 0, H-1)

print(x1.shape)
print('x0 max value after clip', np.amax(x0))
print('x1 max value after clip', np.amax(x1))
print('y0 max value after clip', np.amax(y0))
print('y1 max value after clip', np.amax(y1))
# look up pixel values at corner coords
Ia = input_img[np.arange(num_batch)[:,None,None], y0, x0]
Ib = input_img[np.arange(num_batch)[:,None,None], y1, x0]
Ic = input_img[np.arange(num_batch)[:,None,None], y0, x1]
Id = input_img[np.arange(num_batch)[:,None,None], y1, x1]

print(Ia.shape)
print(Ia[0, 29, 29, :])
print(Ib[0, 29, 29, :])
print(Ic[0, 29, 29, :])
print(Id[0, 29, 29, :])

# calculate deltas
wa = (x1-x) * (y1-y)
wb = (x1-x) * (y-y0)
wc = (x-x0) * (y1-y)
wd = (x-x0) * (y-y0)

print('x0: ', x0[0, 29, 29])
print('y0: ', y0[0, 29, 29])
print('x1: ', x1[0, 29, 29])
print('y1: ', y1[0, 29, 29])
print('x: ', x[0, 29, 29])
print('y: ', y[0, 29, 29])

print('wa', wa[0, 29, 29])
print('wb', wb[0, 29, 29])


# add dimension for addition
wa = np.expand_dims(wa, axis=3)
wb = np.expand_dims(wb, axis=3)
wc = np.expand_dims(wc, axis=3)
wd = np.expand_dims(wd, axis=3)


# compute output
out = wa*Ia + wb*Ib + wc*Ic + wd*Id
print(np.amin(out))
print(out[0,399,399,:])
plt.imshow(out[0])
plt.show()


'''
如果 grid 在 图片外部 的部分，他们都对应着 原始图片 上的4个相同的点，导致最后的插值为0
'''


