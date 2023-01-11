import os



pic_dir = 'data/img_align_celeba/img_align_celeba/'
image_count = 10000
ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2
WIDTH = 128
HEIGHT = 128
crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)
LATENT_DIM = 32
CHANNELS = 3