import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib

if __name__ == '__main__':

    root = './result'

    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
                            minibatch_size=8)

    tflib.init_tf()
    os.makedirs(root, exist_ok=True)

    ckpt = './ckpt/network-snapshot-008940.pkl'
    with open(ckpt, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    # n = 50
    # seeds = np.random.randint(10000,size=n)

    seeds = np.arange(0,100)
    for seed in seeds:
        latent = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1])])
        dlatents = Gs.components.mapping.run(latent, None)  # [seed, layer, component]
        img = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)
        img = PIL.Image.fromarray(img[0], 'RGB')
        img.save(os.path.join(root, 'seed=%d.png' % seed))
