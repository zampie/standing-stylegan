import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import time

if __name__ == '__main__':

    root = './'

    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
                            minibatch_size=1)

    tflib.init_tf()
    os.makedirs(root, exist_ok=True)

    ckpt = './ckpt/network-snapshot-008940.pkl'
    with open(ckpt, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    seeds = [0, 75, 97, 71]
    imgs = []
    sec = 2
    n = 24 * sec

    for i in range(len(seeds)):

        latent_str = np.stack([np.random.RandomState(seeds[i]).randn(Gs.input_shape[1])])

        i_plus = i + 1
        if i_plus >= len(seeds):
            i_plus = 0

        latent_end = np.stack([np.random.RandomState(seeds[i_plus]).randn(Gs.input_shape[1])])

        # for lamb in np.arange(n) / (n):
        for lamb in np.arange(n) / (n - 1):
            latent = (1 - lamb) * latent_str + lamb * latent_end
            dlatents = Gs.components.mapping.run(latent, None)  # [seed, layer, component]
            img = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)
            img = PIL.Image.fromarray(img[0], 'RGB')
            # img.save(os.path.join(root, 'out%.3f.png'%lamb))
            imgs.append(img)

    # imgs[0].save(os.path.join(root, 'inter_%s_seeds_%s.gif' % (
    #     time.ctime().replace(' ', '_').replace(':', '_'), str(seeds)[1:-1].replace(', ', '_')))
    #              , save_all=True, append_images=imgs[1:], optimize=False, duration=41, loop=0)

    imgs[0].save(os.path.join(root, 'inter_%s_seeds_%s_model_%s.gif' % (
        time.ctime().replace(' ', '_').replace(':', '_'), str(seeds)[1:-1].replace(', ', '_'), ckpt.split('/')[-1]))
                 , save_all=True, append_images=imgs[1:], optimize=False, duration=41, loop=0)
# PIL自动去除相邻重复帧
1
# seed = 3
# latent = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1])])
# dlatents = Gs.components.mapping.run(latent, None)  # [seed, layer, component]
# img = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)
# img = PIL.Image.fromarray(img[0], 'RGB')
# img.save(os.path.join(root, 'seed=%d.png' % seed))
