import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

images = glob.glob('test_images/*')
for f in images:
    print('processing ', f)
    suffix = f[len('test_images/'):]
    fnames = [f] + ['output_images/{0}_{1}'.format(prefix,suffix) \
            for prefix in ['ud','bn','wp','wf','final']]
    plt.figure(figsize=(20,10))
    for i,fname in enumerate(fnames):
        plt.subplot(2,3,i+1)
        img = mpimg.imread(fname)
        plt.imshow(img)
        plt.title(fname[fname.index('/')+1:])
    plt.savefig('output_images/summary_{0}'.format(suffix))

