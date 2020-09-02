import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(1)
plt.subplot(211)
plt.imshow(mpimg.imread('figures/0/rank_0_cluster_0_Philadelphia272.png', 'png'))
plt.subplot(212)
plt.imshow(mpimg.imread('figures/0/rank_0_cluster_0_Philadelphia272.png', 'png'))
plt.axis('off')
plt.show()