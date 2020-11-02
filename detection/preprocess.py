from sklearn.cluster import KMeans
import pickle
import skimage
from skimage import exposure
import numpy as np

def get_domimant_colors(img, top_colors=2):
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    ## # grab the number of different clusters and create a histogram
    ## # based on the number of pixels assigned to each cluster
    ## numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    ## (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    ## # normalize the histogram, such that it sums to one
    ## hist = hist.astype("float")
    ## hist /= hist.sum()
    return clt.cluster_centers_, None #hist

class ImageClassIdentifier(object):
    def __init__(self, kmeans_pkl='../kmeans.pkl'):
        with open(kmeans_pkl, 'rb') as f:
            self.kmeans = pickle.load(f)

    def detect(self, img):
        imghsv = skimage.color.rgb2hsv(img)
        color, _ = get_domimant_colors(imghsv, top_colors=1)
        predicted_class = self.kmeans.predict(color)[0]
        return predicted_class

class ImagePreProcessor1(object):
    def __init__(self, kmeans_pkl='../kmeans.pkl'):
        self.identifier = ImageClassIdentifier(kmeans_pkl)

    def __call__(self, image, img_id):
        predicted_class = self.identifier.detect(np.array(image))
        if predicted_class == 0:
            img_eq = exposure.equalize_adapthist(np.array(image))
            return img_eq
        return img
