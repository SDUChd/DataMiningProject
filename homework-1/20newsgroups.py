from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# 提取tfidf特征
vectorizer = TfidfVectorizer()
reduced_data = vectorizer.fit_transform(newsgroups_train.data)
n_digits = len(np.unique(newsgroups_train.target))
labels_true=newsgroups_train.target
# print(reduced_data)
print(reduced_data.shape)
# print(vectors.nnz / float(vectors.shape[0]))

# reduced_data = PCA(n_components=2).fit_transform(reduced_data)



n_pick_topics = 4            # 设定主题数为4
lsa = TruncatedSVD(n_pick_topics)
X2 = lsa.fit_transform(reduced_data)


#---------k-means--------
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
labels_predict=kmeans.fit_predict(reduced_data)
print(labels_predict)
centroids = kmeans.cluster_centers_


# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
#
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=100, linewidths=3,
#             color='black', zorder=10)
# # plt.show()
# plt.clf()

#Make Evaluation
KMeansNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
KMeansHomo=metrics.homogeneity_score(labels_true, labels_predict)
KmeansComplete=metrics.completeness_score(labels_true, labels_predict)

print(KMeansNMI)
print(KMeansHomo)
print(KmeansComplete)


#--------Affinity propagation--------

af = AffinityPropagation()
labels_predict=af.fit_predict(reduced_data)
centroids = af.cluster_centers_indices_
# print(centroids)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
# # plt.show()
# plt.clf()

#Make Evaluation
AFNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
AFHomo=metrics.homogeneity_score(labels_true, labels_predict)
AFComplete=metrics.completeness_score(labels_true, labels_predict)

print(AFNMI)
print(AFHomo)
print(AFComplete)


#--------Mean-Shift--------

bandwidth = estimate_bandwidth(X2, quantile=0.2, n_samples=5000)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X2)
labels_predict=ms.labels_
# print(labels_predict)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
#
# # plt.show()
# plt.clf()

#Make Evaluation
MSNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
MSHomo=metrics.homogeneity_score(labels_true, labels_predict)
MSComplete=metrics.completeness_score(labels_true, labels_predict)

print("mean shift:")
print(MSNMI)
print(MSHomo)
print(MSComplete)

#--------Spectral Clustering--------


s = SpectralClustering()
labels_predict=s.fit_predict(reduced_data)

# print(centroids)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
# # plt.show()
# plt.clf()

#Make Evaluation
SNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
SHomo=metrics.homogeneity_score(labels_true, labels_predict)
SComplete=metrics.completeness_score(labels_true, labels_predict)

print(SNMI)
print(SHomo)
print(SComplete)


#--------Ward Hierarchical Clustering--------


ward = AgglomerativeClustering(n_clusters=n_digits,
        linkage='ward')
labels_predict=ward.fit_predict(reduced_data.toarray())

# print(centroids)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
# # plt.show()
# plt.clf()

#Make Evaluation
WardNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
WardHomo=metrics.homogeneity_score(labels_true, labels_predict)
WardComplete=metrics.completeness_score(labels_true, labels_predict)

print(WardNMI)
print(WardHomo)
print(WardComplete)


#--------Agglomerative Clustering--------


ac=AgglomerativeClustering(n_clusters=n_digits,affinity='euclidean',linkage='complete')

labels_predict=ac.fit_predict(X2)

# print(centroids)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
# # plt.show()
# plt.clf()

#Make Evaluation
acNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
acHomo=metrics.homogeneity_score(labels_true, labels_predict)
acComplete=metrics.completeness_score(labels_true, labels_predict)

print(acNMI)
print(acHomo)
print(acComplete)


#--------DBSCAN--------


ds=DBSCAN(eps=0.05,min_samples=30)

labels_predict=ds.fit_predict(X2)

# print(centroids)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
# # plt.show()
# plt.clf()

#Make Evaluation
dsNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
dsHomo=metrics.homogeneity_score(labels_true, labels_predict)
dsComplete=metrics.completeness_score(labels_true, labels_predict)

print(dsNMI)
print(dsHomo)
print(dsComplete)




#--------Gaussian Mixtures--------


gmm = GaussianMixture(n_components=n_digits)

labels_predict=gmm.fit_predict(X2)

# print(centroids)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
# plt.show()
# plt.clf()

#Make Evaluation
gmmNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
gmmHomo=metrics.homogeneity_score(labels_true, labels_predict)
gmmComplete=metrics.completeness_score(labels_true, labels_predict)

print(gmmNMI)
print(gmmHomo)
print(gmmComplete)




print("done!")

