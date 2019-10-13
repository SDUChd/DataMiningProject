from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
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

digits=load_digits()
print(digits.data.shape)
# print(digits.data)
# print(len(digits.target))
data = scale(digits.data)

labels_true=digits.target


#---------Draw the original images----------

# fig=plt.figure(figsize=(6,6))
# fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
# #绘制数字：每张图像8*8像素点
# for i in range(64):
#     ax=fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
#     ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
#     #用目标值标记图像
#     ax.text(0,7,str(digits.target[i]))
# plt.show()
#
# plt.clf()

#run PCA to reduce dimention
n_digits = len(np.unique(digits.target))
reduced_data = PCA(n_components=2).fit_transform(data)
reduced_data = data
# print(reduced_data)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# plt.show()

plt.clf()


#---------k-means--------
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
labels_predict=kmeans.fit_predict(reduced_data)
print(labels_predict)
centroids = kmeans.cluster_centers_


plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)


plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=100, linewidths=3,
            color='black', zorder=10)
# plt.show()
plt.clf()

#Make Evaluation
KMeansNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
KMeansHomo=metrics.homogeneity_score(labels_true, labels_predict)
KmeansComplete=metrics.completeness_score(labels_true, labels_predict)


print("NMI of k-means: "+str(KMeansNMI))
print("Homogeneity of k-means: "+str(KMeansHomo))
print("Completeness of k-means: "+str(KmeansComplete))


#--------Affinity propagation--------

af = AffinityPropagation()
labels_predict=af.fit_predict(reduced_data)
centroids = af.cluster_centers_indices_
# print(centroids)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)

# plt.show()
plt.clf()

#Make Evaluation
AFNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
AFHomo=metrics.homogeneity_score(labels_true, labels_predict)
AFComplete=metrics.completeness_score(labels_true, labels_predict)

print(AFNMI)
print(AFHomo)
print(AFComplete)


#--------Mean-Shift--------

bandwidth = estimate_bandwidth(reduced_data, quantile=0.2, n_samples=5000)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(reduced_data)
labels_predict=ms.labels_
# print(labels_predict)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)


# plt.show()
plt.clf()

#Make Evaluation
MSNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
MSHomo=metrics.homogeneity_score(labels_true, labels_predict)
MSComplete=metrics.completeness_score(labels_true, labels_predict)

print(MSNMI)
print(MSHomo)
print(MSComplete)

#--------Spectral Clustering--------

#
# s = SpectralClustering()
# labels_predict=s.fit_predict(reduced_data)
#
# # print(centroids)
#
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)
#
# # plt.show()
# plt.clf()
#
# #Make Evaluation
# SNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
# SHomo=metrics.homogeneity_score(labels_true, labels_predict)
# SComplete=metrics.completeness_score(labels_true, labels_predict)
#
# print(SNMI)
# print(SHomo)
# print(SComplete)


#--------Ward Hierarchical Clustering--------


ward = AgglomerativeClustering(n_clusters=n_digits,
        linkage='ward')
labels_predict=ward.fit_predict(reduced_data)

# print(centroids)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)

# plt.show()
plt.clf()

#Make Evaluation
print("----------xxxxx")
WardNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
WardHomo=metrics.homogeneity_score(labels_true, labels_predict)
WardComplete=metrics.completeness_score(labels_true, labels_predict)

print(WardNMI)
print(WardHomo)
print(WardComplete)


#--------Agglomerative Clustering--------


ac=AgglomerativeClustering(n_clusters=n_digits,affinity='euclidean',linkage='complete')

labels_predict=ac.fit_predict(reduced_data)

# print(centroids)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)

# plt.show()
plt.clf()

#Make Evaluation
acNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
acHomo=metrics.homogeneity_score(labels_true, labels_predict)
acComplete=metrics.completeness_score(labels_true, labels_predict)

print(acNMI)
print(acHomo)
print(acComplete)


#--------DBSCAN--------


ds=DBSCAN(eps=0.6,min_samples=30)

labels_predict=ds.fit_predict(reduced_data)

# print(centroids)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)

# plt.show()
plt.clf()

#Make Evaluation
dsNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
dsHomo=metrics.homogeneity_score(labels_true, labels_predict)
dsComplete=metrics.completeness_score(labels_true, labels_predict)

print(dsNMI)
print(dsHomo)
print(dsComplete)




#--------Gaussian Mixtures--------


gmm = GaussianMixture(n_components=n_digits)

labels_predict=gmm.fit_predict(reduced_data)

# print(centroids)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_predict,s=2)

plt.show()
plt.clf()

#Make Evaluation
gmmNMI=metrics.normalized_mutual_info_score(labels_true, labels_predict)
gmmHomo=metrics.homogeneity_score(labels_true, labels_predict)
gmmComplete=metrics.completeness_score(labels_true, labels_predict)

print(gmmNMI)
print(gmmHomo)
print(gmmComplete)




print("done!")