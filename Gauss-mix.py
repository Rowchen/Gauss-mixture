import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
class Gauss_mix:
    def __init__(self,k,iter):
        self.k=k
        self.iter=iter
        self.labels=[]
    def fit(self,x):
        c=x[np.random.randint(0,x.shape[0],(self.k,x.shape[0]/self.k))]
        u=np.zeros((self.k,x.shape[1]))
        sigma=np.zeros((self.k,x.shape[1],x.shape[1]))
        for i in xrange(c.shape[0]):
            u[i]=np.mean(c[i])
            sigma[i]=(c[i]-u[i]).T.dot((c[i]-u[i]))/c[i].shape[0]
        prior=np.full(self.k,1.0/self.k)
        posterior=np.zeros((x.shape[0],self.k))
        gausspro=np.zeros((x.shape[0],self.k))
        t=0

        cons1=((2*3.14159)**(x.shape[1]/2))
        while(t<self.iter):
            t+=1
            for j in xrange(self.k):
                sigma_inv=inv(sigma[j])
                sigma_det=np.linalg.det(sigma[j])
                cons=sigma_det*cons1
                for i in xrange(x.shape[0]):
                    gausspro[i,j]=np.exp(-((x[i]-u[j]).dot(sigma_inv)).dot((x[i]-u[j]).T)/0.5)
                    gausspro[i,j]/=cons

            for i in xrange(x.shape[0]):
                ls=np.sum(prior*gausspro[i])
                for j in xrange(self.k):
                    posterior[i,j]=prior[j]*gausspro[i,j]/ls
            for i in xrange(self.k):
                u[i]=np.sum(posterior[:,i:i+1]*x,axis=0)/np.sum(posterior[:,i:i+1])
                prior[i]=np.sum(posterior[:,i:i+1])/x.shape[0]
                sigma[i] = (posterior[:,i:i+1]*(x - u[i])).T.dot((x - u[i]))/np.sum(posterior[:,i:i+1])

        self.labels=np.argmax(posterior,axis=1)
        print self.labels




from sklearn.datasets.samples_generator import make_blobs
centers = [[1, 1], [-1, -1], [1, -1]]
data, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
s=Gauss_mix(3,10)
s.fit(data)
color=['ro','bo','go','yo','co','ko','mo','wo']
for i in xrange(data.shape[0]):
    plt.plot(data[i, 0], data[i, 1], color[s.labels[i]])
plt.show()
