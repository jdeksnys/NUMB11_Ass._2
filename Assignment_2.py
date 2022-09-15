from array import array
from cgitb import grey, reset
from email.mime import image
from importlib.util import set_loader
from os import system
from tokenize import PlainToken
from unittest import result
import numpy as np
from numpy import linalg as ln
import matplotlib
from matplotlib import pyplot
from matplotlib.image import imread
system("clear")



class Orthogonalization:
    def Gramschmidt(A): # only for linearly independent set of vetors! else zero matrix returned (No non-zero solutions for Ax=0)
        m = np.shape(A)[0] # rows
        n = np.shape(A)[1] # cols
    
        if m >= n:

            for k in range(n):

                v_k = A[:, k]
                
                for j in range(0,k):
                    u_j = A[:, j]
                    v_k = v_k - np.dot(u_j, v_k) * u_j
                
                if np.array_equal(v_k, np.zeros(v_k.shape)):
                    raise np.linalg.LinAlgError("The column vectors are not linearly independent")
                
                v_k = v_k / np.sqrt(np.dot(v_k, v_k))

                A[:, k] = v_k
        
        else:
            print("Error: m < n")
            return

        return np.round(A,3)



    def QR_householder(A):
        m=A.shape[0] # rows
        B=A # in order to get n form A below
        QA = A
        while m > 1:
            n = np.shape(A)[0] - (m)
            a=B[:,0]
            ahat=np.array([np.linalg.norm(a)]+(m-1)*[0.]) # ahat[0] = norm; ahat[1:] = zeros
            
            if a[0] >= 0:
                v=a+ahat
            else:
                v=a-ahat
            
            vNorm = np.linalg.norm(v)
            v = v / vNorm

            H=np.eye(m)-2*np.outer(v,v) # np.eye - identity matrix; np.outer - outer product of two vectors
            B = np.around(np.matmul(H,B),2)

            QA[n:,n:] = B
            B = B[1:,1:]
            m -= 1

        # i = np.shape(A)[0]
        # z = np.array(np.shape(A)[1] * [0.])
        # while i > 1:
        #     a = QA[i-1,:] 
        #     if np.allclose(a,z,0.01):
        #         QA = QA[0:i-1,:]
            # i -= 1

        Q = np.transpose(np.matmul(QA, np.linalg.inv(A)))
        Q = np.round(Q,2)

        result = [Q,A,QA] # Q: product of householder transf.(triangular matrix); A: og matrix; QA: result
        return result

    def Compress(self,imgDir, tol):
        res = self.MakeGrayscale(imgDir)
        imgGreyArr = res[1]
        svd = np.linalg.svd(imgGreyArr)
        U = svd[0]
        S = svd[1]
        Vt = svd[2]
        diff = imgGreyArr - np.matmul(np.matmul(U,S),Vt)
        twoNorm = np.linalg.norm(diff,2)
        rows = np.shape(U)[0]
        rank = 0
        # r=rank is such that overlaying svd product matrices r times results in image of desired quality (larger r = larger resolution)

        layer = np.zeros((rows,rows))
        for i in range(0,rows):
            if S[i] >= tol:
                a1 = U[:,i],
                a2 = S[i]
                a = np.outer(a1,a2)
                b = np.outer(a,Vt[i,:])
                layer = layer + b
                rank += 1
            else:
                pyplot.imshow(layer, cmap=pyplot.get_cmap("gray"))
                pyplot.show()
                return [twoNorm,layer,rank]

    def MakeGrayscale(self,imgDir,imgArr=None):
        imgArr = self.imgToArray(imgDir)
        # pyplot.imshow(imgArr)
        # pyplot.show()
        rgb = [0.2989, 0.5870, 0.1140]
        imgGreyArr = np.dot(imgArr[...,:3], rgb)
        # pyplot.imshow(imgGreyArr, cmap=pyplot.get_cmap("gray"))
        # # pyplot.show()
        return [imgArr,imgGreyArr]

    def imgToArray(self,imgDir):
        return matplotlib.image.imread(imgDir)



A = np.array([[1,2,4],[0,0,5], [0,3,6]])
B = np.array([[1,2,4,5],[0,0,5,6], [0,3,6,9],[1,2,3,4],[9,1,3,3]])
C = np.array([[1.,-7.,0.],[2.,-20.,5.],[2.,1.,1.]])
D = np.array([[5,6,4,3,6],[6,7,8,9,1],[1,2,1,1,2],[0,0,2,3,9],[0,1,2,3,0],[9,5,9,8,6]])
E = np.array([[1,-1,1],[1,0,1],[1,1,2]])
F = np.array([[1,1,1],[2,1,0],[5,1,3]])



# TESTS:
def ex_1(A):
    print(f"matrix A:\n{A}")
    res = Orthogonalization.Gramschmidt(A)
    print(f"\nGram-Schmidt orthogonalized:\n {res}")

def ex_2(A):
    ex_1(A)
    res = Orthogonalization.Gramschmidt(A)
    I = np.identity(np.shape(res)[0])
    QtQ = np.matmul(np.transpose(res),res)
    print(f"\n2-norm: {np.linalg.norm(res, ord=2)}")
    print(f"\n2-norm of(I - Q^tQ) = {np.linalg.norm(1-QtQ,ord=2)}")
    print(f"\neigenvalues of QtQ: {[e for e in np.linalg.eig(QtQ)[0]]}") # either 1 or -1 for OG matrix
    print(f"\ndet(QtQ) = {np.linalg.det(QtQ)}") # either 1 or -1 for OG matrix
    print(f"\nGram-Schmidt unstable (Sciencedirect.com): 'The computation also yields poor results when some of the vectors are almost linearly dependent. For these reasons, it is said that the classical Gram-Schmidt process is numerically unstable. Subtracting the projections of vi onto the ej all at once causes the problem.' ")

def ex_3(A):
    # res = Orthogonalization.Gramschmidt(A)
    qr = np.linalg.qr(A)[0]
    # print(f"GS:\n{res}\n")
    print(f"QR:\n{qr}\n")
    print("================ repeating test of A ================\n")
    ex_2(A)
    print("\n================ repeating test of qr ================\n")
    ex_2(qr)

def ex_4(A):
    res = Orthogonalization.QR_householder(A)
    print(f"diagonal matrix:\n{res[2]}\n")
    print(f"\nQ is orthogoanl if Q^tQ = I.")
    print(f"\nQ^tQ:\n{np.matmul(np.transpose(res[0]),res[0])}")

def ex_6(imgDirectory, tolerance):
    brain = Orthogonalization()
    compressedImage = brain.Compress(imgDirectory, tolerance)[2]



# # EXECUTE TESTS:
# ex_1(C)
# ex_2(C)
# ex_3(C)
# ex_4(C)
# ex_6('google.png',1) # fileDirectory, tolerance
