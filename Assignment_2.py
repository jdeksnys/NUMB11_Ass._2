from cgitb import reset
from os import system
import numpy as np
from numpy import linalg as ln
system("clear")



class Orthogonalization:
    def Gramschmidt(A):
        n = np.shape(A)[0]
        m = np.shape(A)[1]
    
        if m >= n:

            for i in range(m):

                q = A[:, i]
                
                for j in range(i):
                    q = q - np.dot(A[:, j], A[:, i]) * A[:, j]
                
                if np.array_equal(q, np.zeros(q.shape)):
                    raise np.linalg.LinAlgError("The column vectors are not linearly independent")
                
                q = q / np.sqrt(np.dot(q, q))

                A[:, i] = q

        return A

    def QR_householder(A):
        # m = np.shape(A)[1]
        m=A.shape[0]
        B=A # in order to get n form A below
        # A = np.zeros(np.shape(A))
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
            # print(2*np.outer(v,v))

            H=np.eye(m)-2*np.outer(v,v) # np.eye - identity matrix; np.outer - outer product of two vectors
            B = np.around(np.matmul(H,B),2)

            # print(A)
            # print(B)
            A[n:,n:] = B
            B = B[1:,1:]
            # print(A)
            # print()
            m = m-1
        print(A)
    # TODO: check if last iteration needed (ex: 3 - 3 or 2???)
    # TODO: Rx = Hb; R = A; x - unknown; H = H_1H_1...; b - given
        


A = np.array([[1,2,4],[0,0,5], [0,3,6]])
B = np.array([[1,2,4,5],[0,0,5,6], [0,3,6,9],[1,2,3,4],[9,1,3,3]])
C = np.array([[1.,-7.,0.],[2.,-20.,5.],[2.,1.,1.]])



# TESTS:
def ex_1(A):
    print(f"matrix A:\n{A}")
    res = Orthogonalization.gramschmidt(A)
    print(f"\nGram-Schmidt orthogonalized:\n {res}")

def ex_2(A):
    ex_1(A)
    res = Orthogonalization.gramschmidt(A)
    I = np.identity(np.shape(res)[0])
    QtQ = np.matmul(np.transpose(res),res)
    print(f"\n2-norm: {np.linalg.norm(res, ord=2)}")
    print(f"\n2-norm of(I - Q^tQ) = {np.linalg.norm(1-QtQ,ord=2)}")
    print(f"\neigenvalues of QtQ: {[e for e in np.linalg.eig(QtQ)[0]]}") # either 1 or -1 for OG matrix
    print(f"\ndet(QtQ) = {np.linalg.det(QtQ)}") # either 1 or -1 for OG matrix

def ex_3(A):
    # res = Orthogonalization.gramschmidt(A)
    qr = np.linalg.qr(A)[0]
    # print(f"GS:\n{res}\n")
    print(f"QR:\n{qr}\n")
    print("================ repeating test of A ================\n")
    ex_2(A)
    print("\n================ repeating test of qr ================\n")
    ex_2(qr)

def ex_4(A):
    Orthogonalization.QR_householder(A)



# # EXECUTE TESTS:
# ex_1(A)
# ex_2(A)
ex_3(C)
# ex_4(_C)


# print(A)
print()
# print(Orthogonalization.newGramschmidt(A))