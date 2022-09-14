from cgitb import reset
from os import system
import numpy as np
from numpy import linalg as ln
system("clear")



class Orthogonalization:
    def gramschmidt(V):
        m = np.shape(V)[0] # rows
        n = np.shape(V)[1] # cols
        if(m>=n):
            U = [];            

            for i in range(0,n):
                if(i == 0):
                    U = []
                    U.append(V[:,i])
                    # print(U)
                    # print()

                elif(i>0):
                    v_k = V[:,i]
                    # print(v_k)

                    # U_k = v_k - sum( dot(v_k,u_i) * u_i / abs(u_i)^2 )

                    sigma = np.zeros([1,m])[0]
                    # print(sigma)
                    for j in range(0,i):
                        sigma = np.add(sigma,(np.dot(v_k, U[j]) / np.dot(U[j], U[j]) ) * U[j]) # sum(...)
                    u = np.add(v_k, -sigma) # v_k - sum(...)
                    U.append(u / np.linalg.norm(u)) # make unit length
                    # print(U)
                    # print(U)

            # return np.around(U,2)
            # TODO:
            # Subtracting the projections of vi onto the ej all at once causes the problem. Split the computation into smaller parts by removing the projections one at a time
            return U

        else:
            print("error: m>n does not hold")
            return

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
        


_A = np.array([[1,2,4],[0,0,5], [0,3,6]])
_B = np.array([[1,2,4,5],[0,0,5,6], [0,3,6,9],[1,2,3,4],[9,1,3,3]])
_C = np.array([[1.,-7.,0.],[2.,-20.,5.],[2.,1.,1.]])



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
# ex_2(_C)
# ex_3(_B)
# ex_4(_C)



# test
A=np.array([[1,2],[4,5],[7,8]])
Q=np.linalg.qr(A)[0]
R = Orthogonalization.QR_householder(A)
print(A)
print(Q)
print(np.matmul(Q,R))
