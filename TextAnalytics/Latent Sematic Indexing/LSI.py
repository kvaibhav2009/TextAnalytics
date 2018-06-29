import pandas as pd
import numpy
from scipy import linalg

training=pd.DataFrame.as_matrix(pd.read_csv("/Dekstop items/Desktop JUNE '17/QnA TFIDF/TFIDFMatrix.csv"))
training=training[:,1:]

u,sigma,vt = numpy.linalg. svd(training)

Y=numpy.transpose(u,axes=None)

result = ([[0 for row in range(len(u))] for col in range(len(Y[0]))])
result=numpy.array(result).reshape(len(u),len(Y[0]))
result=numpy.matrix(result)


for i in range(len(u)):
   # iterate through columns of Y
   for j in range(len(Y[0])):
       # iterate through rows of Y
       for k in range(len(Y)):
           result[i][j] += u[i][k] * Y[k][j]

S = linalg.diagsvd(sigma, len(sigma), len(sigma))


#
# import pandas as pd
# import numpy
# from scipy import linalg, dot
# import sklearn
# from __future__ import print_function
# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import Normalizer
# from sklearn import metrics
# from sklearn.cluster import KMeans, MiniBatchKMeans
#
# training = pd.DataFrame.as_matrix(
#     pd.read_csv("C:/Users/vaibhav.vijay.kotwal/Desktop/Cathy TFIDF Azure model/TFIDFMatrix.csv"))
#
# u, sigma, vt = numpy.linalg.svd(training)
#
# Y = numpy.transpose(u, axes=None)
#
# result = ([[0.0 for row in range(len(u))] for col in range(len(Y[0]))])
#
# for i in range(len(u)):
#     # iterate through columns of Y
#     for j in range(len(Y[0])):
#         # iterate through rows of Y
#         for k in range(len(Y)):
#             result[i][j] += u[i][k] * Y[k][j]
#
# result = numpy.array(result).reshape(len(u), len(Y[0]))
# result = numpy.matrix(result)
#
# lsa = TruncatedSVD(2, algorithm='arpack')
#
# df = [[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]]
#
# # for i in range(5):
# #    result[i,i]=float(q[i])
#
# for i in range(5):
#     result[i, i] = float(q[i])
#
#
# def matrixmult(A, B):
#     rows_A = len(A)
#     cols_A = len(A[0])
#     rows_B = len(B)
#     cols_B = len(B[0])
#
#     if cols_A != rows_B:
#         # print "Cannot multiply the two matrices. Incorrect dimensions."
#         return
#
#     # Create the result matrix
#     # Dimensions would be rows_A x cols_B
#     C = [[0 for row in range(cols_B)] for col in range(rows_A)]
#     # print C
#
#     for i in range(rows_A):
#         for j in range(cols_B):
#             for k in range(cols_A):
#                 C[i][j] += A[i][k] * B[k][j]
#
#     return C
#
#
# # matrixmult(sigma,vt)
#
#
# R = numpy.asmatrix(r)
#
# R[0:((R.shape[0]) - 1), :]
#
# for i in range(6):
#     for j in range(5):
#         for k in range(5):
#             result[i, j] += E[i, k] * R[k, j]
#
# z = training
# for i in range(len(training)):
#     for j in range(len(training[0])):
#         if z[i, j] > 0:
#             z[i, j] = 1
#
# Z = numpy.matmul(E, R)
