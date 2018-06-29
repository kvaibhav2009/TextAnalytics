import pandas as pd
import numpy
from scipy import linalg
from scipy.sparse.linalg import svds
from scipy import sparse
from numpy import dot
from numpy.linalg import norm
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.spatial.distance import cosine

training = pd.DataFrame.as_matrix(pd.read_csv("/Dekstop items/Desktop JUNE '17/QnA TFIDF/TFIDFMatrix.csv"))
training = training[:, 1:]
z = training
z = numpy.transpose(z, axes=None)

for i in range(len(z)):
    for j in range(len(z[0])):
        if z[i, j] > 0:
            z[i, j] = 1

u, sigma, vt = svds(z)

n = len(sigma)
# reverse the n first columns of u
u[:, :n] = u[:, n - 1::-1]
# reverse s
sigma = sigma[::-1]
# reverse the n first rows of vt
vt[:n, :] = vt[n - 1::-1, :]

squareSum = 0.0
for i in range(len(sigma)):
    squareSum += (sigma[i]) ** 2

subsetSum = 0.0
count = 0;
for i in range(len(sigma)):
    subsetSum += (sigma[i]) ** 2
    count += 1
    if subsetSum >= squareSum * 0.85:
        break;
k = count
sigma = linalg.diagsvd(sigma, len(sigma), len(sigma))

##Dimensionality reduction-Removing k singular values
E = sigma[:k, :k]
U = u[:, :k]
Vt = vt[:k, :]

##
#####
u1, sigma1, vt1 = svds(z, 4, which='LM')

#####
Y = numpy.transpose(u, axes=None)

result = ([[0.0 for row in range(len(u[0]))] for col in range(len(vt))])
result = numpy.matrix(result).reshape(len(u[0]), len(vt))

z[:, 0]

D = z[:, 0]

C = numpy.matmul(D, U)

C.shape

U.shape

E.shape

Z = numpy.matmul(C, numpy.linalg.inv(E))

# cos_sim = dot(a, b)/(norm(a)*norm(b))

Z_result = pd.DataFrame()

## transposing nd array Z
#
#
Z = (numpy.vstack(numpy.array(Z)))

Z = pd.DataFrame(Z.transpose())

Z_result = pd.DataFrame()
for i in range(len(z[0])):
    D = z[:, i]
    C = numpy.matmul(D, U)
    Z = numpy.matmul(C, numpy.linalg.inv(E))
    Z = (numpy.vstack(numpy.array(Z)))
    Z = pd.DataFrame(Z.transpose())
    Z_result = Z_result.append(Z)

Z_result.to_excel("Document_LSIMatrix.xlsx")




### query building
queryText = "what is deductible"
words = word_tokenize(queryText)
ps = PorterStemmer()

queryText=""
for w in words:
    print(ps.stem(w))
    queryText=ps.stem(w)+" "

vocab=(pd.read_csv("/Dekstop items/Desktop JUNE '17/QnA TFIDF/TFIDFMatrix.csv"))

list(vocab)

query="what is deductible"
query = ([[0.0 for row in range(1)] for col in range(len(training[0]))])

query = numpy.matrix(query).reshape(1, len(training[0]))

if "what" in list(vocab):
    print "true"


for w in words:
    print(ps.stem(w))
    if ps.stem(w) in list(vocab):
        i=list(vocab).index(ps.stem(w))
        query[:,i]=1


C = numpy.matmul(query, U)
Z = numpy.matmul(C, numpy.linalg.inv(E))
Z = (numpy.vstack(numpy.array(Z)))
Z = pd.DataFrame(Z.transpose())


document = pd.DataFrame.as_matrix(pd.read_excel("/Users/vaibhavkotwal/Desktop/Document_LSIMatrix.xlsx"))
document=document[0,:]
document = (numpy.vstack(numpy.array(document)))
document = pd.DataFrame(document)

print(cosine(document[0], Z[0]))

result_list=list()
documents=pd.DataFrame.as_matrix(pd.read_excel("/Users/vaibhavkotwal/Desktop/Document_LSIMatrix.xlsx"))
for i in range(len(documents)):
    document=documents[i,:]
    document = (numpy.vstack(numpy.array(document)))
    document = pd.DataFrame(document)
    CosSim=cosine(document[0], Z[0])
    result_list.append((CosSim))

Z_result=pd.DataFrame(result_list)
Z_result.to_excel("Similarity_Results.xlsx")