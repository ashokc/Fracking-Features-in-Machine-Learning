import sys
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import random as rn
import numpy as np
import matplotlib.pyplot as plt
rn.seed(1)
np.random.seed(1)

#    prp ./fracking.py 250 4

args = sys.argv
nPoints = int(args[1])
nClasses = int(args[2])
target_names = []
for i in range(0, nClasses):
    target_names.append(str(i))

lmin, lmax, delta, threshold_prob = -1.0, 1.0, 0.05, 0.75

def generateData (zones):
    for i in zones:
        data, labels = getPointsInTriangle(nPoints, triangles[i], int(i/2))
        diagonalPoints = np.where(np.logical_or((np.abs(data[:,0] - data[:,1]) < delta),(np.abs(data[:,0] + data[:,1]) < delta)))[0]
        strippedData = np.delete(data,diagonalPoints,0)
        strippedLabels = np.delete(labels,diagonalPoints)
        if (i == 0):
            allData = np.copy(strippedData)
            allLabels = np.copy(strippedLabels)
        else:
            allData = np.vstack ( (allData, strippedData) )
            allLabels = np.concatenate ( (allLabels, strippedLabels), axis=None )
    return allData, allLabels

def getPointsInTriangle(n, vertices, label=None):
    p0x, p0y, p1x, p1y, p2x, p2y = vertices
    s = np.random.random_sample(10*n)
    t = np.random.random_sample(10*n)
    s_plus_t = s + t
    inside_points = np.where(s_plus_t < 1.0)[0]
    s0 = s[inside_points][0:n]
    t0 = t[inside_points][0:n]
    data = np.zeros((n,2))
    data[:,0] = p0x + (p1x - p0x) * s0 + (p2x - p0x) * t0
    data[:,1] = p0y + (p1y - p0y) * s0 + (p2y - p0y) * t0
    if label != None:
        labels = np.ones(n) * float(label)
        return data, labels
    else:
        return data

def getF1Score (dataIn, labelsIn):
    predictedLabels = model.predict(dataIn)
    c_report = classification_report(labelsIn, predictedLabels, digits=4, target_names=target_names, output_dict=True)
    return c_report['weighted avg']['f1-score']

def splitData(data, labels, test_size):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1).split(data, labels)
    train_indices, test_indices = next(sss)
    train_data, test_data  = data[train_indices], data[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]
    return train_data, test_data, train_labels, test_labels

def plotData(data, size, marker):
    fig = plt.figure(figsize=(8,8),dpi=720)
    plt.xlim(lmin-0.005, lmax+0.005)
    plt.ylim(lmin-0.005, lmax+0.005)

    yvals = np.zeros(100)
    xvals = np.linspace(lmin-0.005, lmax+0.005, 100)
    plt.plot(xvals, yvals)
    plt.plot(yvals, xvals)

    plt.plot(data[:,0], data[:,1], color='k', linestyle='', markersize=size,marker='.')

    data0 = data[(np.where((data[:,0] > 0.0) & (data[:,1] > 0.0)))]
    plt.plot(data0[:,0], data0[:,1], color=colors[0], linestyle='', markersize=size,marker='.')
    data0 = data[(np.where((data[:,0] < 0.0) & (data[:,1] > 0.0)))]
    plt.plot(data0[:,0], data0[:,1], color=colors[1], linestyle='', markersize=size,marker='.')
    data0 = data[(np.where((data[:,0] < 0.0) & (data[:,1] < 0.0)))]
    plt.plot(data0[:,0], data0[:,1], color=colors[2], linestyle='', markersize=size,marker='.')
    data0 = data[(np.where((data[:,0] > 0.0) & (data[:,1] < 0.0)))]
    plt.plot(data0[:,0], data0[:,1], color=colors[3], linestyle='', markersize=size,marker='.')

    fig.tight_layout()
    fig.savefig('data-plot-' + marker + '.png', format='png', dpi=720)
    plt.close()

def getLowClassificationConfidence (probabilities, data):
    driftPoints = []
    for i in range (0, probabilities.shape[0]):
        if (np.amax(probabilities[i,:]) < threshold_prob):  # the ith class points with probability < threshold_prob
            driftPoints.append(i)
    return data[driftPoints]

triangles = [ [0.0+delta, 0.0+delta, 1.0-delta, 0.0+delta, 1.0-delta, 1.0-delta], [0.0+delta, 0.0+delta, 1.0-delta, 1.0-delta, 0.0+delta, 1.0-delta], [0.0-delta, 0.0+delta, 0.0-delta, 1.0-delta, -1.0+delta, 1.0-delta], [0.0-delta, 0.0+delta, -1.0+delta, 1.0-delta, -1.0+delta, 0.0+delta], [0.0-delta, 0.0-delta, -1.0+delta, 0.0-delta, -1.0+delta, -1.0+delta], [0.0-delta, 0.0-delta, -1.0+delta, -1.0+delta, 0.0-delta, -1.0+delta], [0.0+delta, 0.0-delta, 0.0+delta, -1.0+delta, 1.0-delta, -1.0+delta], [0.0+delta, 0.0-delta, 1.0-delta, -1.0+delta, 1.0-delta, 0.0-delta] ]

initialZones = [0, 2, 4, 6]
data, labels = generateData (initialZones)
colors = ['r', 'g', 'b', 'c', 'r', 'g', 'b', 'c']
plotData(data, 1, '0')    # all initial data

train_data, test_data, train_labels, test_labels = splitData(data, labels, 0.2)
model = LogisticRegression(max_iter=10000, multi_class='multinomial', verbose=0, tol=1.0e-8, solver='lbfgs' )
model.fit(train_data, train_labels)
probabilities = model.predict_proba(test_data)
print ('starting f1-score:', getF1Score (test_data, test_labels))
lcc_data = getLowClassificationConfidence (probabilities, test_data)
n_lcc = [lcc_data.shape[0]]

nNew = int(nPoints/100)
zones = initialZones.copy()
for j in [0, 1, 3, 5, 7]:
    if j > 0:
        zones.append(j)
    for i in range(1, 5):
        new_data, new_labels = generateData(zones)
        new_probabilities = model.predict_proba(new_data)
        new_lcc_data = getLowClassificationConfidence (new_probabilities, new_data)
        if j == 0 and i == 1:
            all_lcc_data = np.copy(lcc_data)
            all_data = np.copy(data)
        else:
            all_lcc_data = np.vstack ( (all_lcc_data, new_lcc_data) )
            all_data = np.vstack ( (all_data, new_data) )

        n_lcc.append(all_lcc_data.shape[0])
        print ('i, j, n_lcc:', i, j, all_lcc_data.shape[0])
    plotData(all_data, 1, str(j))    # all initial data
    plotData(all_lcc_data, 2, 'drift-' + str(j))

fig = plt.figure(figsize=(8,8),dpi=720)
plt.plot(n_lcc, color='r')
fig.tight_layout()
fig.savefig('n_lcc.png', format='png', dpi=720)
plt.close()

