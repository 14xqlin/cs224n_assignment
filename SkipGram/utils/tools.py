

def _getNegSamples(outsideWordIdx, dataset, K):

    negSampleWordIndices = [None] * K

    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()

        negSampleWordIndices[k] = newidx

    return negSampleWordIndices
