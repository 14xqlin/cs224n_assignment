import random
import time

import numpy as np

from utils.treebank import StanfordSentiment

random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

dims = 10
C = 5

random.seed(31415)
np.random.seed(9265)

startTime = time.time()