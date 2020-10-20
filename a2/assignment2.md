# assignment2

&nbsp;&nbsp;&nbsp; **assignment2** 主要是构造 **skip-gram** 模型来完成词向量训练。该次作业中使用的数据集为：***Stanford Sentiment Treebank (SST)***。

___

#### 数据处理

1. 读取***“SST”***中的***“datasetSentences.txt”***文件

2. 将文件中的句子进行**分词**、**构造词与数字映射表**

3. 构造词向量矩阵***wordVectors***，即作业中的***U***和***V***；在代码中，***wordVectors***的前半部分的初始值是随机生成的数据值，而后半部分是以0作为初始值。
   $$
   wordVectors \in R^{(2*n\_words, d)}
   $$

4. 将参数带入进***word2vec_sgd_wrapper()***函数中，最后返回的结果再调用***sgd()***进行更新。

#### naiveSoftmaxLossAndGradient()

   作业内容中，首先实现的是***naive softmax 损失函数***，如下：
$$
\begin{aligned}
J_{naive-softmax} (v_c, o, U) &= -logP(O=0|C=c) \\
P(O=0|C=c) &= { exp( u_o ^T v_c ) \over \sum _{\omega\in Vocab} exp(u_{\omega} ^T v_c) }
\end{aligned}
$$
其关于参数**$v_c$**的导数为：

$$
\begin{aligned}
{\partial J_{naive-softmax} \over \partial v_c} & = -u _o ^T + \sum _{\omega^{'}} { e^{ u ^T _{\omega ^{'} } v_c} \over {\sum _{\omega} e^{u ^T _{\omega} v_c} } } u ^T _{\omega ^{'}} \\
& = -u ^T _o + \sum _{\omega ^{'}} P(\omega^{'} | c) u ^T _{\omega ^{'}} \\
& = -u ^T _o + U^T \cdot \hat y  \\
& = -U ^T \cdot y + U^T \cdot \hat y \\
& = (\hat y - y) \cdot U^T
\end{aligned}
$$

关于参数**$U$**的导数为：
$$
\begin{aligned}
{\partial J \over \partial u_o} &= (\hat y_o - 1) \cdot v_c \\
{\partial J \over \partial u_w} &= \hat y_w \cdot v_c \\
\end{aligned}
$$



即：
$$
{\partial J \over \partial U} = ( \hat y - \left[
 \begin{matrix}
   0  \\
   \vdots  \\
   1  \\
   \vdots  \\
   0
  \end{matrix}
  \right] ) \cdot v_c \\
  = (\hat y - y) \cdot v_c
$$
具体实现：

```python
    s = np.dot(outsideVectors, centerWordVec)           #with shape (num words in vocab, )
    p = softmax(s)                                      #with shape (num words in vocab, )

    loss = -np.log( p[outsideWordIdx] )

    p[outsideWordIdx] -= 1
    gradCenterVec = np.dot(outsideVectors.T, p)
    gradOutsideVecs = np.dot( p[:, np.newaxis], centerWordVec[:, np.newaxis].T )
```

#### negSamplingLossAndGradient()

  在***navie softmax***中需要更新的参数***$U(U \in R^{|V| \times n})$***由于***$|V|$***的量级很大，因此需要较大的词汇量来训练模型，模型的训练难度会较大、训练时间会较长。因此在***Word2vec***中，可以通过***negative sampling***使得每次训练样本更新，只更新其中小部分的参数。采用***negative sampling***后的损失函数为：
$$
J_{neg-sample}(v_c, o, U) = -log( \sigma (u_o^T v_c ) ) - \sum _{k  = 1} ^{K} {log( \sigma(-u_k v_c) )} \\
\sigma(x) = {1 \over 1 + e^{-x} }
$$
损失函数中的更新***$U$***部分，不再是整个$U$矩阵进行更新，而是从中抽取***K***个与***$u_o$***不同的词汇进行更新。

关于参数的偏导是：
$$
\begin{aligned}

{\partial J \over \partial v_c} &= [-1 + \sigma(u_o^T \cdot v_c)] \cdot u_0 ^T - \sum _{k = 1} ^K {[-1 + \sigma(-u_k^T \cdot v_c)] \cdot u_k ^T} \\

{\partial J \over \partial u_o} &= [-1 + \sigma(u_o^T \cdot v_c)] \cdot v_c \\

{\partial J \over \partial u_k} &= [1 - \sigma(-u_k ^T \cdot v_c)] \cdot v_c

\end{aligned}
$$
具体实现：

```
    dot_one = np.dot(outsideVectors[outsideWordIdx].T, centerWordVec)
    neg_samples = outsideVectors[negSampleWordIndices]
    dot_two = np.dot(-neg_samples, centerWordVec)
    loss = -np.log( sigmoid(dot_one) ) - np.sum( np.log( sigmoid(dot_two) ) )

    gradCenterVec = -(1 - sigmoid(dot_one)) * outsideVectors[outsideWordIdx] + np.sum(
        (np.expand_dims(1 - sigmoid(dot_two), axis=1) * neg_samples), axis=0)

    gradOutsideVecs = np.zeros(shape=outsideVectors.shape)
    gradOutsideVecs[outsideWordIdx] = -(1 - sigmoid(dot_one)) * centerWordVec
    for i, neg_idx in enumerate(negSampleWordIndices):
        gradOutsideVecs[neg_idx] += (1 - sigmoid(dot_two)[i]) * centerWordVec
```

#### skip-gram

  ***skip-gram***模型主要是根据***center word($w_t$)***来预测其***context  window words([ $w_{t - m}, \dots, w_{t - 1}, w_{t + 1}, \dots, w_{t + m}$ ])***，采取***negative sampling*** 的***skip-gram***模型的损失函数为：
$$
J_{skip-gram} (v_c, w_{t - m}, \dots, w_{t + m}, U) = \sum _{-m \leq j \leq m, j \neq 0} J_{neg-sample} (v_c, w_{t + j}, U)
$$
只需将***center word***与***context  window words***所求的损失函数值相加即可；而其梯度也是如此。

具体实现：

```
    for w in outsideWords:
        l, gradCenterVec, gradOutsideVecs = word2vecLossAndGradient(centerWordVectors[word2Ind[currentCenterWord]],
                                                                          word2Ind[w], outsideVectors, dataset)

        loss += l
        gradCenterVecs[word2Ind[currentCenterWord]] += gradCenterVec
        gradOutsideVectors += gradOutsideVecs
        gradOutsideVectors
```

___

#### 词向量训练的大致流程

  对词向量进行分词，然后构成词库，即形成词的***one-hot***形式；然后根据使用的模型，抽样选出***center word***和***context word***，分别***embedding***成**$v_c$**和**$u_o$**；然后就可代入相应的损失函数中，进行计算并完成反向误差传播。