MNIST dataset: <http://yann.lecun.com/exdb/mnist>

```
# MNIST data 불러오기 (Tensorflow 내장 함수)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
```