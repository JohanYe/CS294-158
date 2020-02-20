import numpy as np

def sample_data():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    return np.digitize(samples, np.linspace(0.0, 1.0, 100))

def sample_data2d(seed=0, count=100000):
    data = np.load('Hw1/distribution.npy')
    rng = np.random.RandomState(seed)
    xs = rng.choice(40000, count, p=data.ravel())
    return np.array([(x//200, x%200) for x in xs]) #?????????????