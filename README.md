# New and improved MHA

This repository contains an in improved implementation of the [MHA](https://github.com/piomonti/MHA) algorithm by [Monti and Hyvärinen (2018)](https://arxiv.org/abs/1805.09567) which was used for brain age prediction in [Monti et al. (2020)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232296).

There is a silent [bug](./mha/legacy.py#L16) in the original implementation, which has been fixed in the current implementation. Because of this, the output of the improved implementation differs from the original's.

This implementation greatly improves memory usage, and run time, and fixes some silent bugs that sometime hamper the convergence of the algorithm.

The file `main.py` can be used to run simulations on toy data, and compare the new and legacy implementations
