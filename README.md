CROWN-IBP: Towards Stable and Efficient Training of Verifiably Robust Neural Networks
========================

CROWN-IBP is a certified defense to adversarial examples, which combines a
tight linear relaxation based verification bound (CROWN) with the non-linear
and efficient interval bound propagate (IBP) training method.  We achieved the
best verifiable error on MNIST: **7.46\%** at `epsilon=0.3` and **12.96\%** at
`epsilon=0.4` (L\_infinity norm distortion).

Our repository provides **high quality PyTorch implementations** of Pure-IBP,
Natural-IBP [(Gowal et
al., 2018)](https://github.com/deepmind/interval-bound-propagation), CROWN-IBP (our
algorithm) and [Convex Adversarial
Polytope](https://github.com/locuslab/convex_adversarial) (Wong et al., 2018).

Intro
---------

We aim to train a robust deep neural network that is *verifiable*: given a test
example and a certain perturbation radius `epsilon`, it is able to verify if
there is definitely no adversarial example inside, or there might exist an
adversarial example, through some efficient neural network verification
algorithm. Conversely, PGD based adversarial training is generally not a
verifiable training method, as existing verification methods typically give
vacuous bounds for a PGD-trained network, and there might exists stronger
attacks that can break a PGD-trained model.

The *certified robustness* of a model can be evaluated using *verified error*,
which is a guaranteed *upper bound* of test error under *any* attack.
CROWN-IBP can achieve **7.46\%** verified error on MNIST test set at
`epsilon=0.3`.  The verified error is even lower than the unverified PGD error
(around **12\%**) provided by [Adversarial training by Madry et
al.](https://github.com/MadryLab/mnist_challenge) using PGD-based adversarial
training. Previous convex relaxation based method by [Wong et
al.](https://github.com/locuslab/convex_adversarial) has about **40\%**
verified error.

CROWN-IBP combines [Interval Bound Propagation
(IBP)](https://arxiv.org/abs/1810.12715) training (Gowal et al.  2018) and
[CROWN (Zhang et al. 2018)](http://arxiv.org/pdf/1811.00866.pdf), a tight
linear relaxation based verification method. Ordinary CROWN is very expensive
for training, and it may also over-regularize the network.  IBP along
outperforms many existing linear relaxation based methods due to its non-linear
representation power (see our paper for more explanation), however since IBP
bounds are very loose at the beginning phase of training, training stability is
a big concern.  We propose to use IBP in a forward pass to compute all
intermediate layers' pre-activation bounds and use a CROWN-style bound to
obtain the final layer's bounds in a backward pass. The combination of IBP and
CROWN gives us an efficient, stable and well-performing certified defense
method, strictly within the framework of robust optimization.

Getting Started with the Code
---------

Our program is tested on Pytorch 1.1 and Python 3.6. 

We have all training parameters included in JSON files, under the `config` directory.
We provide configuration files which can reproduce all CROWN-IBP models in our paper.

To train CROWN-IBP on MNIST, 10 small models, run:
```bash
python train.py --config config/mnist_crown.json
```

To train CROWN-IBP on MNIST, 8 large models, run:
```bash
python train.py --config config/mnist_crown_large.json
```

You will also find configuration files for 9 small CIFAR models, 8 large CIFAR models,
10 Fashion-MNIST small models and 8 Fashion-MNIST large models in `config` folder.

We also implement baseline methods including Pure-IBP, Natural-IBP (Gowal et al. 2018)
and Convex adversarial polytope (Wong et al. 2018). They can be used by adding command
line parameters to override the training method defined in the JSON file. For example,

```bash
# for Pure-IBP
python train.py "training_params:method_params:bound_type=interval" --config config/mnist_crown.json 
# for Natural-IBP (Gowal et al., 2018)
python train.py "training_params:method=robust_natural" "training_params:method_params:bound_type=interval" --config config/mnist_crown.json
# for Convex adversarial polytope (Wong et al. 2018)
python train.py "training_params:method_params:bound_type=convex-adv" --config config/mnist_crown.json
```

All hyperparameters can also be changed in the configuration JSON file.

Pre-trained Models
----------------

CROWN-IBP pretrained models used in our paper can be downloaded here:

```bash
wget https://download.huan-zhang.com/models/crown-ibp/models_crown-ibp.tar.gz
tar xvf models_crown-ibp.tar.gz
```

The folder `crown-ibp_models` contains several directories, each one
corresponding to a set of models and a `epsilon` value.  They can be evaluated
using the script `eval.py`. For example, to evaluate the 8 large MNIST models
under `epsilon=0.4` use this command:

```bash
python eval.py "eval_params:epsilon=0.4" --config config/mnist_crown_large.json --path_prefix crown-ibp_models/mnist_0.4_mnist_crown_large
```

The parameter `"eval_params:epsilon=0.4"` overrides the `epsilon` in configuration file,
and the parameter `--path_prefix` changes the default path that stores models and logs.

Reproduce the State-of-the-art MNIST Defense
----------------

To train the best MNIST defense model at `epsilon=0.3`, run this command

```bash
python train.py --config config/mnist_crown_large.json --model_subset 4
```

The argument `--model_subset` selects the 4th model defined in configuration
file, which is the largest model in our model pool. You should be able to
achieve about 7.5% verified error after 100 epochs.  This is a large model, so
training will be slower. Note that our implementation of CROWN-IBP is still
preliminary and not optimized, and further speedups can be achieved after code
optimization.

For `epsilon=0.4` run this command:

```bash
python train.py "training_params:epsilon=0.4" --config config/mnist_crown_large.json --model_subset 4
```

where the parameter `"training_params:epsilon=0.4"` overrides the corresponding
`epsilon` value in configuration file.

Training Your Own Robust Model Using CROWN-IBP
-----------------

Our implementation can be easily extended to other datasets or models. You only need to do three things:

* Add your dataset loader to `datasets.py` (see examples for MNIST, Fashion-MNIST, CIFAR-10 and SVHN)
* Add your model architecture to `model_defs.py` (you can also reuse any existing models)
* Create a JSON configuration file for training. You can copy from `crown_mnist.json` or `crown_cifar.json`.
You need to change `"dataset"` name in config file, also update model structure in the `"models"` section.

For example, the following in `"models"` section of `"mnist_crown.json"` defines a model with 2 CNN layers:

```json
{
    "model_id": "cnn_2layer_width_1",
    "model_class": "model_cnn_2layer",
    "model_params": {"in_ch": 1, "in_dim": 28, "width": 1, "linear_size": 128}
}
```

where "model\_id" is an unique name for each model, "model\_class" is the
function name to create the model in `model_defs.py`, and "model\_params" is a
dictionary of all parameters passing to the function that creates the model.
Then your will be able to train with CROWN-IBP with your JSON:

```bash
python train.py --config your_config.json
```

Reproducing Paper Results
-------------------

In our paper, we evaluate training stability by setting different `epsilon`
schedule length among different models and methods. `epsilon` schedule length
can be controlled through changing the configuration file, or overriding
configuration file parameters as shown below.

To train CROBW-IBP on MNIST, 10 small models with `epsilon=0.3` and schedule length as 10, run this command:
```bash
python train.py training_params:schedule_length=11 --config config/mnist_crown.json 
```
To train Pure-IBP on MNIST, 10 small models with `epsilon=0.3` and schedule length as 10, run this command:
```bash
python train.py training_params:schedule_length=11 training_params:method_params:bound_type=interval --config config/mnist_crown.json 
```
To train Natural IBP with final `kappa=0.5` on MNIST, 10 small models with `epsilon=0.3` and schedule length as 10, run this command:
```bash
python train.py training_params:schedule_length=11 training_params:method_params:bound_type=interval training_params:method_params:final-kappa=0.5 training_params:method=robust_natural --config config/mnist_crown.json  
```
To train Natural IBP with final `kappa=0` on MNIST, 10 small models with `epsilon=0.3` and schedule length as 10, run this command:
```bash
python train.py training_params:schedule_length=11 training_params:method_params:bound_type=interval training_params:method_params:final-kappa=0 training_params:method=robust_natural --config config/mnist_crown.json
```

References
-------------------

We stand on the shoulders of giants and we greatly appreciate the inspiring
works of pioneers in this field.  Here we list references directly related to
this README. A full bibliography can be found in our paper.

Gowal, S., Dvijotham, K., Stanforth, R., Bunel, R., Qin, C., Uesato, J., Mann,
T., and Kohli, P. On the effectiveness of interval bound propagation for
training verifiably robust models. arXiv preprint arXiv:1810.12715, 2018.

Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. Towards deep
learning models resistant to adversarial attacks. In International Conference
on Learning Representations, 2018.

Zhang, H., Weng, T.-W., Chen, P.-Y., Hsieh, C.-J., and Daniel, L. Efficient
neural network robustness certification with general activation functions. In
Advances in Neural Information Processing Systems (NIPS), 2018.
