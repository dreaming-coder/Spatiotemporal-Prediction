# Spatiotemporal-Prediction

<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/dreaming-coder/Spatiotemporal-Prediction">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/dreaming-coder/Spatiotemporal-Prediction">
    <img src="https://img.shields.io/github/stars/dreaming-coder/Spatiotemporal-Prediction?style=social" alt="github star"/>
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/dreaming-coder/Spatiotemporal-Prediction?style=social">
</p>

You are looking at the repository for the spatiotemporal prediction neural networks. I override such great works with
PyTorch according the papers and the codes if the authors provide. Since I'm very 'vegetable', there are some latent
bugs hiding in the implements. So just think of twice while reading rather than be insatiable when get errors with the
direct copy.

## EnhancedModule and Trainer

The native codes for models and training processes may be tedious as we have to repeat some common blocks for each
model. Recently, the `PyTorch-Lightning` sounds great while the usage is a little hard to me. Therefore, I refer the
design idea of `PyTorch-Lightning` to design my own tool class: `EnhancedModule` and `Trainer`.

### EnhancedModule

This class inherits from `torch.nn.Module` with some additional aspect methods to custom the operation in training,
validation or test loop while the repeatable blocks are packaged in the `Trainer` class. The additional methods you may
need to override is shown as the following:

- `configure_optimizer()`
- `configure_lr_scheduler()`
- `training_step()`
- `training_step_end()`
- `validation_step()`
- `validation_step_end()`
- `predict_step()`

> See details in the source code and examples.

### Trainer

The `Trainer` class just provide 2 functions: `fit()` to train and valid dataset and `predict()` to generate the outputs
by the convergent network. Specially, the main process in `fit()` is shown below.
<p align="center">
    <img src="resources/imgs/trainer-process.png" />
</p>

## Tools

### Dataset

I hava collected some open datasets about spatiotemporal
prediction——<a href="https://pan.baidu.com/s/1XGZFQuu-4RXEntiVnQ-GVw">Moving MNIST</a>
, <a href="https://pan.baidu.com/s/1A4OPwg7cMoXCZtYYfkYFJA">KTH</a>
and <a href="https://pan.baidu.com/s/13x3VGEFLIUm2o284glxv_g">Taxi BJ</a> with the password `LOVE`. The prepared `train`
, `validation` and `test` folder is what the `XXXDataset` load, such
as <a href="./src/utils/data/MovingMNISTDataset.py">MovingMNISTDataset</a>,
<a href="./src/utils/data/KTHDataset.py">KTHDataset</a>, <a href="./src/utils/data/TaxiBJDataset.py">TaxiBJDataset</a>.
Therefore, you can download these dataset and load them with my implemented class so that you can pay attention to the
neural network itself.

### Patch

This trick was proposed first by Xing Jian Shi in his paper _Convolutional LSTM Network: A Machine Learning Approach for
Precipitation Nowcasting-NIPS15_convLSTM_. It helps reduce the usage of Video Memory in order to train more samples in
each batch. The following shows how it works.
<p align="center">
    <img src="resources/imgs/patch.png" />
</p>