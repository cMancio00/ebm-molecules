# Tesi

# Install requirements

> [!NOTE]
> The project uses `Python 3.12.3` and `CUDA 12.4`.

In the project directory, execute the following commands:

```bash
python3 -m venv .venv
```
> [!NOTE]
> The name of the virtual environment will match the name of the hidden folder, 
> in this case, `.venv`.

To activate the virtual environment, run:

```bash
source .venv/bin/activate
```
Next, install the required packages with:

```bash
pip install --upgrade pip & pip install -r requirements.txt
```

# Training example

The model can be trained via `CLI` with the following command:

```bash
python3 cli.py fit --trainer.max_epochs=60 --data=DataModules.MNISTDataModule --model.batch_size=256 --model.mcmc_steps=128 --model.mcmc_learning_rate=5.0
```


# Papers

**General EBM**:
- [How to Train Your Energy-Based Models](https://arxiv.org/pdf/2101.03288)
- [Introduction to Latent Variable Energy-Based Models:A Path Towards Autonomous Machine Intelligence](https://arxiv.org/pdf/2306.02572)
- [Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One](https://arxiv.org/pdf/1912.03263)
- [Compositional Visual Generation with Energy Based Models](https://arxiv.org/pdf/2004.06030)
- [Implicit Generation and Modeling with Energy-Based Models](https://arxiv.org/pdf/1903.08689)
- [VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models](https://arxiv.org/pdf/2010.00654)
- [GraphEBM: Molecular Graph Generation with Energy-Based Models](https://arxiv.org/pdf/2102.00546)

**Learning**:
- [In-Context Learning of Energy Functions](https://arxiv.org/pdf/2406.12785)
- [Learning Probabilistic Models from Generator Latent Spaces with Hat EBM](https://arxiv.org/pdf/2210.16486)
- [Learning Discrete Distributions by Dequantization](https://arxiv.org/pdf/2001.11235)
- [Learning Latent Space Hierarchical EBM Diffusion Models](https://arxiv.org/pdf/2405.13910)
- [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)
- [Learning Energy-Based Prior Model with Diffusion-Amortized MCMC](https://arxiv.org/pdf/2310.03218)

**MCMC**:
- [Langevin dynamics with constraints and computation of free energy differences](https://arxiv.org/pdf/1006.4914v2)
- [MCMC using Hamiltonian dynamics](https://arxiv.org/abs/1206.1901v1)