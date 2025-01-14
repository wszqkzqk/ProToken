# ProToken: Differentiable Protein Engineering through using Diffusion Transformer over Neural Amino Acids
This is the github repo for the paper *Differentiable Protein Engineering through using Diffusion Transformer over ProTokens*. An early version is preprinted at [biorxiv](https://doi.org/10.1101/2023.11.27.568722).

<p align="center"><img src="https://github.com/issacAzazel/ProToken/blob/main/figs/model_arch.jpg" width="100%"></p>
**ProTokens**, developed by the Gao Group at Peking University, machine-learned "amino acids" derived from protein structure databases via self-supervised learning, providing a compact yet informative representation that bridges "1D" (sequence) and "3D" (structure) modalities of proteins. Building on **ProTokens**, we develop **PT-DiT**, a diffusion transformer that jointly models protein sequences and structures through a generative objective. **ProTokens** and **PT-DiT** enable efficient encoding of 3D folds, contextual protein design, sampling of metastable states, and directed evolution for diverse objectives.

## Installation 
Running example scripts in [example_scripts](./example_scripts) requires:
### Main dependencies
* python==3.10
* jax==0.4.28, jaxlib==0.4.28
* flax==0.8.3
* ml-collections==0.1.1
### Other dependencies
* numpy, scipy, scikit-learn, biopandas, biopython 

In theory, any environment compatible with the packages mentioned above should run successfully. We also provide [environment.yml](environment.yml) which can directly create a compatible conda environment via `conda env create -f environment.yml` (with possible redundant dependencies). Our configuration includes Ubuntu 22.04 (GNU/Linux x86_64), NVIDIA A100-SXM4-80GB, CUDA 12.2 and Anaconda 23.7.2. The complete notebook execution takes approximately 0.5 hours.

## Citation
```python
@article{lin2023tokenizing,
    title={Tokenizing Foldable Protein Structures with Machine-Learned Artificial Amino-Acid Vocabulary},
    author={Lin, Xiaohan and Chen, Zhenyu and Li, Yanheng and Ma, Zicheng and Fan, Chuanliu and Cao, Ziqiang and Feng, Shihao and Gao, Yi Qin and Zhang, Jun},
    journal={bioRxiv},
    pages={2023--11},
    year={2023},
    publisher={Cold Spring Harbor Laboratory}
}
```

## Contact
For questions or further information, please contact [jzhang@cpl.ac.cn](jzhang@cpl.ac.cn).