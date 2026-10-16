# Diff Gaussian Rasterization 

This repository is a fork of the original [Diff Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) used in "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields".

## Modifications

This version includes **Depth Rendering & Gradients** by adding support for rendering inverse depth maps for depth regularization in 3DGS-MCMC, Calculation of gradients with respect to inverse depth (`dL_dout_invdepth`), allowing for depth-based supervision.

## Original Citation

If you use this code, please cite the original paper:

```bibtex
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## License

Please refer to `LICENSE.md` for license details (inherited from the original repository).
