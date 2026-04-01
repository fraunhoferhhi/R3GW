# R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild

![Teaser image](./assets/Teaser.png)

 <div align="center">
  <p align="center">
    <a href="https://fraunhoferhhi.github.io/R3GW/"><strong>Project Page</strong></a>
    ·
    <a href="https://arxiv.org/pdf/2603.02801v1" target="_blank"><strong>arXiv</strong></a>
  </p>

</div>

### Installation and Dataset

Clone the repository:
```
git clone git@github.com:fraunhoferhhi/R3GW.git
cd R3GW
```
Create the environment:
```
conda env create --file environment.yml
conda activate R3GW
```
Install manually nvdiffrast:
```
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```
To train and evaluate our model we used the version of the [NeRF-OSR](https://4dqv.mpi-inf.mpg.de/NeRF-OSR/) dataset provided by [LumiGauss](https://lumigauss.github.io/), which can be downloaded [here](https://zenodo.org/records/15455694).

Our model requires sky masks for training. For each scene, these masks must be generated in advance and stored in a floder named `sky_masks` within the directory `undistorted`. We generated the sky masks from the segmentation masks provided by [NeuSky](https://github.com/JADGardner/neusky).

## Usage
The script `run_all.sh` contains instructions to train, render, and evaluate all scenes, while `run_relighting.sh` provides a relighting example.

## Acknowledgments
We acknowledge the following amazing repositories that contributed to the development of our code:

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- [LumiGauss: Relightable Gaussian Splatting in the Wild](https://github.com/joaxkal/lumigauss)
- [GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces](https://github.com/Asparagus15/GaussianShader)
- [nvdiffrec](https://github.com/NVlabs/nvdiffrec)
- [Spherical Harmonics by Andrew Chalmers](https://github.com/chalmersgit/SphericalHarmonics)

## Citation

If you use our method in your research, please cite our paper. You can use the following BibTeX entry:

```bibtex
@InProceedings{corona2026r3gw,
  author    = {Margherita Lea Corona and Wieland Morgenstern and Peter Eisert and Anna Hilsmann},
  title     = {R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild},
  booktitle = {Proceedings of the 21st International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISAPP 2026)},
  year      = {2026},
  pages     = {432-443},
  publisher = {SCITEPRESS - Science and Technology Publications},
  doi       = {https://doi.org/10.5220/0014332200004084}
}
```


