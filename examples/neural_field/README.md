
## Example scripts for neural-field localization


### Setup

Apart from Theseus, install the following:

```bash
pip install open3d trimesh pyrender opencv-python gdown
```

Download the SDF/mesh data: 
```bash
cd examples/neural_field
gdown --fuzzy https://drive.google.com/file/d/1kz8k849n1shLDkSOu_8ngueHd1Zti77Q/view?usp=sharing
unzip gt_sdfs.zip
rm gt_sdfs.zip
```

### Run scripts 

```bash
python neural_field.py optimizer=ADAM 
```

*TODO*: shows local-minima problem with large gradient updates in translation over rotation, leading to wrong solution 

```bash
python neural_field.py optimizer=GN 
```

*TODO*: show ill-conditioned error with the AutoDiffCostFunction (The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: 2).)

https://drive.google.com/file/d/1kz8k849n1shLDkSOu_8ngueHd1Zti77Q/view?usp=sharing