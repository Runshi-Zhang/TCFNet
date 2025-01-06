# TCFNet: Bidirectional Face-bone Transformation by a Transformer-based Coarse-to-ﬁne Point Moving Network
Here is the <strong><big>PyTorch implementation</big></strong> of the paper.
## Update progress
1/6/2025 - TCFNet is now publicly available!
## Requirments
conda create -n point python=3.8 -y

conda activate point

conda install ninja -y

\# We trained our models depending on Pytorch 2.1.0+cu118 and Python 3.8.

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install h5py pyyaml -c anaconda -y

conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y

conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

pip install torch-geometric

cd pointnet2_ops_lib

python setup.py install

cd ..

cd Chamfer3D

python setup.py install

cd ..

cd earth_movers_distance

python setup.py install

cd ..

\# spconv (SparseUNet)

\# refer https://github.com/traveller59/spconv

pip install spconv-cu118  # choose version match your local cuda version

\# Open3D (visualization, optional)

pip install open3d


## Train and infer
'main-p2p' is used to train different core file.
'core/test_jsd_local.py' is used to test.
## Reference and Acknowledgments
[P2PConv](https://github.com/Marvin0724/Face_bone_transform)

[Point Transformer V3](https://github.com/Pointcept/PointTransformerV3)
