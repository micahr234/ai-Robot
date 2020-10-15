# ai-Robot

Installation instructions for Windows:

1. Download and install Python 3.7 x64

2. Install pytorch
Go to https://pytorch.org/get-started/locally/ for most current installation commands. Currently:

       pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
    
3. Install gym

       pip install gym
       pip install git+https://github.com/Kojoley/atari-py.git
      
4. Install Microsoft Visual C++ 14.0 (only if PyBullet environments are used)

Goto the URL below and look for the built tool installer. Install C++ built tools.
https://visualstudio.microsoft.com/downloads/

Download and install the C++ redistributable
https://aka.ms/vs/16/release/vc_redist.x64.exe

5. Install PyBullet (only if PyBullet environments are used)

       pip install pybullet
       
7. Install Unity (only if Unity environments are used)

Reference the URL for further instructions.
https://github.com/Unity-Technologies/ml-agents/blob/release_6_docs/docs/Installation.md

Install the Unity devoloper. Now add the com.unity.ml-agents Unity package.
Now install the following python packages:

       pip install mlagents-envs
       pip install gym-unity
    
6. Install Tensorboard

       pip install tensorboard
