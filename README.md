# RibOrthrus: A whole-transcript deep learning pipeline to predict RNA-seq and Ribo-seq coverage

- System requirements:

  This code has been tested on a system with a GH200 Superchip (with pytorch-gpu installed). The required softwares are listed in env.yml

- To install project requirements:
  ```bash
  git clone https://github.com/bowang-lab/Orthrus.git
  cd [orthrus project directory]
  mamba env create -f env.yml
  conda activate orthrus
  pip install causal_conv1d==1.2.0.post2
  pip install mamba-ssm==1.2.0.post1 --no-cache-dir
  pip install -e .
  cd ../
  git clone [riborthrus git link]
  pip install jaxtyping
  pip install ribopy
  pip install lightning
  ```
