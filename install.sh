conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install "transformers==4.26.0" "datasets==2.9.0" "accelerate==0.16.0" "evaluate==0.4.0" --upgrade
# install deepspeed and ninja for jit compilations of kernels
pip install "deepspeed==0.8.0" ninja --upgrade
pip install sacrebleu
pip install sacremoses
pip install langcodes
pip install sentencepiece
