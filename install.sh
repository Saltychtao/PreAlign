#!/bin/bash
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

pip install sentencepiece
pip install deepspeed
pip install accelerate
pip install transformers==4.31.0
pip install datasets
pip install fsspec==2023.9.2
