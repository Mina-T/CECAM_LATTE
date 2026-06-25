#!/usr/bin/env bash

# 1. create env
cd ~
python3 -m venv venv

# 2. activate it
source venv/bin/activate
echo Virtual environment created
# 3. install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4. clone + install your gitlab package
git clone https://gitlab.com/PANNAdevs/panna.git
cd panna

pip install .
echo LATTE was successfully installed
# 5. Create the working directories and download the datasets
cd ~/Desktop
mkdir LATTE
cd LATTE
echo ⬇️  Now downloading PANNA ...
git clone https://github.com/Mina-T/CECAM_LATTE.git
cd CECAM_LATTE
echo ✅ Ready to go!
