#!/bin/bash

cd OpenRadar
pip install -r requirements.txt
python setup.py develop

cd ..
cd Emd
python3 setup.py install