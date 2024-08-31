# -*- coding: utf-8 -*-
# Time    : 2023/12/15 14:30
# Author  : fanc
# File    : drop_result.py

import os
import shutil
files = os.listdir('results')
for file in files:
    if not os.path.exists(f'results/{file}/models') or not os.listdir(f'results/{file}/models'):
        shutil.rmtree(f'results/{file}')
        print(f'{file} is remove')