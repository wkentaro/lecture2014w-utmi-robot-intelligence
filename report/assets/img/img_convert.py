#!/usr/bin/env python
# -*- coding: utf-8 -*-
# img_convert.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import os

files = os.listdir('.')
for file in files:
    base, ext = os.path.splitext(file)
    if ext in ['.png', '.jpeg', '.jpg']:
        os.system('convert {0} {1}.eps'.format(file, base))
        # os.system('convert {0} {1}.pdf'.format(file, base))
        # os.system('extractbb {0}'.format(file))