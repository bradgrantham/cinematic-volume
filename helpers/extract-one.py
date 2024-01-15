#!/usr/bin/env python3

import pydicom
import sys

f = pydicom.dcmread(sys.argv[1])
# print(f)
print("%d by %d" % (f.Rows, f.Columns))
f.pixel_array.tofile(sys.argv[2])

