#!/usr/bin/env python3

import pydicom
import sys

f = pydicom.dcmread(sys.argv[1])
# print(dir(f))
print("%d by %d" % (f.Rows, f.Columns))
print("slope %f, intercept %d" % (f.RescaleSlope, f.RescaleIntercept))
f.pixel_array.tofile(sys.argv[2])

