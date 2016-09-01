
import numpy as np
from   scipy.optimize import brentq
import matplotlib.pyplot as plt

data = np.loadtxt('zvu_d-05.dat')
x0 = data[:,0]
y0 = data[:,4]

fitx= np.polyfit(x0, y0, 3)
fun = np.poly1d(fitx)
print brentq(fun-1.0,0.0, 10.0)

nint = 20
x1 = np.linspace(0,4.0,nint)
y1 = np.zeros(nint)
for ix, xx in enumerate(x1):
    y1[ix] = fun(xx)

plt.ylim(0.0,1.0)
plt.plot(x0,y0,'o',markersize=8)
plt.plot(x1,y1,'s',markersize=8)
plt.show()
