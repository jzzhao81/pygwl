
import numpy as np
from   scipy.optimize import brentq
import matplotlib.pyplot as plt

data = np.loadtxt('zvu.dat')
x0 = data[:,0]
y0 = data[:,1]

fitx= np.polyfit(x0, y0, 5)
fun = np.poly1d(fitx)
print brentq(fun, 0.0, 10.0)

nint = 40
x1 = np.linspace(2,7.0,nint)
y1 = np.zeros(nint)
for ix, xx in enumerate(x1):
    y1[ix] = fun(xx)

plt.ylim(0.0,1.0)
plt.plot(x0,y0,'o',markersize=8)
plt.plot(x1,y1,'--',linewidth=2)
plt.show()
