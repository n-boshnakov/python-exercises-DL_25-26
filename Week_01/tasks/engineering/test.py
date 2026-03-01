import matplotlib.pyplot as plt

a = list(range(1,10))
b = list(map(lambda x: x**2, a))
c = list(map(lambda x: x**3, a))
d = list(map(lambda x: x+200, a))

plots = []

fig=plt.figure()
plt.plot(a, b)
plots.append(fig)
plt.close()

fig=plt.figure()
plt.plot(a, c)
plots.append(fig)
plt.close()

fig=plt.figure()
ax3= plt.plot(b, c)
plots.append(fig)
plt.close()

fig=plt.figure()
plt.plot(a, d)
plots.append(fig)
plt.close()

fig,axs =plt.subplots(2,2)
axs[0,0].plot(a,b)
axs[0,1].plot(a,c)
axs[1,0].plot(b,c)
axs[1,1].plot(a,d)
plt.tight_layout()
plt.show()