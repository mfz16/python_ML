import scipy as sp
data = sp.genfromtxt("d:/datasets/web_traffic.tsv")
print(data[:10])
x = data[:,0]
y = data[:,1]
print (x[0])
x[0]=1
print ("x before removal")
print(x)
print("y before removal")
print(y)
s=sp.sum(sp.isnan(y)) #no of nan values
print (s)

x = x[~sp.isnan(y)] #remove nan valus from x and y by negating
y = y[~sp.isnan(y)]
print ("x after removal")
print(x)
print("y after removal")
print(y)
import matplotlib.pyplot as plt
# plot the (x,y) points with dots of size 10
plt.scatter(x, y, s=10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
 ['week %i' % w for w in range(10)])
#plt.xticks([1*7*24,2*7*24,3*7*24,4*7*24,],('week1','week2','week3','week4'))
plt.autoscale(tight=True)
# draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-', color='0.75')


def error(f, x, y):
 return sp.sum((f(x)-y)**2)

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print("Model parameters: %s" % fp1)
print (residuals)

f1 = sp.poly1d(fp1)
print(error(f1, x, y))

fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
plt.plot(fx, f1(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")


fp2 = sp.polyfit(x, y, 2)
print(fp2)

f2 = sp.poly1d(fp2)
print(error(f2, x, y))
plt.plot(fx, f2(fx), linewidth=3)
#plt.legend(["d=%i" % f2.order], loc="upper left")

fp3 = sp.polyfit(x, y, 3)
f3 = sp.poly1d(fp3)
plt.plot(fx, f3(fx), linewidth=3)
#plt.legend(["d=%i" % f3.order], loc="upper left")

fp10 = sp.polyfit(x, y, 10)
f10 = sp.poly1d(fp10)
plt.plot(fx, f10(fx), linewidth=3)

fp100 = sp.polyfit(x, y, 53)
f100 = sp.poly1d(fp100)
plt.plot(fx, f100(fx), linewidth=3)
plt.legend(["d1","d2","d3","d10","d53"], loc="upper left")
#plt.legend(["d=%i" % f1.order], loc="upper left")

inflection = 3.5*7*24 # calculate the inflection point in hours

xa = x[:inflection] # data before the inflection point
ya = y[:inflection]
xb = x[inflection:] # data after
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
plt.plot(fx,fa(fx),linewidth=4)
print("Error inflection=%f" % (fa_error + fb_error))
plt.show()

