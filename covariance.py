x=[15,  12,  8,   8,   7,   7,   7,   6,   5,   3]
y=[10 , 25,  17,  11,  13,  17,  20,  13,  9,   15]

def Corr(X,Y):
    x2 = []
    y2 = []
    xy = []
    varx =[]
    vary=[]

    sumx, sumy, sumxy, sumx2, sumy2,sumvarx,sumvary = 0, 0, 0, 0, 0,0,0
    for i in range(len(x)):
        x[i] = float(x[i])
        sumx = sumx + x[i]
        y[i] = float(y[i])
        sumy = sumy + y[i]
        x2.append(x[i] * x[i])
        sumx2 = sumx2 + x2[i]
        y2.append(y[i] * y[i])
        sumy2 = sumy2 + y2[i]
        xy.append(x[i] * y[i])
        sumxy = sumxy + xy[i]

    import math
    u = (len(x) * sumxy - (sumx * sumy))
    d1 = (len(x) * sumx2 - (sumx * sumx))
    d2 = (len(y) * sumy2 - (sumy * sumy))
    #print x
    #print y
    d = math.sqrt(d1 * d2)
    pearsons = u / d
    #print pearsons
    #import numpy as np
    #print np.corrcoef(x, y)

    meanx = sumx / len(x)
    meany = sumy / len(y)
    for i in range(len(x)):
        varx.append(math.pow((x[i]-meanx),2))
        sumvarx=sumvarx+varx[i]
        vary.append(math.pow((y[i] - meany),2))
        sumvary = sumvary + vary[i]
    slope=pearsons*(math.sqrt(sumvary)/math.sqrt(sumvarx))
    #print round(slope,3)
    intercept=meany-slope*meanx
    #print intercept
    print (format((slope * 10 + intercept), '.3f'))
def m(x,y):
    ar1=x
    ar2=y
    sx = 0;
    sy = 0;
    sxy = 0;
    sx2 = 0;
    n = len(ar1)
    for i in range(len(ar1)):
        sx = sx + ar1[i]
        sy = sy + ar2[i]
        sxy = sxy + ar1[i] * ar2[i]
        sx2 = sx2 + ar1[i] ** 2
    m = ((n * sxy - sx * sy) / (n * sx2 - sx ** 2))
    c = (sy * (sx2) - sx * sxy) / (n * sx2 - sx ** 2)
    print("m ",m)
    print("c",c)
    print (format((m * 10 + c), '.3f'))

Corr(x,y)





