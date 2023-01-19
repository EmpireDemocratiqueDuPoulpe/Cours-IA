# CODE FROM THE EXERCISE
def f(x):
    return x**2+1

def df(x):
    return 2*x

def descente(n, a, x0):
    x, gradient, valeurs = [None]*(n+1), [None]*(n+1), [None]*(n+1)
    x[0], gradient[0], valeurs[0] = x0, df(x0), f(x0)
    for i in range(n):
        x[i+1]=x[i]-a*gradient[i]
        gradient[i+1]=df(x[i+1])
        valeurs[i+1]=f(x[i+1])
    return x, gradient, valeurs

def display(x, gradient, valeurs):
    print("k      xk     f\'(xk)   f(xk)")
    for k in range(len(x)):
        print(f"{k:.0f}    {x[k]:.3f}    {gradient[k]:.3f}    {valeurs[k]:.3f}")
# END OF CODE FROM THE EXERCISE

# Q2.1
print("#### Q2.1 #################################")
x1, gradient1, values1 = descente(n=20, a=0.2, x0=2)
display(x=x1, gradient=gradient1, valeurs=values1)

# Q2.2
print("#### Q2.2 #################################")
x2, gradient2, values2 = descente(n=20, a=0.9, x0=2)
display(x=x2, gradient=gradient2, valeurs=values2)

# Q2.3
print("#### Q2.3 #################################")
x3, gradient3, values3 = descente(n=20, a=1.1, x0=2)
display(x=x3, gradient=gradient3, valeurs=values3)

# Q2.4
print("#### Q2.4 #################################")
x4, gradient4, values4 = descente(n=20, a=0.05, x0=2)
display(x=x4, gradient=gradient4, valeurs=values4SQ)
