import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def edo(t,y):
    f = 17.5
    w = 1
    r = 0.2
    v = y[0]
    x = y[1]
    wo2 = -1
    b = 0.1
    dvdt = f * np.cos(w*t) - r*v - wo2*x - b*x**3
    dxdt = v

    return [dvdt,dxdt]

condições_iniciais = [0,0] # velocidade e posição
t0 = 0
tfinal = 40000
incremento = 0.001
ts = np.arange(t0, tfinal, incremento)

sol = solve_ivp(edo,t_span=[t0,tfinal],y0=condições_iniciais,t_eval=ts)

x = sol.y[1]
v = sol.y[0]

# plt.plot(x,v)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
# plt.close()
T = 2 * np.pi  # Período do sistema
indice_poincare = np.arange(0, len(ts), int(T / (ts[1] - ts[0])))
plt.figure()
plt.scatter(x[indice_poincare], v[indice_poincare], s=1)
plt.xlabel('Posição (x)')
plt.ylabel('Velocidade (v)')
plt.title('Seção de Poincaré')
plt.show()