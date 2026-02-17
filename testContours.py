import matplotlib.pyplot as plt
import numpy as np

from contours2D import get_kpoints_and_weights_FS

NK = 200
x = np.linspace(0, 1, NK, endpoint=False)
y = np.linspace(0, 1, NK, endpoint=False)

x1 = np.hstack([x, [1]])
y1 = np.hstack([y, [1]])


xg = x[:, np.newaxis]
yg = y[np.newaxis, :]

z = np.zeros((NK, NK))


for g, v in [((1, 0), 0.4),
             ((0, 2), 0.3), ((0, 1), 2.1), ((2, 1), 0.4),
             ((1, 2), -2.4), ((2, 0), 0.3)
             ]:
    z += v * np.cos(2*np.pi*(g[0]*xg + g[1]*yg))


ef = np.linspace(-10, 10, 100)
dos = []
conductivity = []
for e in ef:
    kpoints, weights, grad = get_kpoints_and_weights_FS(
        z, np.eye(2), e, gradient=True)
    dos.append(np.sum(weights))
    conductivity.append(np.einsum('i,ia,ib->ab', weights, grad, grad))

conductivity = np.array(conductivity)


plt.plot(ef, dos)
plt.xlabel('Energy')
plt.ylabel('Density of states')
plt.title(f'Density of states for a 2D system, sum = {
          np.sum(dos)*(ef[1]-ef[0])}')
plt.show()

plt.plot(ef, conductivity[:, 0, 0], label='sigma_xx')
plt.plot(ef, conductivity[:, 1, 1], label='sigma_yy')
plt.plot(ef, conductivity[:, 0, 1], label='sigma_xy')
plt.plot(ef, conductivity[:, 1, 0], label='sigma_yx')
plt.legend()
plt.xlabel('Energy')
plt.ylabel('Conductivity')
plt.title('Conductivity for a 2D system')
plt.show()
