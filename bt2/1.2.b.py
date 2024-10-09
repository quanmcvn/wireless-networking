from math import pi, sin
import matplotlib.pyplot as plt

Ts= 0.0001  # % khoảng thời gian lấy mẫu (s)

t = range(0, int(0.5 / Ts), 1) #0:Ts:0.5; #% thời gian mô phỏng (s)
real_t = [t_ * Ts for t_ in t]

def A(t):
	if 0.2 <= t <= 0.3:
		return 1
	return 0

f = 100;        #% tần số (Hz)

phi = 0;    #   % pha

s = [A(t_ * Ts) *sin(2*pi*f*t_ * Ts + phi) for t_ in t]

plt.title('A(t) * sin(2*pi*f*t)')
plt.plot(real_t, s)
plt.xlabel('t')
plt.ylabel('s')

plt.show()
