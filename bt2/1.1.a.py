from math import pi, sin
import matplotlib.pyplot as plt

Ts= 0.0001  # % khoảng thời gian lấy mẫu (s)

t = range(0, int(0.5 / Ts), 1) #0:Ts:0.5; #% thời gian mô phỏng (s)
real_t = [t_ * Ts for t_ in t]
A = 1;        # % biên độ

f = 100;        #% tần số (Hz)

phi = 0;    #   % pha

s = [A *sin(2*pi*f*t_ * Ts + phi) for t_ in t]

plt.title('Sine wave 100Hz')
plt.plot(real_t, s)
plt.xlabel('t')
plt.ylabel('s')

plt.show()
