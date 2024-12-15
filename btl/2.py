import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class BFSK:
	def __init__(self):
		# Thông số
		self.bit_rate = 100  # Tốc độ truyền bit (bit/s)
		self.n_bits = 10     # Số bit dữ liệu
		
		self.A = 1           # Biên độ của sóng cơ sở
		self.f_1 = 100       # Tần số của sóng 1
		self.f_2 = 200       # Tần số của sóng 2

		self.C = 3           # Biên độ của sóng mang
		self.f_c = 2000      # Tần số sóng mang (Hz)
		
		self.fs = 10000      # Tần số mẫu (samples per second)

		self.A_n = 0.2       # Cường độ nhiễu AWGN

		self.lo = 0.5        # Hệ số suy hao

	def do_one(self, do_n = False, do_vals = False, do_random_bit=True):
		# Thời gian và tín hiệu điều chế BFSK
		self.omega_1 = 2 * np.pi * self.f_1  # Tần số góc của sóng cơ sở 1
		self.omega_2 = 2 * np.pi * self.f_2  # Tần số góc của sóng cơ sở 2
		self.omega_c = 2 * np.pi * self.f_c  # Tần số góc của sóng mang
		
		if do_random_bit:
			self.bit_sequence = np.random.randint(0, 2, self.n_bits)  # Tạo chuỗi 10 bit ngẫu nhiên
		
		self.T_bit = 1 / self.bit_rate  # Thời gian của 1 bit
		self.T = self.n_bits * self.T_bit    # Tổng thời gian cho tín hiệu điều chế
		self.t = np.linspace(0, self.T, int(self.T * self.fs), endpoint=False)  # Mảng thời gian mẫu

		# Điều chế BFSK
		self.s = np.zeros_like(self.t)
		if do_n:
			self.n = []
			for i in range(self.n_bits):
				for _ in range(int(1/self.T_bit)):
					self.n.append(self.bit_sequence[i])
		for i in range(self.n_bits):
			# Tạo sóng cho mỗi bit (bit = 0 hoặc bit = 1)
			start_time = i * self.T_bit
			end_time = (i + 1) * self.T_bit
			self.s[(self.t >= start_time) & (self.t < end_time)] = \
			    self.A * (self.bit_sequence[i] * np.sin(self.omega_1 * self.t[(self.t >= start_time) & (self.t < end_time)])) \
			  + self.A * ((1 - self.bit_sequence[i]) * np.sin(self.omega_2 * self.t[(self.t >= start_time) & (self.t < end_time)]))

		# Sóng mang (tuy nhiên không dùng như này)
		# self.c = self.C * np.sin(self.omega_c * self.t)
		# Điều biến AM
		self.m = (self.C + self.s) * np.sin(self.omega_c * self.t)  # Tín hiệu AM

		# Tạo nhiễu AWGN
		self.noise = np.sqrt(self.A_n) * np.random.randn(len(self.t))  # Nhiễu Gaussian với cường độ A_n

		# Tín hiệu thu được sau suy hao và nhiễu
		self.r = self.lo * self.m + self.noise

		# Nhân tín hiệu thu được với sóng mang
		r_mult = self.r * np.sin(self.omega_c * self.t)

		# Bộ lọc tần số thấp (Low-pass filter)
		cutoff = 200   # Tần số cắt
		order = 10     # Độ dốc của bộ lọc
		b, a = butter(order, 2 * cutoff / self.fs, btype='low', analog=False)
		z = filtfilt(b, a, r_mult)

		self.s_reconstructed = (2 * z / self.lo) / self.C - self.C + (self.C - self.A) # Không biết tại sao phải cộng (C - A)

		self.x_1 = self.s_reconstructed * (self.A * np.sin(self.omega_1 * self.t))
		self.x_2 = self.s_reconstructed * (self.A * np.sin(self.omega_2 * self.t))

		# Phân biệt bit 0 và 1 dựa vào năng lượng trong mỗi bit
		self.bit_decoded = []
		if do_vals:
			self.vals = []
		for i in range(self.n_bits):
			start_time = i * self.T_bit
			end_time = (i + 1) * self.T_bit
			# Tính tích phân mức năng lượng của tín hiệu trong khoảng thời gian 1 bit
			# z1 = np.trapz(y=np.square(self.s_reconstructed[(self.t >= start_time) & (self.t < end_time)]), x=self.t[(self.t >= start_time) & (self.t < end_time)])
			z1 = np.trapz(y=np.square(self.x_1[(self.t >= start_time) & (self.t < end_time)]), x=self.t[(self.t >= start_time) & (self.t < end_time)])
			z2 = np.trapz(y=np.square(self.x_2[(self.t >= start_time) & (self.t < end_time)]), x=self.t[(self.t >= start_time) & (self.t < end_time)])
			
			if do_vals:
				for _ in range(self.fs // self.bit_rate):
					self.vals.append(z1 - z2)
			# z1 *= (self.fs / self.f)
			# So sánh với ngưỡng để xác định bit
			# def get_val(x):
			# 	coef = 4 * np.pi * self.f
			# 	return (np.sin(coef * x) + coef * x) / (2 * coef)
			# print(self.bit_sequence[i], z1)
			# decide = get_val(end_time) - get_val(start_time)
			if z1 > z2: 
				self.bit_decoded.append(1)
			else:
				self.bit_decoded.append(0)

		bit_error = np.sum(self.bit_sequence != self.bit_decoded)  # Tính số bit bị lỗi
		self.ber = bit_error / self.n_bits  # Tỷ lệ bit bị lỗi

	def part_a(self):
		self.do_one()

		# Vẽ tín hiệu theo thời gian
		plt.figure(figsize=(20, 10))
		plt.subplot(2, 1, 1)
		plt.plot(self.t, self.s)
		plt.title('Tín hiệu sau điều chế BFSK theo thời gian')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		# Tính và vẽ phổ tần số của tín hiệu
		S_f = np.fft.fft(self.s)
		frequencies = np.fft.fftfreq(len(self.t), 1 / self.fs)

		# Chỉ lấy phổ 1 phía (single-sided spectrum)
		half_range = len(frequencies) // 2
		S_f = S_f[:half_range]
		frequencies = frequencies[:half_range]

		plt.subplot(2, 1, 2)
		plt.plot(frequencies, 2 * np.abs(S_f) / len(self.s))  # Biểu diễn phổ tần số
		plt.title('Phổ tần số của tín hiệu BFSK')
		plt.xlabel('Tần số (Hz)')
		plt.ylabel('Biên độ')

		plt.tight_layout()
		plt.show()
	
	def part_b(self):
		l_f_1, l_f_2 = (self.f_1, self.f_2)
		self.f_1 = 50
		self.f_2 = 100
		self.part_a()
		self.f_1, self.f_2 = (l_f_1, l_f_2)

	def part_c(self):
		self.do_one()

		plt.figure(figsize=(20, 10))

		plt.plot(self.t, self.m)
		plt.title('Tín hiệu sau khi điều biến AM')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		plt.tight_layout()
		plt.show()

	def part_d(self):
		self.do_one()

		plt.figure(figsize=(20, 10))

		plt.plot(self.t, self.r)
		plt.title('Tín hiệu thu được r(t) sau suy hao và nhiễu')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		plt.tight_layout()
		plt.show()

	def part_e(self):
		self.do_one()

		plt.figure(figsize=(20, 10))

		plt.subplot(1, 1, 1)
		plt.plot(self.t, self.s_reconstructed)
		plt.title('Tín hiệu sau khi giải điều biến AM')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		print("Dữ liệu gốc: ", self.bit_sequence)
		print("Dữ liệu sau giải điều chế: ", self.bit_decoded)
		print(f"Tỷ lệ bit bị lỗi (BER): {self.ber:.4f}")

		plt.tight_layout()
		plt.show()

	def part_f(self):
		l_A_n, l_n_bits = self.A_n, self.n_bits

		self.n_bits = 100
		self.bit_sequence = np.random.randint(0, 2, self.n_bits)
		# Các giá trị A_n được thử có dạng 1e-6 * 1.1 ** i (i từ 0->300), là hàm mũ
		A_ns = [1e-6]
		for _ in range(300):
			A_ns.append(A_ns[-1] * 1.1)
		bers = []
		snrs = []
		for A_n in A_ns:
			self.A_n = A_n
			self.do_one(do_random_bit=False)
			signal_power = np.mean(self.s ** 2)
			noise_power = np.mean(self.noise ** 2)
			snr_linear = signal_power / noise_power
			snr_db = 10 * np.log10(snr_linear)
			bers.append(self.ber)
			snrs.append(snr_db)
		self.A_n, self.n_bits = l_A_n, l_n_bits
		
		plt.figure(figsize=(20, 10))
		plt.plot(snrs, bers, marker='o', linestyle='-', color='b', label='SNR vs BER')

		plt.xlabel('SNR (dB)')
		plt.ylabel('BER')
		plt.title('SNR vs BER')
		plt.grid(True)

		plt.legend()

		plt.tight_layout()
		plt.show()

	def part_c_x(self):
		self.do_one()

		plt.figure(figsize=(20, 10))

		plt.subplot(3, 1, 1)
		plt.plot(self.t, self.r)
		plt.title('Tín hiệu thu được r(t) sau suy hao và nhiễu')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		R_f = np.fft.fft(self.r)
		frequencies = np.fft.fftfreq(len(self.t), 1 / self.fs)
		half_range = len(frequencies) // 2
		R_f = R_f[:half_range]
		frequencies = frequencies[:half_range]

		plt.subplot(3, 1, 2)
		plt.plot(frequencies, 2 * np.abs(R_f) / len(self.r))
		plt.title('Phổ tần số của tín hiệu r(t)')
		plt.xlabel('Tần số (Hz)')
		plt.ylabel('Biên độ')

		plt.subplot(3, 1, 3)
		plt.plot(self.t, self.s_reconstructed)
		plt.title('Tín hiệu sau khi lọc (s(t) đã tái tạo lại)')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		plt.tight_layout()
		plt.show()

	def part_d_b(self):
		self.do_one(do_n=True, do_vals=True)

		plt.figure(figsize=(20, 10))

		plt.subplot(4, 1, 1)
		plt.plot(self.t, self.n)
		plt.title('Dãy bit ban đầu')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Giá trị')

		plt.subplot(4, 1, 2)
		plt.plot(self.t, self.s)
		plt.title('Tín hiệu sau điều chế BFSK theo thời gian')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		plt.subplot(4, 1, 3)
		plt.plot(self.t, self.s_reconstructed)
		plt.title('Tín hiệu sau khi lọc (s(t) đã tái tạo lại)')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		plt.subplot(4, 1, 4)
		plt.plot(self.t, self.vals)
		plt.title('Hiệu tích phân x_1(t) và x_2(t)')
		plt.xlabel('Thời gian (s)')
		plt.ylabel('Biên độ')

		plt.tight_layout()
		plt.show()

def main():
	b = BFSK()
	b.part_a()
	b.part_b()
	b.part_c()
	b.part_d()
	b.part_e()
	b.part_f()

main()