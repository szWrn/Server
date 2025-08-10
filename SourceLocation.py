import numpy as np
import pyaudio
import time
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, correlate
from threading import Thread, Event
from queue import Queue

# 根据规格表配置参数
num_mics = 4
sample_rate = 48000
chunk_size = 1024  # 每次处理的样本数
mic_space = 0.04
sound_speed = 343
max_angle = 180  # 最大拾音角度前向180度

# 麦克风位置（线型等距阵列，沿X轴排列）
mic_positions = np.array([
    [-1.5 * mic_space, 0],  # 左1
    [-0.5 * mic_space, 0],  # 左2
    [0.5 * mic_space, 0],   # 右1
    [1.5 * mic_space, 0]    # 右2
])

# 采集音频数据的类
class MicrophoneArray:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.open_stream()

    def open_stream(self):
        # device_index = self.find_usb_device()
        # if device_index is None:
        #     raise ValueError("未检测到4通道USB麦克风设备")

        self.stream = self.audio.open(
            input_device_index=device_index,
            format=pyaudio.paInt16,
            channels=num_mics,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )

    def find_usb_device(self):
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if "USB" in dev_info["name"] and dev_info["maxInputChannels"] >= num_mics:
                return i
        return None

    def read_chunk(self):
        raw_data = self.stream.read(chunk_size, exception_on_overflow=False)
        data = np.frombuffer(raw_data, dtype=np.int16)
        data = data.reshape(-1, num_mics).T
        return data.astype(np.float32) / 32768.0  # 归一化到[-1, 1]

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

# 音频处理类
class AudioProcessor:
    def __init__(self):
        self.b, self.a = self.butter_bandpass(300, 4000, sample_rate)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass(self, signals):
        filtered = np.zeros_like(signals)
        for i in range(signals.shape[0]):
            filtered[i] = lfilter(self.b, self.a, signals[i])
        return filtered

    def normalize_signal(self, signal):
        return signal / np.max(np.abs(signal))

# 基于TDOA的方向定位类
class TDOALocalizer:
    def __init__(self, mic_positions):
        self.mic_positions = mic_positions
        self.ref_index = 1  # 以左2号麦克风为参考

    def compute_tdoa(self, signals):
        ref_signal = signals[self.ref_index]
        delays = np.zeros(num_mics)

        for i in range(num_mics):
            if i == self.ref_index:
                delays[i] = 0.0
                continue

            correlation = correlate(signals[i], ref_signal, mode='full')
            lags = np.arange(-len(ref_signal) + 1, len(ref_signal))

            max_index = np.argmax(correlation)
            lag = lags[max_index]
            delays[i] = lag / sample_rate

        return delays

    def estimate_direction(self, delays):
        A = []
        b = []
        for i in range(num_mics):
            if i == self.ref_index:
                continue
            delta_x = self.mic_positions[i, 0] - self.mic_positions[self.ref_index, 0]
            A.append([delta_x])
            b.append(sound_speed * delays[i])

        A = np.array(A)
        b = np.array(b)

        sin_theta = np.linalg.lstsq(A, b, rcond=None)[0][0]
        sin_theta = np.clip(sin_theta, -1, 1)
        angle_rad = np.arcsin(sin_theta)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

# 实时方向追踪类
class RealTimeDirectionTracker:
    def __init__(self):
        self.mic_array = MicrophoneArray()
        self.processor = AudioProcessor()
        self.localizer = TDOALocalizer(mic_positions)
        self.data_queue = Queue(maxsize=10)
        self.stop_event = Event()
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        plt.ion()

    def start(self):
        self.thread = Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        try:
            while not self.stop_event.is_set():
                raw_signals = self.mic_array.read_chunk()
                filtered_signals = self.processor.apply_bandpass(raw_signals)
                normalized_signals = self.processor.normalize_signal(filtered_signals)

                delays = self.localizer.compute_tdoa(normalized_signals)
                angle = self.localizer.estimate_direction(delays)

                self.data_queue.put(angle)

        except Exception as e:
            print(f"处理错误: {e}")
        finally:
            self.mic_array.close()

    def update_display(self):
        while not self.stop_event.is_set():
            if not self.data_queue.empty():
                angle = self.data_queue.get()
                self.update_plot(angle)
            plt.pause(0.05)

    def update_plot(self, angle_deg):
        self.ax.clear()
        angle_rad = np.deg2rad(angle_deg)

        self.ax.arrow(angle_rad, 0, 0, 1, width=0.05, color='red')

        self.ax.set_thetamin(-90)
        self.ax.set_thetamax(90)
        self.ax.set_theta_zero_location('N')
        self.ax.set_title(f"声源方向: {angle_deg:.1f}°")

        plt.draw()

    def stop(self):
        self.stop_event.set()
        plt.close()

# 启动实时定位
if __name__ == "__main__":
    tracker = RealTimeDirectionTracker()
    tracker.start()
    tracker.update_display()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tracker.stop()
