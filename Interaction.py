import os
import signal
import sys
import time
import json
import dashscope
import pyaudio
import numpy as np
import noisereduce as nr
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from openai import OpenAI

# 通义千问API密钥
DASHSCOPE_API_KEY = "sk-00925f3e562e418e946103804bfcf2ca"
# OpenAI兼容客户端（用于对话分析）
client = OpenAI(
    api_key="sk-bcc8cec0828648a18876556009620126",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 录音参数
sample_rate = 16000
# 修改为1通道（单声道），因为您的设备不支持4通道
channels = 1
format_pcm = 'pcm'
block_size = 3200

# 定向识别参数 - 由于单声道，定向功能无法使用
# target_direction = 60  # 目标拾音方向 (0-360度)

# 降噪参数
noise_threshold = 0.1  # 噪声阈值 (0-1)
noise_profile = None   # 存储噪声样本
noise_profile_collected = False  # 噪声样本是否已采集

# 流式输出控制
stream_output = True  # 启用实时流式输出

# 全局变量
mic = None
stream = None
recognition = None  # ASR识别服务实例
dialog_queue = []  # 对话历史队列（存储识别到的句子）




def init_dashscope_api_key():
    """初始化通义千问API密钥"""
    dashscope.api_key = DASHSCOPE_API_KEY

system_prompt = """
你是一个实时对话分析AI，用于判断对话是否答非所问。请严格遵守以下规则：
1. 输入是连续对话历史，仅分析最后两条消息（最近一次问答）。
2. 输出规则：
   - 若回答与问题明显无关，或对方表示没听清（如“啊？”“再说一遍”），输出`No`；
   - 其他情况（回答合理、澄清问题），输出`Yes`；
   - 仅输出`Yes`或`No`，禁止其他内容。
3. 对话不完整时（只有问题无回答），默认输出`Yes`。
"""

def apply_noise_reduction(audio_data):
    """应用降噪处理"""
    global noise_profile

    # 如果尚未采集噪声样本，使用默认降噪
    if noise_profile is None:
        return nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=True)

    # 使用采集的噪声样本进行降噪
    return nr.reduce_noise(
        y=audio_data,
        y_noise=noise_profile,
        sr=sample_rate,
        stationary=False
    )

def check_dialog(dialog_history):
    """调用大模型分析对话是否答非所问"""
    if not dialog_history:
        return "No"  # 空对话默认异常

    # 构造对话消息
    messages = [{"role": "system", "content": system_prompt}]
    for i, content in enumerate(dialog_history):
        # 交替标记为"user"和"assistant"（模拟问答角色）
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": content})

    # 调用大模型分析
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            stream=False
        )
        result = response.choices[0].message.content.strip()
        print(f"\n【对话分析结果】: {result}")
        return result
    except Exception as e:
        print(f"对话分析出错: {e}")
        return "No"

class ASRCallback(RecognitionCallback):
    """实时语音识别回调：处理录音、识别文本，并触发对话分析"""
    def __init__(self):
        super().__init__()
        self.current_sentence = ""  # 临时存储当前识别的句子
        self.last_update_time = time.time()  # 最后更新时间
        self.stream_buffer = []  # 流式输出缓冲区

    def on_open(self) -> None:
        """打开麦克风，开始录音"""
        global mic, stream
        print("→ 开始录音（请说话）...")
        mic = pyaudio.PyAudio()

        # 列出可用音频设备
        print("可用音频输入设备:")
        for i in range(mic.get_device_count()):
            dev_info = mic.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels', 0) > 0:
                print(f"  [{i}] {dev_info['name']} - 输入通道: {dev_info['maxInputChannels']}")

        # 尝试打开默认设备
        try:
            stream = mic.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=block_size
            )
            print("→ 使用默认输入设备")
        except Exception as e:
            print(f"→ 默认设备打开失败: {e}")
            # 尝试手动选择设备
            for i in range(mic.get_device_count()):
                dev_info = mic.get_device_info_by_index(i)
                if dev_info.get('maxInputChannels', 0) > 0:
                    try:
                        stream = mic.open(
                            format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            input_device_index=i,
                            frames_per_buffer=block_size
                        )
                        print(f"→ 成功打开设备 [{i}] {dev_info['name']}")
                        break
                    except Exception:
                        continue

        if stream is None:
            print("→ 错误: 无法打开任何音频输入设备")
            sys.exit(1)

        # 采集初始噪声样本
        print("→ 正在采集噪声样本（保持安静）...")
        noise_frames = []
        for _ in range(10):  # 采集0.5秒噪声样本
            data = stream.read(block_size, exception_on_overflow=False)
            noise_frames.append(data)

        # 处理噪声样本
        noise_data = np.frombuffer(b''.join(noise_frames), dtype=np.int16)
        global noise_profile
        noise_profile = noise_data.astype(np.float32)
        print("→ 噪声样本采集完成")

    def on_close(self) -> None:
        """关闭麦克风，释放资源"""
        global mic, stream
        print("→ 停止录音")
        if stream:
            stream.stop_stream()
            stream.close()
        if mic:
            mic.terminate()
        stream = None
        mic = None

    def on_error(self, message) -> None:
        """处理识别错误"""
        print(f"→ 识别错误: {message.message}")
        sys.exit(1)

    def on_event(self, result: RecognitionResult) -> None:
        """处理识别结果：拼接文本，句子结束时添加到对话队列并分析"""
        global dialog_queue
        sentence = result.get_sentence()

        if 'text' in sentence:
            # 更新当前识别的文本
            self.current_sentence += sentence['text']

            # 流式输出：每0.5秒或句子有更新时输出
            current_time = time.time()
            if stream_output and (current_time - self.last_update_time > 0.5 or
                                 len(self.stream_buffer) > 0 and self.stream_buffer[-1] != self.current_sentence):
                print(f"→ 流式识别: {self.current_sentence}", end="\r")
                self.last_update_time = current_time
                self.stream_buffer.append(self.current_sentence)

            # 当检测到句子结束（如包含标点、停顿）
            if RecognitionResult.is_sentence_end(sentence):
                print(f"\n→ 完整句子: {self.current_sentence}")
                # 将完整句子添加到对话队列
                dialog_queue.append(self.current_sentence)
                # 保留最近10条对话（避免队列过长）
                if len(dialog_queue) > 10:
                    dialog_queue = dialog_queue[-10:]
                # 重置当前句子缓存
                self.current_sentence = ""
                # 清空流式缓冲区
                self.stream_buffer = []

                # 当对话队列中有至少2条消息时，触发分析（判断最近一次问答）
                if len(dialog_queue) >= 2:
                    print("\n→ 正在分析对话...")
                    check_dialog(dialog_queue)

def signal_handler(sig, frame):
    """处理Ctrl+C退出信号"""
    global recognition
    print("\n→ 用户终止程序")
    if recognition:
        recognition.stop()
    sys.exit(0)

def process_audio_frame(data):
    """处理音频帧：降噪"""
    # 将二进制数据转换为numpy数组
    audio_data = np.frombuffer(data, dtype=np.int16)

    # 应用降噪处理
    denoised = apply_noise_reduction(audio_data)

    # 将处理后的音频转换为二进制
    return denoised.astype(np.int16).tobytes()


from openai import OpenAI

import json

API_KEY = "sk-bcc8cec0828648a18876556009620126"
client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

system_prompt = """
你是一个实时对话分析AI，用于帮助听障人士判断对方是否听清或理解说话内容。请严格遵守以下规则：

1. **输入**：用户提供的是一段连续对话（最后两条消息是最近一次问答）。  
2. **输出规则**：  
   - 如果回答与问题 **明显无关**（如答非所问），或对方 **明确表示没听清**（如“啊？”、“再说一遍”），输出 `No`；  
   - 其他情况（回答合理、符合问题意图），输出 `Yes`。  
   - 对方重复或澄清问题（如"你问XX吗？"），输出`Yes`
3. **严格限制**：  
   - 仅分析 **最近一次问答**（最后两条消息）。  
   - 只输出 `Yes` 或 `No`，**禁止解释、添加标点或换行**。  
4. **特殊处理**：  
   - 若对话不完整（如只有问题无回答），默认输出 `Yes`（避免误报）。  
"""


def check(q):
    s = time.time()
    m = [{
        "role": "system",
        "content": system_prompt
    }]
    for i in q:
        m.append({"role": "user", "content": i})

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=m,
        stream=False
    )

    result = response.choices[0].message.content
    print(result)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(response.model_dump_json(), f, ensure_ascii=False, indent=4)

    return result


queue = []

if __name__ == '__main__':
    init_dashscope_api_key()

    asr_callback = ASRCallback()
    recognition = Recognition(
        model='paraformer-realtime-v2',  # 实时ASR模型
        format=format_pcm,
        sample_rate=sample_rate,
        semantic_punctuation_enabled=True,
        callback=asr_callback
    )

    # 启动ASR服务
    recognition.start()

    # 注册退出信号
    signal.signal(signal.SIGINT, signal_handler)
    print("按Ctrl+C停止程序...\n")
    print(f"→ 流式输出: {'启用' if stream_output else '禁用'}")

    try:
        while True:
            if stream:
                # 读取原始音频帧
                raw_data = stream.read(block_size, exception_on_overflow=False)

                # 处理音频：降噪
                processed_data = process_audio_frame(raw_data)

                # 发送处理后的音频给ASR
                recognition.send_audio_frame(processed_data)

            time.sleep(0.01)

            a =asr_callback
            string = input(a).strip()
            if string.lower() == "exit":
                break
            queue.append(string)
            result = check(queue)
            # 降低CPU占用
    except Exception as e:
        print(f"主循环出错: {e}")
        recognition.stop()