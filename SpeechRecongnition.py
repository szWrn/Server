# For prerequisites running the following sample, visit https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen
import os
import signal  # for keyboard events handling (press "Ctrl+C" to terminate recording and translation)
import sys

import dashscope
import pyaudio
from dashscope.audio.asr import *

mic = None
stream = None

# Set recording parameters
sample_rate = 16000  # sampling rate (Hz)
channels = 1  # mono channel
dtype = 'int16'  # data type
format_pcm = 'pcm'  # the format of the audio data
block_size = 3200  # number of frames per buffer


class SpeechRecognition:
    def __init__(self, callback):
            # Create the translation callback
            self.callback = callback

            # Call self.recognition service by async mode, you can customize the self.recognition parameters, like model, format,
            # sample_rate For more information, please refer to https://help.aliyun.com/document_detail/2712536.html
            self.recognition = Recognition(
                model='paraformer-realtime-v2',
                # 'paraformer-realtime-v1'、'paraformer-realtime-8k-v1'
                format=format_pcm,
                # 'pcm'、'wav'、'opus'、'speex'、'aac'、'amr', you can check the supported formats in the document
                sample_rate=sample_rate,
                # support 8000, 16000
                semantic_punctuation_enabled=False,
                callback=self.callback)
            
    def init_dashscope_api_key(self):
        """
            Set your DashScope API-key. More information:
            https://github.com/aliyun/alibabacloud-bailian-speech-demo/blob/master/PREREQUISITES.md
        """

        if 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ[
                'DASHSCOPE_API_KEY']  # load API-key from environment variable DASHSCOPE_API_KEY
        else:
            dashscope.api_key = 'sk-00925f3e562e418e946103804bfcf2ca'  # set API-key manually

    def signal_handler(self, sig, frame):
        print('Ctrl+C pressed, stop translation ...')
        # Stop translation
        self.recognition.stop()
        print('Translation stopped.')
        print(
            '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
            .format(
                self.recognition.get_last_request_id(),
                self.recognition.get_first_package_delay(),
                self.recognition.get_last_package_delay(),
            ))
        # Forcefully exit the program
        sys.exit(0)

    def start(self):
        self.init_dashscope_api_key()
        print('Initializing ...')

        # Start translation
        self.recognition.start()

        signal.signal(signal.SIGINT, self.signal_handler)
        print("Press 'Ctrl+C' to stop recording and translation...")
        # Create a keyboard listener until "Ctrl+C" is pressed

        while True:
            if stream:
                data = stream.read(3200, exception_on_overflow=False)
                self.recognition.send_audio_frame(data)
            else:
                break

        self.recognition.stop()


# Real-time speech self.recognition callback
class Callback(RecognitionCallback):
    def on_open(self) -> None:
        global mic
        global stream
        print('self.recognitionCallback open.')
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True)

    def on_close(self) -> None:
        global mic
        global stream
        print('self.recognitionCallback close.')
        stream.stop_stream()
        stream.close()
        mic.terminate()
        stream = None
        mic = None

    def on_complete(self) -> None:
        print('self.recognitionCallback completed.')  # translation completed

    def on_error(self, message) -> None:
        print('self.recognitionCallback task_id: ', message.request_id)
        print('self.recognitionCallback error: ', message.message)
        # Stop and close the audio stream if it is running
        if 'stream' in globals() and stream.active:
            stream.stop()
            stream.close()
        # Forcefully exit the program
        sys.exit(1)

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if 'text' in sentence:
            print('self.recognitionCallback text: ', sentence['text'])
            if RecognitionResult.is_sentence_end(sentence):
                print(
                    'self.recognitionCallback sentence end, request_id:%s, usage:%s'
                    % (result.get_request_id(), result.get_usage(sentence)))


if __name__ == "__main__":
    sr = SpeechRecognition(Callback())
    sr.start()