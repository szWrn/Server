import librosa
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_data():
    # loading音频数据
    wrn, sr1 = librosa.load("D:\音色数据\wrn.mp3")
    lyl, sr2 = librosa.load("D:\音色数据\lyl.mp3")

    # 提取特征
    wrn_mfcc = librosa.feature.mfcc(y=wrn, sr=sr1)
    lyl_mfcc = librosa.feature.mfcc(y=lyl, sr=sr2)

    # 特征数据集
    X = np.concatenate((wrn_mfcc.T, lyl_mfcc.T), axis=0)

    # 生成向量
    y = np.concatenate((np.zeros(len(wrn_mfcc.T)), np.ones(len(lyl_mfcc.T))))

    return X, y


X, y = load_data()


def train(X, y):
    # 数据集分成训练和测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 逻辑回归分类
    model = LogisticRegression(multi_class="ovr")

    # 训练
    model.fit(X_train, y_train)

    return model


model = train(X, y)


def predict(model, audio_file):
    # 加载、提取特征
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # 分类预测
    label = model.predict(mfcc.T)
    proba = model.predict_proba(mfcc.T)

    # 概率最大的标签
    max_prob_idx = np.argmax(proba[0])
    max_prob_label = label[max_prob_idx]

    return max_prob_label


# 测试模型
label = predict(model, "")
print("识别音色为：", label)