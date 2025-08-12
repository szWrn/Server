import dashscope
import requests

test_qa_pairs = {
    # 餐厅场景（背景噪音：餐具声、人群交谈）
    "配菜有哪些选择？": 
        "您的牛排要几成熟？... 配菜选土豆泥还是沙拉？... 饮料需要现在上吗？",
    
    # 地铁问路场景（背景噪音：广播报站、列车行驶声）
    "乘客想换乘哪条线路？":
        "请问换乘3号线怎么走？... 工作人员：B出口电梯上楼... 注意末班车是11点半！",
    
    # 医院挂号场景（背景噪音：叫号系统、咳嗽声）
    "耳鼻喉科在几楼？":
        "医保卡带了吗？... 挂哪个科室？... 耳鼻喉科在3楼西侧！",
    
    # 超市购物场景（背景噪音：促销广播、推车声）
    "鲜奶有什么优惠？":
        "鲜奶买二送一... 找零请收好... 需要购物袋吗？加1元...",
    
    # 电话客服场景（背景噪音：电流杂音）
    "客服提到的订单尾号是多少？":
        "订单号尾号3472对吗？... 问题已记录... 3个工作日内会回复您...",
    
    # 公园偶遇场景（背景噪音：儿童嬉闹、鸟叫）
    "复诊时间是周几？":
        "听说你做了手术... 恢复得怎样？... 周三有复诊吗？...",
    
    # 银行办理场景（背景噪音：点钞机声）
    "年利率是多少？":
        "请在这里签名... 身份证复印件给我... 年利率是2.75%...",
    
    # 紧急广播场景（背景噪音：警报声）
    "集合点在哪里？":
        "广播：全体人员注意... 请走安全通道... 勿乘电梯... 集合点在广场南侧..."
}

# Q:Ethan
# A:Cherry

def gen(text, path, color):
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        # 仅支持qwen-tts系列模型，请勿使用除此之外的其他模型
        model="qwen-tts",
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx"
        api_key="sk-00925f3e562e418e946103804bfcf2ca",
        text=text,
        voice=color,
    )
    rp = requests.get(response["output"]["audio"]["url"])
    with open(path, "wb") as f:
        f.write(rp.content)

i = 0
for k in list(test_qa_pairs.keys()):
    gen("提问: " + k, f"audio/k{i}.wav", "Cherry")
    i += 1

i = 0
for k in list(test_qa_pairs.values()):
    gen(k, f"audio/v{i}.wav", "Ethan")
    i += 1