from sentence_transformers import SentenceTransformer, util
import torch

# 1. 载入多语言Sentence-BERT模型
model_name = 'paraphrase-multilingual-mpnet-base-v2'
model = SentenceTransformer(model_name)

qa_dict = {
    "如何计算余弦相似度？": (
        """余弦相似度计算公式如下：

def cosine_similarity(vec1, vec2):
    # 计算两个向量的余弦相似度
    dot_product = torch.dot(vec1, vec2)
    norm_a = torch.norm(vec1)
    norm_b = torch.norm(vec2)
    return dot_product / (norm_a * norm_b)

def l2_distance(vec1, vec2):
    # 计算两个向量的L2距离（欧氏距离）
    diff = vec1 - vec2
    return torch.norm(diff, p=2)

余弦相似度值域为[-1,1]，表示两个向量的方向相似程度。"""
    ),

    "高血压是什么？": (
        """高血压（Hypertension）是指动脉血压持续升高的一种慢性非传染性疾病。根据最新的诊断标准，
收缩压≥140 mmHg和/或舒张压≥90 mmHg，或正在服用降压药物的患者可诊断为高血压。
高血压是心血管疾病的重要风险因素，长期血压升高可导致心脏病、脑卒中、肾衰竭等器官损害。
高血压分为原发性（约占90%-95%，病因复杂且不明）和继发性（由肾脏疾病、内分泌疾病等引起）。
多数患者早期无明显临床表现，被称为“无声杀手”，部分患者可能出现头痛、头晕、心悸等症状。
治疗包括生活方式干预（低盐饮食、减重、戒烟、运动）、药物治疗（钙通道阻滞剂、ACE抑制剂、利尿剂等），
目标在于有效控制血压，预防心脑肾等并发症。患者需定期监测血压，遵医嘱用药，保持长期管理。"""
    ),

    "糖尿病有哪些症状？": (
        """糖尿病（Diabetes Mellitus）是一组以慢性高血糖为特征的代谢性疾病，主要分为1型和2型糖尿病。
典型症状称为“多尿、多饮、多食、体重下降”，并可能伴有疲乏、视力模糊、伤口愈合缓慢等表现。
糖尿病若血糖控制不良，长期可引发心脑血管疾病、糖尿病肾病、视网膜病变及神经病变等严重并发症。
诊断依据为空腹血糖、口服葡萄糖耐量测试及糖化血红蛋白（HbA1c）数值。
管理强调生活方式调整、药物治疗与定期监测，目标实现血糖稳定，减少相关并发症风险。"""
    ),

    "心脏病的危险因素有哪些？": (
        """冠心病是全球最常见的心脏疾病，危险因素涵盖了多方面：
包括高血压、高血脂（尤其低密度脂蛋白升高）、糖尿病、吸烟、肥胖、不良饮食习惯、缺乏运动、
家族遗传及精神压力等。
综合管理涉及控制血压和血脂，戒烟限酒，控制体重，增加运动，并根据需要使用药物治疗。
早期识别和干预风险因素对预防心脏病至关重要。"""
    ),

    "如何预防感冒？": (
        """感冒主要由多种病毒引起，传播途径包括飞沫传播和接触传播。
预防措施包括坚持良好的个人卫生习惯，如勤洗手，避免用手触摸眼、口、鼻，远离感冒患者和人群密集场所，
保证室内空气流通。
增强免疫力也很关键，保持均衡饮食、充足睡眠和适度锻炼。
流感高发季节建议接种流感疫苗，以减少感染风险及严重并发症。"""
    ),

    "哮喘的常见诱因有哪些？": (
        """哮喘是一种慢性气道炎症病症，表现为反复发作的喘息、胸闷、咳嗽和呼吸困难。
诱因常见的有：过敏原（如尘螨、花粉、宠物皮屑）、呼吸道感染（尤其是病毒感染）、冷空气及空气污染。
运动、某些药物（如阿司匹林）、情绪波动和职业暴露等也可诱发哮喘。
有效控制哮喘需长期遵医嘱使用吸入性糖皮质激素及支气管扩张剂，并避免接触诱因。"""
    ),

    "冠心病的主要症状有哪些？": (
        """冠状动脉粥样硬化性心脏病主要症状为心绞痛，即胸骨后压榨样或紧缩性疼痛，常放射至左肩、左臂、颈部或背部。
患者可能出现呼吸困难、心悸、恶心等伴随症状。
部分患者存在无症状缺血。
急性冠脉事件，如心肌梗死时，疼痛持续时间长且剧烈，伴随出汗、意识障碍。
控制危险因素和科学治疗对缓解症状和预防心肌损伤至关重要。"""
    ),

    "脑卒中（中风）有哪些急救措施？": (
        """脑卒中包括缺血性和出血性，两者均为急危重症。
临床表现为偏瘫、语言障碍、意识障碍、面部歪斜等急性神经功能缺失。
患者出现这些症状时，应立即拨打急救电话，避免自行移动患者，保持呼吸道通畅并观察生命体征。
早期送医，争取窗口期内给予溶栓或介入治疗是关键，能有效改善预后。"""
    ),

    "骨质疏松症如何预防？": (
        """骨质疏松症是骨量减少及骨组织微结构破坏导致骨脆性增加的系统性骨病，易引发骨折。
预防措施包括充足的钙和维生素D摄入（每日钙1000-1200mg，维生素D800-1000IU）、规律的负重及抗阻运动、戒烟限酒。
老年人应特别注意防跌倒措施，并定期进行骨密度检查。
对于高危患者，医生可考虑药物治疗以降低骨折风险。"""
    ),

    "什么是VeraCrypt？": (
        """VeraCrypt是一个免费的开源磁盘加密软件，是著名TrueCrypt项目的继任者，在其基础上加强了安全性，修复了多个已知漏洞。
它允许用户创建加密的虚拟磁盘（加密卷），也能对整个分区或物理磁盘进行实时加密，保护数据不被未授权访问。
VeraCrypt支持多种强大加密算法，包括AES、Serpent和Twofish，以及它们的组合进行多重加密，
采用XTS加密模式确保数据的机密性和完整性。
为了增强安全性，VeraCrypt支持复杂的密码和长密钥（通常建议使用至少20个字符以上的强密码），
并允许结合使用额外的密钥文件（keyfiles）和PIN码作为双重认证因素，提升整体防护能力。
其独有的隐藏卷（Hidden Volume）技术能在受胁迫时保护敏感数据，使对方即使获取密码也无法发现隐藏内容。
同样，隐藏操作系统功能让用户能够创建一个看似普通但实际被加密保护的操作系统环境，防止被迫泄露真实数据。
VeraCrypt跨平台支持Windows、macOS和Linux，适用于个人用户及企业级数据安全保护场景，
并且活跃社区持续更新，保障软件安全和使用体验。"""
    ),

    "什么是PGP（Pretty Good Privacy）？": (
        """PGP是一种广泛使用的加密和数字签名协议，主要用于电子邮件和文件的加密保护。
PGP结合了非对称加密和对称加密，使用公钥/私钥体系来加密对称密钥，
利用非对称加密保证密钥分发安全，同时对称加密提高数据传输效率。
此外，PGP还能实现数字签名，保证数据完整性和身份认证。
常用实现有GNU Privacy Guard (GPG)，兼容PGP标准。"""
    ),

    "Tor是什么？": (
        """Tor（The Onion Router）是一种匿名通信网络，通过多层加密和多节点中继实现流量混淆和隐藏用户真实IP。
数据在传输过程中被多次加密，经过3个以上的随机选取的中继节点，每个节点仅能解密到下一跳节点地址，
无法获知完整路径，有效防止流量分析和追踪。
Tor广泛用于保护用户隐私，规避网络审查，以及匿名访问互联网资源。"""
    ),

    "什么是Signal？": (
        """Signal是一款由非营利组织Signal基金会（Signal Foundation）开发的开源端到端加密通讯应用，支持文本消息、语音通话、视频通话及多媒体文件传输。
        它以高标准的隐私保护和安全防护著称，核心依赖业界领先的Signal协议（Signal Protocol），该协议融合了多种先进密码学技术，为用户通讯提供强大的加密保障。
Signal协议由Open Whisper Systems设计，结合了多种关键算法。首先，其核心是“双重Ratchet算法”（Double Ratchet Algorithm），该算法结合了椭圆曲线Diffie-Hellman密钥交换与对称密钥Ratchet机制。
它通过每发送一条消息都生成新的会话密钥，保证了前向安全（Forward Secrecy）和后向安全（Future Secrecy）。这意味着即使某次密钥被泄露，之前或之后的消息内容仍然无法被破解。
在会话初始化阶段，Signal采用“扩展三重Diffie-Hellman”协议（X3DH，Extended Triple Diffie-Hellman）实现安全的密钥协商。
X3DH综合利用长期身份密钥（Identity Key）、一次性预发布密钥（One-Time Prekey）和短期临时密钥，抵御中间人攻击和重放攻击，确保双方能安全地建立共享密钥。
Signal协议中使用Curve25519椭圆曲线算法进行Diffie-Hellman密钥交换，具有高效且安全的特点。
消息加密则采用AES-256-GCM认证加密方式（Authenticated Encryption with Associated Data，AEAD），确保消息的机密性和完整性。消息认证和密钥派生则使用HMAC-SHA256算法进行。
在具体实现方面，Signal官方提供跨平台的加密库，如libsignal-protocol-c和libsignal-protocol-java，供客户端调用以实现信封加密、密钥管理和消息解密等功能。
Signal应用自身还借助libsodium库（一套基于NaCl的密码学库）实现安全的随机数生成和加密操作。数据传输中，Signal利用Google的Protocol Buffers（protobuf）进行高效的消息序列化与反序列化。
除了加密传输外，Signal对隐私保护做了大量优化。它通过“Sealed Sender”技术掩盖发送方身份，使服务器无法得知消息发送者是谁。
Signal服务器只保存极少用户数据——主要是注册的电话号码和一些通信路由信息，不存储消息内容和大部分元数据。此外，Signal支持多设备同步，所有设备间消息均经过端到端加密，保证历史消息不会泄漏。
应用提供“自毁消息”功能，用户可设置消息在指定时间后自动删除，防止信息长久存储和泄漏。
对于群组聊天，Signal采用“群组状态同步”协议（Group State Protocol）。
该协议结合了多条双重Ratchet会话，确保群组中每个成员的端到端加密安全，且消息仅对群组成员可见。
Signal还支持安全的密钥验证手段，如安全数字指纹比对及二维码扫描，防止中间人攻击。
Signal的开源代码库对外公开，任何人都可审计算法实现和安全性，进一步增强了透明度和用户信任。
Signal的加密协议已被众多主流通讯应用采用，比如Facebook旗下WhatsApp的全部端对端加密通信，Facebook Messenger的“秘密对话”，以及早期的Google Allo等。
官方主页：https://signal.org
GitHub开源代码库：https://github.com/signalapp
总结来说，Signal通过结合双重Ratchet算法、X3DH密钥协商、Curve25519椭圆曲线、AES-256-GCM加密和HMAC-SHA256消息认证等密码学技术，
配合元数据最小化和隐私增强机制，构建了业界领先的安全通信生态，适用于普通用户、记者、活动家等对隐私保护要求极高的使用场景。
"""
    ),

    "常见的加密算法与加密模式有哪些？": (
        """对称加密算法：AES（高级加密标准）是目前最广泛使用的，它支持128、192、256位密钥，
通常配合块加密模式如CBC、GCM使用，提供数据机密性和完整性。
其他算法有ChaCha20，适合性能受限环境。
非对称加密算法：RSA、椭圆曲线密码学(ECC)，用于密钥交换、数字签名。
混合加密结合两者优势，用非对称加密保护对称密钥，提升效率和安全。"""
    ),

    "什么是端到端加密（E2EE）？": (
        """端到端加密指通信双方在设备端将信息加密，仅接收方能解密，
整个传输过程和服务端均无法访问明文，实现通信内容完全保密。
常用于即时通讯、文件传输，保护用户隐私，抵抗中间人攻击和服务商监听。"""
    ),

    "主流加密库有哪些？": (
        """OpenSSL：功能强大的开源密码学库，支持TLS/SSL协议和多种加密算法，广泛用于网络安全。
libsodium：易用性优先的现代加密库，提供安全的对称加密、数字签名、密钥交换。
GnuPG (GPG)：PGP兼容软件，用于数据和通信加密及数字签名。
Bouncy Castle：Java和C#环境的加密库，支持丰富的密码学算法和协议。
Crypto++：C++加密库，涵盖众多算法和工具，支持跨平台。"""
    ),

    "如何评估加密软件安全性？": (
        """加密软件安全性评估包含以下方面：
1. 加密算法和协议是否经过学术及工业界验证，避免使用弱算法；
2. 密钥管理是否安全，有没有暴露风险；
3. 软件是否开源并接受第三方安全审计和漏洞响应；
4. 实现中是否避免侧信道攻击和常见编程漏洞；
5. 用户身份认证与访问控制机制是否完善；
6. 及时更新和修复安全漏洞。"""
    ),
}
questions = list(qa_dict.keys())
question_embeddings = model.encode(questions, convert_to_tensor=True)
def answer_question(user_question, top_k=1):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    k = min(top_k, len(questions))
    top_k_scores, top_k_indices = torch.topk(cosine_scores, k=k, largest=True)
    results = []
    i = 0
    while i < k:
        idx = top_k_indices[i].item()
        score = top_k_scores[i].item()
        matched_question = questions[idx]
        matched_answer = qa_dict[matched_question]
        results.append("匹配问题: {}\n答案:\n{}\n余弦相似度: {:.4f}".format(matched_question, matched_answer, score))
        i += 1
    if len(results) == 0:
        return "抱歉，未能找到匹配的答案。请尝试换个问法。"
    else:
        return "\n\n".join(results)
print("欢迎使用多语言问答系统，输入exit退出。")
while True:
    query = input("请输入你的问题：").strip()
    if query == "":
        print("输入不能为空，请重新输入。")
        continue
    if query.lower() == "exit":
        print("退出问答系统。")
        break

    response = answer_question(query, top_k=1)
    print(response)
