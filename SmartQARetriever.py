import numpy as np
from collections import Counter
from math import log, sqrt

def computeLcsLength(firstSeq, secondSeq):
    """
    计算两个序列（字符串）的最长公共子序列长度（LCS）。
    动态规划：dpMatrix[i][j] 表示 firstSeq[i:] 与 secondSeq[j:] 的 LCS 长度。
    """
    # 序列长度
    lengthA = len(firstSeq)
    lengthB = len(secondSeq)

    # 初始化 DP 矩阵，尺寸 (lengthA+1) x (lengthB+1)，初始值全为 0
    # 额外一行一列用于处理边界情况（空串）
    dpMatrix = []
    for rowIndex in range(lengthA + 1):
        # 每一行初始化 lengthB+1 个元素为 0
        rowList = [0] * (lengthB + 1)
        dpMatrix.append(rowList)
    # 从后向前填表：i 从 lengthA-1 到 0，j 从 lengthB-1 到 0
    # 若 firstSeq[i] == secondSeq[j]，则 dp[i][j] = dp[i+1][j+1] + 1
    # 否则取 dp[i+1][j] 和 dp[i][j+1] 中的最大值
    for indexA in range(lengthA - 1, -1, -1):
        for indexB in range(lengthB - 1, -1, -1):
            if firstSeq[indexA] == secondSeq[indexB]:
                # 字符匹配，公共长度加 1
                dpMatrix[indexA][indexB] = dpMatrix[indexA + 1][indexB + 1] + 1
            else:
                # 字符不匹配，选择跳过 firstSeq 或跳过 secondSeq
                downValue = dpMatrix[indexA + 1][indexB]
                rightValue = dpMatrix[indexA][indexB + 1]
                dpMatrix[indexA][indexB] = downValue if downValue > rightValue else rightValue
    # 最终 dpMatrix[0][0] 即为整个序列的 LCS 长度
    return dpMatrix[0][0]
def computeEditDistance(firstSeq, secondSeq):
    """
    计算两个序列（字符串）的编辑距离（Levenshtein Distance）。
    允许操作：删除、插入、替换（替换成本为 1）。
    dpMatrix[i][j] 表示将 firstSeq[:i] 转换为 secondSeq[:j] 的最小操作次数。
    """
    # 序列长度
    lengthA = len(firstSeq)
    lengthB = len(secondSeq)
    # 初始化 DP 矩阵，尺寸 (lengthA+1) x (lengthB+1)
    dpMatrix = []
    for rowIndex in range(lengthA + 1):
        rowList = [0] * (lengthB + 1)
        dpMatrix.append(rowList)
    # 边界条件：
    # 将长度为 i 的序列转换为空序列，需要 i 次删除
    for rowIndex in range(lengthA + 1):
        dpMatrix[rowIndex][0] = rowIndex
    # 将空序列转换为长度为 j 的序列，需要 j 次插入
    for colIndex in range(lengthB + 1):
        dpMatrix[0][colIndex] = colIndex
    # 主循环：填充 i 从 1 到 lengthA，j 从 1 到 lengthB
    for rowIndex in range(1, lengthA + 1):
        for colIndex in range(1, lengthB + 1):
            # 判断当前字符是否相同
            if firstSeq[rowIndex - 1] == secondSeq[colIndex - 1]:
                costReplace = 0
            else:
                costReplace = 1
            # 删除操作：从 firstSeq[:i-1] 转换到 secondSeq[:j]，再删除 one character
            costDelete = dpMatrix[rowIndex - 1][colIndex] + 1
            # 插入操作：从 firstSeq[:i] 转换到 secondSeq[:j-1]，再插入 one character
            costInsert = dpMatrix[rowIndex][colIndex - 1] + 1
            # 替换操作：从 firstSeq[:i-1] 转换到 secondSeq[:j-1]，再替换（或不替换）
            costSubstitute = dpMatrix[rowIndex - 1][colIndex - 1] + costReplace
            # 三种操作取最小值
            dpMatrix[rowIndex][colIndex] = min(costDelete, costInsert, costSubstitute)
    # dpMatrix[lengthA][lengthB] 即为所求编辑距离
    return dpMatrix[lengthA][lengthB]


def buildTfIdfMatrix(documentList):
    """
    为一组文档构建 TF-IDF 矩阵，并对每行（文档向量）做 L2 归一化。
    返回：
      - vocabularyList: 排序后的词汇表列表
      - idfVector: 每个词的逆文档频率向量
      - tfIdfMatrix: 归一化后的 TF-IDF 矩阵（shape: 文档数 x 词汇表大小）
    """
    # 文档数
    docCount = len(documentList)
    # 1. 构建词汇表：遍历所有文档，拆词并去重
    vocabularySet = set()
    for doc in documentList:
        tokens = doc.split()
        for token in tokens:
            vocabularySet.add(token)
    # 排序以确定固定词序
    vocabularyList = sorted(vocabularySet)
    vocabSize = len(vocabularyList)
    # 2. 计算 TF 矩阵（raw counts）
    #    大小 docCount x vocabSize
    tfMatrix = np.zeros((docCount, vocabSize), dtype=float)
    for docIndex in range(docCount):
        # 统计文档中每个词的出现次数
        tokenCounts = Counter(documentList[docIndex].split())
        for termIndex in range(vocabSize):
            term = vocabularyList[termIndex]
            tfMatrix[docIndex][termIndex] = tokenCounts.get(term, 0)
    # 3. 计算 DF 向量：记录每个词出现过的文档数
    dfVector = np.zeros(vocabSize, dtype=float)
    for termIndex in range(vocabSize):
        countDocsWithTerm = 0
        for docIndex in range(docCount):
            if tfMatrix[docIndex][termIndex] > 0:
                countDocsWithTerm += 1
        dfVector[termIndex] = countDocsWithTerm
    # 4. 计算 IDF 向量：idf = log((N + 1)/(df + 1)) + 1
    idfVector = np.zeros(vocabSize, dtype=float)
    for termIndex in range(vocabSize):
        idfVector[termIndex] = log((docCount + 1) / (dfVector[termIndex] + 1)) + 1
    # 5. 计算原始 TF-IDF 矩阵（未归一化）
    tfIdfMatrix = np.zeros_like(tfMatrix)
    for docIndex in range(docCount):
        for termIndex in range(vocabSize):
            tfIdfMatrix[docIndex][termIndex] = tfMatrix[docIndex][termIndex] * idfVector[termIndex]
    # 6. 对每个文档向量做 L2 归一化：vec / ||vec||
    for docIndex in range(docCount):
        # 计算二范数
        squaredSum = 0.0
        for termIndex in range(vocabSize):
            squaredSum += tfIdfMatrix[docIndex][termIndex] ** 2
        vectorNorm = sqrt(squaredSum)
        # 防止除以0
        if vectorNorm == 0.0:
            vectorNorm = 1.0
        # 归一化
        for termIndex in range(vocabSize):
            tfIdfMatrix[docIndex][termIndex] /= vectorNorm
    return vocabularyList, idfVector, tfIdfMatrix


def computeCosineSimilarity(vectorA, vectorB):
    """
    计算两个已归一化向量的余弦相似度。
    由于向量已做 L2 归一化，cosine = dot(vectorA, vectorB)。
    """
    return float(np.dot(vectorA, vectorB))


def retrieveAnswer(userQuestion, qaPairs, methodName="tfidf"):
    """
    根据用户问题，从 qaPairs 中检索最匹配的问句及其答案。
    支持三种匹配方式：lcs、edit、tfidf。
    返回：匹配问句、对应答案、匹配得分。
    """
    # 将 QA 对拆分为问题列表和答案列表
    questionList = []
    answerList = []
    for pair in qaPairs:
        questionList.append(pair[0])
        answerList.append(pair[1])
    # 方法一：最长公共子序列（LCS），得分越大越匹配
    if methodName == "lcs":
        # 计算每个候选问题与用户问题的 LCS 长度
        scoreList = []
        for storedQuestion in questionList:
            lengthLcs = computeLcsLength(userQuestion, storedQuestion)
            scoreList.append(lengthLcs)
        # 在 scoreList 中找到最大得分及其索引
        bestIndex = 0
        bestScore = scoreList[0]
        for idx in range(1, len(scoreList)):
            if scoreList[idx] > bestScore:
                bestScore = scoreList[idx]
                bestIndex = idx
        # 返回匹配最高的问题、答案及得分
        return questionList[bestIndex], answerList[bestIndex], bestScore

    # 方法二：编辑距离（Edit Distance），得分（距离）越小越匹配
    if methodName == "edit":
        # 计算每个候选问题与用户问题的编辑距离
        scoreList = []
        for storedQuestion in questionList:
            distance = computeEditDistance(userQuestion, storedQuestion)
            scoreList.append(distance)
        # 在 scoreList 中找到最小距离及其索引
        bestIndex = 0
        bestScore = scoreList[0]
        for idx in range(1, len(scoreList)):
            if scoreList[idx] < bestScore:
                bestScore = scoreList[idx]
                bestIndex = idx
        return questionList[bestIndex], answerList[bestIndex], bestScore

    # 方法三：TF-IDF + 余弦相似度，得分越大越匹配
    if methodName == "tfidf":
        # 先针对所有候选问题构建 TF-IDF 矩阵
        vocabularyList, idfVector, tfIdfMatrix = buildTfIdfMatrix(questionList)
        # 1) 计算用户问题的 TF 向量（raw counts）
        vocabSize = len(vocabularyList)
        userTfVector = np.zeros(vocabSize, dtype=float)
        userTokenCounts = Counter(userQuestion.split())
        for termIndex in range(vocabSize):
            term = vocabularyList[termIndex]
            userTfVector[termIndex] = userTokenCounts.get(term, 0)
        # 2) 将用户向量转为 TF-IDF
        for termIndex in range(vocabSize):
            userTfVector[termIndex] *= idfVector[termIndex]
        # 3) 对用户向量做 L2 归一化
        squaredSum = 0.0
        for termIndex in range(vocabSize):
            squaredSum += userTfVector[termIndex] ** 2
        vectorNorm = sqrt(squaredSum)
        if vectorNorm > 0.0:
            for termIndex in range(vocabSize):
                userTfVector[termIndex] /= vectorNorm
        # 4) 计算候选问题向量与用户向量的余弦相似度
        similarityList = []
        numQuestions = len(questionList)
        for docIndex in range(numQuestions):
            sim = computeCosineSimilarity(tfIdfMatrix[docIndex], userTfVector)
            similarityList.append(sim)
        # 5) 找到相似度最高的位置
        bestIndex = 0
        bestScore = similarityList[0]
        for idx in range(1, len(similarityList)):
            if similarityList[idx] > bestScore:
                bestScore = similarityList[idx]
                bestIndex = idx
        return questionList[bestIndex], answerList[bestIndex], bestScore
    # 如传入未知方法，则抛出异常提示
    raise ValueError("Unknown method: " + methodName)
# ======================
# 示例 QA 数据列表
qaPairs = [
    ("什么是最长公共子序列？", "最长公共子序列（LCS）是两个序列中最长的公共子序列。"),
    ("如何计算编辑距离？", "编辑距离通过动态规划，用插入/删除/替换三种操作计步。"),
    ("TF–IDF 是什么？", "TF–IDF = TF(term frequency) × IDF(inverse document frequency)。"),
    ("怎么计算余弦相似度？", "余弦相似度 = 两向量点积除以模长乘积。"),
]


qaPairs = [
    # 基本数学
    ("勾股定理？",
     "勾股定理（Euclid’s theorem）指出：在任意直角三角形中，两条直角边长分别为 a、b，斜边长为 c，则有 a² + b² = c²。常见证明之一是：在直角三角形 ABC（∠C=90°）中，从 C 向斜边 AB 作高 CD，形成两个与 ABC 相似的小三角形。利用相似比可得到 a² + b² = c²。该定理是几何、解析几何、工程测量等领域的基础工具。"),
    ("复数几何？",
     "在复平面上，复数 z = x + yi 对应平面点 (x, y)：\n"
     "1. 模长 |z| = √(x² + y²)，表示该点到原点的距离；\n"
     "2. 辐角 θ = Arg(z) = arctan2(y, x)，表示从正实轴逆时针到 (x, y) 的角度；\n"
     "3. 极坐标形式：z = r e^{iθ}（其中 r = |z|），乘法操作 z₁z₂ = r₁r₂ e^{i(θ₁+θ₂)} 对应几何上的放缩与旋转。"),

    # 高等数学
    ("Lebesgue 与 Riemann？",
     "Riemann 积分通过给定区间的等分子区间求和，对函数的分段连续性要求较高。Lebesgue 积分基于测度论，将值域分层再测量原像，对间断点具有更好的容忍性，并拥有单调收敛定理、主导收敛定理等强大性质。任何 Riemann 可积的有界函数均 Lebesgue 可积且积分相同，但存在 Lebesgue 可积而 Riemann 不可积的函数。"),
    ("傅里叶级数？",
     "傅里叶级数将周期函数 f(x)（周期 2π）展开为\n"
     "f(x) = a₀/2 + Σ_{n=1}^∞ [a_n cos(nx) + b_n sin(nx)],\n"
     "系数由正交性决定：\n"
     "a_n = (1/π)∫_{−π}^{π} f(x) cos(nx) dx，\n"
     "b_n = (1/π)∫_{−π}^{π} f(x) sin(nx) dx。\n"
     "若 f 在分段连续且有有限个极值，则级数在连续点收敛于 f(x)，在跳跃不连续点收敛于左右极限平均值（Gibbs 现象）。傅里叶展开是信号处理、偏微分方程、振动分析的核心工具。"),

    # 线性代数
    ("对称矩阵？",
     "实对称矩阵 A（A = Aᵀ）具有如下性质：\n"
     "1. 所有特征值均为实数；\n"
     "2. 不同特征值对应的特征向量正交；\n"
     "3. 可通过正交矩阵 P 对角化：PᵀAP = D，其中 D 对角元为特征值。\n"
     "该谱分解便于计算矩阵幂、矩阵指数，以及二次型 xᵀAx 的最优化等。"),
    ("SVD 降维？",
     "奇异值分解（SVD）将任意 A ∈ ℝ^{m×n} 写作 A = U Σ Vᵀ：\n"
     "U ∈ ℝ^{m×m}、V ∈ ℝ^{n×n} 为正交矩阵，Σ ∈ ℝ^{m×n} 对角且奇异值 σ₁ ≥ σ₂ ≥ … ≥ 0 排序。\n"
     "取前 k 个最大奇异值及对应列向量，构造 A_k = U_k Σ_k V_kᵀ，为 A 的秩-k 最佳近似（Frobenius 范数意义下误差最小）。广泛用于 PCA、降噪、图像压缩与推荐系统。"),

    # 密码学
    ("对称 vs 非对称？",
     "对称加密（如 AES）使用同一密钥进行加解密，速度快、适合大数据量，但密钥分发和管理是难点。非对称加密（如 RSA、ECC）使用公私钥对：公钥加密或验证签名，私钥解密或生成签名，解决了密钥分发和身份认证问题，但运算速度较慢，通常用于对称密钥交换和数字签名。"),
    ("常见哈希？",
     "主要哈希函数包括：\n"
     "MD5（128 位，已不安全）、\n"
     "SHA-1（160 位，存在碰撞攻击风险）、\n"
     "SHA-2 系列（SHA-256/384/512，广泛使用）、\n"
     "SHA-3（基于 Keccak 海绵结构）。\n"
     "哈希用于消息摘要、完整性校验、数字签名前置、HMAC、区块链数据链接、去重存储等场景。"),
    ("数字签名？",
     "数字签名流程：\n"
     "1. 发送方对消息 M 计算哈希 h = H(M)；\n"
     "2. 使用私钥 sk 对 h 签名，生成签名 s；\n"
     "3. 接收方用发送方公钥 pk 验证 s，得出哈希 h'；\n"
     "4. 重新计算 H(M) 与 h' 比对相同，则签名有效。\n"
     "数字签名提供消息完整性、身份认证和不可否认性。"),
    ("零知识证明？",
     "零知识证明允许证明者在不泄露秘密的前提下，说服验证者某命题为真。以 Schnorr 协议为例：\n"
     "1. Prover 随机选 k，计算 R = g^k 幡 y = g^x，发送 R；\n"
     "2. Verifier 随机挑战 e；\n"
     "3. Prover 计算 s = k + e·x mod q，返回 s；\n"
     "4. Verifier 检查 g^s ?= R · y^e 。\n"
     "若成立，则证明者确实掌握离散对数 x。"),

    # 医学
    ("心脏传导？",
     "心脏电生理传导系统：\n"
     "1. 窦房结（SA node）自发放电启动心搏；\n"
     "2. 信号通过房房传导束传播到房室结（AV node），在此微弱延迟以保证房先收缩；\n"
     "3. 经希氏束分为左右束支，再沿浦肯野纤维网传导至心室肌，触发收缩。\n"
     "该顺序保证心房先于心室收缩，有序推动血液循环。"),
    ("耐药机制？",
     "细菌抗药机理包括：\n"
     "1. 产生酶（如 β-内酰胺酶）降解或修饰抗生素；\n"
     "2. 靶标改变：通过突变或甲基化降低药物结合亲和力；\n"
     "3. 药物外排泵（Efflux pump）主动将抗生素泵出；\n"
     "4. 通透性改变：改变膜蛋白或脂多糖减少药物进入；\n"
     "5. 生物膜（Biofilm）形成，阻碍药物渗透并减慢代谢。"),
    ("mRNA 疫苗？",
     "mRNA 疫苗原理：\n"
     "1. 将编码病原体关键抗原（如病毒刺突蛋白）的 mRNA 包装在脂质纳米颗粒中；\n"
     "2. 进入机体后细胞摄取并利用自身翻译机制合成抗原；\n"
     "3. 抗原由细胞表面展示或分泌，激活先天免疫（TLR 识别）和适应性免疫（B 细胞产生中和抗体、T 细胞介导细胞免疫）。\n"
     "优势：生产周期短、易于扩产、无需活病毒、易于针对新变种快速调整。"),
    ("干细胞疗法？",
     "干细胞疗法应用：\n"
     "1. 多能干细胞（ESC、iPSC）可定向分化为心肌、神经、胰岛细胞等，用于心梗修复、帕金森病、糖尿病等；\n"
     "2. 成体干细胞（如 MSC）通过分泌生长因子和免疫调节促进组织再生，用于骨关节炎、肝纤维化、慢性创面等；\n"
     "3. 挑战：免疫排斥、致瘤风险、分化控制、体内动态监测及伦理法规问题。"),

    # 比特币及区块链
    ("双花攻击？",
     "双花攻击指同一 UTXO 被多次消费。区块链防范机制：\n"
     "1. 全网节点验证交易，已消费 UTXO 无法再次使用；\n"
     "2. 工作量证明（PoW）需攻击者重建被攻击区块及其后续区块并拥有 >50% 算力才可能重写链；\n"
     "3. 建议等待至少 6 个区块确认（约 1 小时）以使交易难以被回滚。"),
    ("隔离见证？",
     "SegWit 将交易签名（witness）数据移至区块扩展区域，带来三大改进：\n"
     "1. 增加每块可承载交易数量，提升吞吐；\n"
     "2. 修复交易延展性，签名不再影响交易 ID；\n"
     "3. 支持 Bech32 新地址格式，节省空间并降低手续费。"),
    ("Taproot？",
     "Taproot 于 2021 年激活，主要特性：\n"
     "1. Schnorr 签名：签名可聚合，提升多签隐私与效率；\n"
     "2. MAST（Merkelized Script Trees）：仅揭示实际使用的脚本分支，其余条件以哈希形式隐藏，增强智能合约隐私；\n"
     "3. 向后兼容：兼顾旧版地址与新功能，简化复杂支付脚本并降低链上数据量。"),
    ("DeFi on BTC？",
     "比特币 DeFi 主要通过侧链与跨链协议实现：\n"
     "1. RSK 智能合约侧链：双向锚定 BTC 提供借贷、AMM 等功能；\n"
     "2. Stacks：基于 PoX（Proof-of-Transfer）将 BTC 锁定发行 STX 并支持智能合约；\n"
     "3. Lightning Network：除支付外，可做去中心化交易、自动化做市和微支付，用于 DeFi 场景。"),
    ("比特币未来？",
     "比特币未来趋势：\n"
     "1. 二层网络（闪电网络、Liquid 等）扩容与互操作性提升；\n"
     "2. 隐私增强（Taproot、MimbleWimble、CoinSwap 等）进一步保护交易匿名性；\n"
     "3. 矿业能效与可再生能源使用率提高，应对环保压力；\n"
     "4. 监管与合规框架完善——托管服务、ETF、合规钱包推动机构和普通用户接受；\n"
     "5. 跨链互操作与资产原子交换促进多链生态融合。"),
]



# 交互式演示主循环
while True:
    print("可选匹配算法：lcs / edit / tfidf")
    userMethod = input("请选择算法（输入其它内容退出）：").strip()
    # 非法输入退出程序
    if userMethod not in ("lcs", "edit", "tfidf"):
        print("检测到非法输入，程序终止。")
        break
    userQuestion = input("请输入您的问题（输入 exit 退出）：").strip()
    # 用户显式退出
    if userQuestion.lower() == "exit":
        print("程序已退出。")
        break
    # 调用检索函数，获得匹配结果
    matchedQuestion, matchedAnswer, matchScore = retrieveAnswer(userQuestion, qaPairs, userMethod)
    # 打印结果
    print("匹配到的问题：", matchedQuestion)
    print("对应的答案：", matchedAnswer)
    print("匹配得分：", matchScore)
    print("-" * 50)   # 分割线，美观输出
