import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

qa_dict = [
    {
        "question": "维生素C在护肤中的主要功效有哪些？",
        "answer": """\
维生素C（抗坏血酸）是一种强效的水溶性抗氧化剂，常见功效包括：
  - 抗氧化、抗自由基：中和紫外线和环境污染导致的氧化损伤；
  - 抑制黑色素生成：通过抑制酪氨酸酶活性，淡化色斑、提亮肤色；
  - 促进胶原蛋白合成：提高皮肤弹性和紧致度，减少细纹；
  - 促进伤口修复：加速皮肤屏障修复和创面愈合。

使用建议：
  - 浓度一般在5%–20%之间，敏感肌建议从低浓度（5%–10%）开始；
  - pH值约为2.5–3.5，确保分子稳定性和吸收率；
  - 建议白天使用并务必搭配防晒，避免氧化分解后降低效果。"""
    },
    {
        "question": "烟酰胺（维生素B3）有什么护肤作用？",
        "answer": """\
烟酰胺是一种水溶性维生素，护肤中的主要功效：
  - 修复屏障：促进神经酰胺生成，增强角质层锁水能力；
  - 均匀肤色：抑制黑色素向角质层迁移，淡化色斑和痘印；
  - 保湿锁水：减少经皮水分流失（TEWL），增加肌肤含水量；
  - 抗炎抗痘：抑制皮脂分泌，舒缓痘痘红肿；
  - 改善细纹：刺激胶原蛋白和弹性蛋白合成，减少初老迹象。

使用贴士：
  - 可与大多数成分共用，如维生素C、A醇、果酸；
  - 浓度5%–10%效果显著又温和，敏感肌可从2%–3%浓度起步；
  - 如出现轻微刺痛，可分次或隔天使用，观察肌肤耐受性。"""
    },
    {
        "question": "Skinceuticals CE Ferulic 有什么特点？",
        "answer": """\
Skinceuticals CE Ferulic 是一款经典“黄金三角”抗氧化精华，配方与功效：
  - 成分组合：15% L-抗坏血酸（维C）、1% α-生育酚（维E）、0.5% 阿魏酸；
  - 协同增效：阿魏酸能稳定维C和维E，使三者抗氧化能力提升约8倍；
  - 功效：中和自由基、抑制光老化、提亮肤色、促进胶原蛋白生成；
  - 质地：略为油润，适合正常至干性肌；油性肌可搭配轻薄保湿乳液或凝胶。

使用方法：
  - 清洁、爽肤后，取4–5滴涂抹于面部及颈部，轻拍至吸收；
  - 搭配日常防晒（SPF30+），存放于阴凉避光处，防止氧化。"""
    },
    {
        "question": "露得清（Neutrogena）的维生素C产品适合哪些肤质？",
        "answer": """\
Neutrogena 的维C系列多采用稳定的维C衍生物（如乙基抗坏血酸），特点：
  - 刺激性低：比纯维C更温和，敏感肌或初尝试者友好；
  - 质地清爽：多为轻薄乳液或啫喱，易吸收、不黏腻，适合混合偏油及油性肌；
  - 价格亲民：性价比较高，适合作日常维稳和提亮肤色使用。

使用建议：
  - 可早晚使用，干性肌可后接滋润面霜，油性肌则可单用或搭配控油精华；
  - 如出现轻微刺痛感，可降低使用频率或与保湿产品同用，帮助缓解不适。"""
    },
    {
        "question": "什么是视黄醇（A醇）？如何在护肤中使用？",
        "answer": """\
视黄醇（Retinol）属于维A醇家族，需在皮肤中转化为视黄酸（Tretinoin）才能发挥作用。主要功效：
  - 促进角质更新、疏通毛孔，减少痘痘和暗沉；
  - 刺激胶原蛋白及弹性纤维生成，淡化细纹和皱纹；
  - 均匀肤色，改善色素沉淀。

使用要点：
  - 浓度：0.1%–1%可选，新手建议从0.1%或更低浓度开始；
  - 频率：初用阶段每周2–3次，待耐受后逐步过渡到每晚使用；
  - 搭配：避免与高浓度果酸、其他视黄酸类同晚使用；可先涂面霜再涂A醇以减轻刺激；
  - 防晒：A醇会增加皮肤光敏感性，白天务必使用SPF30以上防晒并定时补涂。

常见副作用：
  - 脱屑、发红、刺痛：为正常耐受反应，通常在2–4周内改善；
  - 缓解方法：可与面霜同用或在皮肤完全干燥后再涂抹，减少对皮肤的直接刺激。"""
    },
    {
        "question": "如何在日常护肤流程中合理搭配维生素C、烟酰胺和A醇？",
        "answer": """\
为了避免成分冲突并最大化效果，推荐以下流程：

早晨：
  1. 清洁  
  2. 爽肤（可选）  
  3. 维生素C精华（抗氧化、提亮）  
  4. 烟酰胺精华或乳液（修复屏障、控油）  
  5. 保湿面霜  
  6. 防晒（SPF30+，每2小时补涂一次）

晚上：
  1. 清洁  
  2. 爽肤  
  3. A醇（视黄醇）精华/乳液（抗老、促进更新）  
     - 若易刺激，可先涂保湿面霜再涂A醇，或隔夜使用  
  4. 烟酰胺（或高保湿修复）产品  
  5. 滋润面霜

注意事项：
  - 若肌肤敏感，可将维C和烟酰胺分时段使用（维C早上、烟酰胺晚上）  
  - A醇不宜与高浓度果酸、A酸同晚使用，以免叠加刺激  
  - 坚持防晒与补水，缓解潜在的成分刺激。"""
    },
]

# 加载模型
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# 计算所有问题向量，做归一化（余弦相似度搜索）
questions = [item["question"] for item in qa_dict]
q_embs = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
q_embs = q_embs.astype('float32')

# 创建faiss索引，IndexFlatIP 用内积，也是余弦相似度（因归一化了）
d = q_embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(q_embs)

print("模型和索引加载完成，可以开始提问，输入 exit 退出。")

while True:
    query = input("\n请输入你的问题: ").strip()
    if query.lower() in ("exit", "quit"):
        print("退出程序，感谢使用！")
        break
    if not query:
        continue

    q_vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    scores, idxs = index.search(q_vec, 3)  # 找相似度最高的3条

    print("\n最相似的问题及回答：")
    for score, idx in zip(scores[0], idxs[0]):
        print(f"\n相似度: {score:.4f}")
        print(f"问：{qa_dict[idx]['question']}")
        print(f"答：{qa_dict[idx]['answer']}")






qa_dict = [
    {
        "question": "维生素C在护肤中的主要功效有哪些？",
        "answer": """\
维生素C（抗坏血酸）是一种强效的水溶性抗氧化剂，常见功效包括：
  - 抗氧化、抗自由基：中和紫外线和环境污染导致的氧化损伤；
  - 抑制黑色素生成：通过抑制酪氨酸酶活性，淡化色斑、提亮肤色；
  - 促进胶原蛋白合成：提高皮肤弹性和紧致度，减少细纹；
  - 促进伤口修复：加速皮肤屏障修复和创面愈合。

使用建议：
  - 浓度一般在5%–20%之间，敏感肌建议从低浓度（5%–10%）开始；
  - pH值约为2.5–3.5，确保分子稳定性和吸收率；
  - 建议白天使用并务必搭配防晒，避免氧化分解后降低效果。"""
    },
    {
        "question": "烟酰胺（维生素B3）有什么护肤作用？",
        "answer": """\
烟酰胺是一种水溶性维生素，护肤中的主要功效：
  - 修复屏障：促进神经酰胺生成，增强角质层锁水能力；
  - 均匀肤色：抑制黑色素向角质层迁移，淡化色斑和痘印；
  - 保湿锁水：减少经皮水分流失（TEWL），增加肌肤含水量；
  - 抗炎抗痘：抑制皮脂分泌，舒缓痘痘红肿；
  - 改善细纹：刺激胶原蛋白和弹性蛋白合成，减少初老迹象。

使用贴士：
  - 可与大多数成分共用，如维生素C、A醇、果酸；
  - 浓度5%–10%效果显著又温和，敏感肌可从2%–3%浓度起步；
  - 如出现轻微刺痛，可分次或隔天使用，观察肌肤耐受性。"""
    },
    {
        "question": "Skinceuticals CE Ferulic 有什么特点？",
        "answer": """\
Skinceuticals CE Ferulic 是一款经典“黄金三角”抗氧化精华，配方与功效：
  - 成分组合：15% L-抗坏血酸（维C）、1% α-生育酚（维E）、0.5% 阿魏酸；
  - 协同增效：阿魏酸能稳定维C和维E，使三者抗氧化能力提升约8倍；
  - 功效：中和自由基、抑制光老化、提亮肤色、促进胶原蛋白生成；
  - 质地：略为油润，适合正常至干性肌；油性肌可搭配轻薄保湿乳液或凝胶。

使用方法：
  - 清洁、爽肤后，取4–5滴涂抹于面部及颈部，轻拍至吸收；
  - 搭配日常防晒（SPF30+），存放于阴凉避光处，防止氧化。"""
    },
    {
        "question": "露得清（Neutrogena）的维生素C产品适合哪些肤质？",
        "answer": """\
Neutrogena 的维C系列多采用稳定的维C衍生物（如乙基抗坏血酸），特点：
  - 刺激性低：比纯维C更温和，敏感肌或初尝试者友好；
  - 质地清爽：多为轻薄乳液或啫喱，易吸收、不黏腻，适合混合偏油及油性肌；
  - 价格亲民：性价比较高，适合作日常维稳和提亮肤色使用。

使用建议：
  - 可早晚使用，干性肌可后接滋润面霜，油性肌则可单用或搭配控油精华；
  - 如出现轻微刺痛感，可降低使用频率或与保湿产品同用，帮助缓解不适。"""
    },
    {
        "question": "什么是视黄醇（A醇）？如何在护肤中使用？",
        "answer": """\
视黄醇（Retinol）属于维A醇家族，需在皮肤中转化为视黄酸（Tretinoin）才能发挥作用。主要功效：
  - 促进角质更新、疏通毛孔，减少痘痘和暗沉；
  - 刺激胶原蛋白及弹性纤维生成，淡化细纹和皱纹；
  - 均匀肤色，改善色素沉淀。

使用要点：
  - 浓度：0.1%–1%可选，新手建议从0.1%或更低浓度开始；
  - 频率：初用阶段每周2–3次，待耐受后逐步过渡到每晚使用；
  - 搭配：避免与高浓度果酸、其他视黄酸类同晚使用；可先涂面霜再涂A醇以减轻刺激；
  - 防晒：A醇会增加皮肤光敏感性，白天务必使用SPF30以上防晒并定时补涂。

常见副作用：
  - 脱屑、发红、刺痛：为正常耐受反应，通常在2–4周内改善；
  - 缓解方法：可与面霜同用或在皮肤完全干燥后再涂抹，减少对皮肤的直接刺激。"""
    },
    {
        "question": "如何在日常护肤流程中合理搭配维生素C、烟酰胺和A醇？",
        "answer": """\
为了避免成分冲突并最大化效果，推荐以下流程：

早晨：
  1. 清洁  
  2. 爽肤（可选）  
  3. 维生素C精华（抗氧化、提亮）  
  4. 烟酰胺精华或乳液（修复屏障、控油）  
  5. 保湿面霜  
  6. 防晒（SPF30+，每2小时补涂一次）

晚上：
  1. 清洁  
  2. 爽肤  
  3. A醇（视黄醇）精华/乳液（抗老、促进更新）  
     - 若易刺激，可先涂保湿面霜再涂A醇，或隔夜使用  
  4. 烟酰胺（或高保湿修复）产品  
  5. 滋润面霜

注意事项：
  - 若肌肤敏感，可将维C和烟酰胺分时段使用（维C早上、烟酰胺晚上）  
  - A醇不宜与高浓度果酸、A酸同晚使用，以免叠加刺激  
  - 坚持防晒与补水，缓解潜在的成分刺激。"""
    },
]



qa_dict.extend([
    {
        "question": "透明质酸（Hyaluronic Acid）在护肤中的作用是什么？",
        "answer": """\
透明质酸是一种天然保湿因子(NMF)，能在皮肤表面形成保湿膜并深入真皮层吸水。
主要功效包括：锁水保湿、增加皮肤饱满度、改善细纹和干燥。
使用建议：分子量不同渗透深度不同，高低分子可叠加使用；早晚均可涂抹于洁面爽肤后，配合面霜锁水。"""
    },
    {
        "question": "神经酰胺（Ceramides）有什么护肤功效？",
        "answer": """\
神经酰胺是角质层主要脂质成分，负责保持皮肤屏障完整性和防止水分流失。
功效：修复角质层、增强屏障、减少敏感和干纹、提高皮肤耐受力。
使用贴士：配合含神经酰胺的乳霜或精华，坚持早晚使用，敏感肌尤为受益。"""
    },
    {
        "question": "The Ordinary 的烟酰胺10%+锌1%有何特点？",
        "answer": """\
The Ordinary Niacinamide 10% + Zinc 1%是一款高浓度烟酰胺精华，功效包括：抑制黑色素转运、调节皮脂分泌、缩小毛孔、改善痘痘和红印。
使用方式：洁面爽肤后，早晚使用数滴全脸涂抹，若有轻微刺痛可搭配保湿产品或降低使用频率。"""
    },
    {
        "question": "果酸（AHA）和水杨酸（BHA）分别适合什么肤质？",
        "answer": """\
AHA（如甘醇酸、乳酸）为水溶性果酸，作用于表皮角质层，促进角质更新、提亮肤色，适合干燥、暗沉及轻度色斑肌；BHA（如水杨酸）为油溶性，深入毛孔清理油脂和老废角质，适合油痘及闭口肌。
使用建议：浓度通常5%以下，每周1–2次；敏感肌可先从低浓度或低频率尝试。"""
    },
    {
        "question": "多肽（Peptides）在护肤中的作用？",
        "answer": """\
多肽是由氨基酸组成的小分子蛋白质片段，通过模拟天然信号肽刺激皮肤合成胶原蛋白和弹性蛋白。
功效：抗皱紧致、促进修复、增强弹性、改善细纹。
使用方法：洁面爽肤后，涂抹多肽精华或乳液，坚持早晚使用，与保湿成分配合同效更佳。"""
    },
    {
        "question": "如何正确选择和使用防晒产品？",
        "answer": """\
防晒分为物理防晒（矿物质，如氧化锌、二氧化钛）和化学防晒（化学分子吸收UV）。选择SPF30+且PA+++以上的广谱防晒。
使用方法：日常至少涂抹2mg/cm²(约一枚硬币大小)于面部，暴露部位每2小时补涂一次；外出前15–30分钟涂抹；化妆后可用防晒喷雾或粉饼补涂。"""
    }
])



qa_dict.extend([
    {
        "question": "如何判断护肤品成分是否适合自己？",
        "answer": """
要判断是否适合，建议按以下步骤：
1. 明确肤质与敏感史  
   - 干性、油性、混合性还是敏感肌？  
   - 是否对某些常见成分（香精、酒精、防腐剂、特定酸类）有不良反应？  
2. 阅读成分表（INCI）  
   - 把简易成分（植物油、水、甘油等）和活性成分区分开。  
   - 留意刺激性／潜敏成分（酒精、硫酸盐、精油、高浓度果酸、维A酸衍生物等）。  
   - 油痘肌：优选“非致痘性（Non-comedogenic）”配方，含抗炎、控油成分（如水杨酸、锌、茶树精油）。  
3. 浓度与配伍  
   - 活性成分需关注推荐浓度范围（例如：维C 10–20%、AHA 5–10%）。  
   - 避免高刺激成分叠加使用，必要时分晨／晚或隔日使用。  
4. 贴肤测试  
   - 在耳后或下颌线处涂抹少量，等待 24–48 小时，观察是否有红肿、刺痛、瘙痒。  
若无异常，再考虑全脸使用。"""
    },
    {
        "question": "如何理解护肤品中的 pH 值？为什么重要？",
        "answer": """
pH 值直接影响成分稳定性、皮肤耐受性和功效发挥。  
1. 皮肤屏障：健康角质层 pH≈4.5–5.5，偏酸性有助抑菌、锁水。  
2. 常见活性成分最佳 pH：  
   - 维生素 C（L-抗坏血酸）：2.5–3.5  
   - 果酸（AHA，如甘醇酸、乳酸）：3.0–4.0  
   - BHA（水杨酸）：3.0–4.0  
   - 烟酰胺：5.0–7.0  
3. 应用建议：  
   - 低 pH 产品使用后易暂时破坏酸性保护层，后续务必补水、保湿、加强防晒。  
   - 若多种酸类或与其他高 pH 产品冲突，建议错时使用（如早晚交替），避免互相中和或刺激过度。"""
    },
    {
        "question": "化学去角质与物理去角质哪种更好？",
        "answer": """
两者原理不同，适用人群和风险各异：  
1. 化学去角质  
   - 机制：AHA/BHA/酶类等溶解老废角质，温和、均匀、可控。  
   - 优点：对痘痘、闭口、细纹改善明显；低浓度日常可放心使用。  
   - 建议：每周 1–2 次（敏感肌可降至 0.5%–5% 果酸或酶类）。  
2. 物理去角质  
   - 机制：微粒、刷具或粗糙颗粒机械摩擦剥离角质。  
   - 风险：易过度摩擦导致屏障受损、微血管破裂、发红。  
   - 建议：选用超细柔珠、柔软刷头等，频率≤1 次/周，并轻柔打圈。  
综合来说，绝大多数肤质首选化学去角质，敏感肌可优选低浓度酶类；物理去角质需谨慎且频率极低。"""
    },
    {
        "question": "多重精华液如何分层涂抹？",
        "answer": """
分层原则：从最“稀薄”“水润”到最“丰润”“油性”——由内而外分三层：  
1. 水状／凝露（最轻薄）  
   - 例：透明质酸原液、烟酰胺水、精华水  
2. 乳状／乳液（中等稠度）  
   - 例：维C乳、胜肽修复精华、舒缓乳液  
3. 油状／霜状（最厚重）  
   - 例：面部油、多肽霜、神经酰胺霜  
使用方法：  
• 每层取量适中，薄涂后等待 30–60 秒，感到呈半吸收状态再叠加下一层。  
• 若同一类别有早晚不同功效，可按晨／晚顺序优先使用最需要的精华。"""
    },
    {
        "question": "护肤品中的“非致痘性（Non-comedogenic）”是什么意思？",
        "answer": """
“Non-comedogenic”意指经测试配方或单一成分不易堵塞毛孔、引发粉刺。  
主要特点：  
- 分子量适中且不形成厚重膜感  
- 不含或极少含易堵塞成分（如某些矿物油、厚重油脂）  
适用人群：  
- 油痘肌、混合偏油、粉刺倾向者，可优先选择非致痘性标示产品，搭配抗炎（如水杨酸、茶树油）和控油成分，加强痘痘管理。"""
    },
    {
        "question": "如何根据季节调整护肤流程？",
        "answer": """
根据温湿度与肌肤需求微调：  
1. 春秋：气候舒适  
   - 维持基础清洁—爽肤—精华—面霜流程  
   - 加强屏障修复（神经酰胺、磷脂）与适度保湿  
2. 夏季：高温高湿  
   - 清洁：可选温和去油洁面  
   - 精华：偏轻薄水／凝露质地，高效抗氧化（维C、绿茶）  
   - 乳／霜：换清爽型、凝胶型，控油防晒必备  
3. 冬季：寒冷干燥  
   - 清洁：低泡沫、无皂基洁面  
   - 精华：高浓度保湿剂（透明质酸、大分子／小分子搭配）  
   - 霜：厚重锁水霜，定期使用滋润面膜或睡眠面膜  
4. 过渡期可微调去角质频次（春秋可适度增加，夏冬则减少或更换温和配方），并根据肌肤状态随时增减保湿／控油产品。"""
    }
])








qa_dict.extend([
    {
        "question": "抗生素是如何发挥作用的？",
        "answer": """
抗生素通过靶向细菌的关键结构或代谢过程来抑制其生长或直接杀灭，主要作用机制包括：
1. 干扰细胞壁合成  
   - β-内酰胺类（青霉素、头孢菌素、碳青霉烯）：结合并抑制青霉素结合蛋白（PBP），阻止肽聚糖交联→细菌细胞壁失稳→细胞裂解  
   - 糖肽类（万古霉素、替考拉宁）：结合D-Ala-D-Ala末端，阻断肽聚糖前体转运和聚合  
2. 抑制蛋白质合成  
   - 30S核糖体结合：四环素类（四环素、米诺环素）阻断氨酰-tRNA进入A位点  
   - 50S核糖体结合：大环内酯类（红霉素、阿奇霉素）、林可霉素、氯霉素阻断肽链延伸或转位  
3. 干扰核酸合成  
   - 喹诺酮类（环丙沙星、左氧氟沙星）：抑制DNA旋转酶（拓扑异构酶II）和拓扑异构酶IV→阻止DNA解旋与复制  
   - 利福平：抑制细菌DNA依赖性RNA聚合酶→阻断转录  
4. 破坏细胞膜完整性  
   - 多粘菌素（Colistin）：与革兰阴性菌外膜脂多糖结合，改变膜通透性→细胞内容物泄漏  
5. 抑制代谢途径  
   - 磺胺类+甲氧苄啶联合（复方新诺明）：拮抗二氢叶酸合成酶及其前体，阻止叶酸合成→干扰核苷酸合成  
临床应用中务必对症下药，遵医嘱足量足疗程，以减少耐药发生并保护正常菌群。"""
    },
    {
        "question": "抗病毒药物的基本作用机制有哪些？",
        "answer": """
抗病毒药物通过靶向病毒生命周期的不同阶段，抑制其入侵、复制或组装，主要机制包括：
1. 阻断病毒吸附或穿入  
   - HIV融合抑制剂（Enfuvirtide）：结合gp41，阻止病毒与细胞膜融合  
   - CCR5拮抗剂（Maraviroc）：阻断病毒识别CCR5受体  
2. 抑制病毒基因组复制  
   - 核苷/核苷酸类似物（阿昔洛韦、伐昔洛韦、替诺福韦、拉米夫定）：经磷酸化后掺入病毒DNA/RNA链，因无3′-OH基团导致链终止  
   - 非核苷逆转录酶抑制剂（EFV、NVP）：与逆转录酶非竞争性结合，改变酶结构、失去活性  
3. 抑制病毒蛋白加工  
   - HIV/丙肝蛋白酶抑制剂（洛匹那韦、达芦那韦、索非布韦）：阻断多聚蛋白切割，产生非功能性病毒颗粒  
4. 抑制病毒脱壳或组装  
   - 离子通道抑制剂（金刚烷胺、金刚乙胺）：阻止病毒衣壳pH变化，阻碍基因组释放  
   - 下调宿主蛋白（例如：IKBα稳定剂）间接阻断病毒组装  
5. 调节宿主免疫  
   - 干扰素类（IFN-α、IFN-β）：激活抗病毒防御基因，增强细胞抗病毒能力  
使用时需根据病毒类型、耐药突变及患者肝肾功能、合并用药等因素个体化选择。"""
    },
    {
        "question": "什么是耐药性？如何预防细菌耐药？",
        "answer": """
耐药性指病原菌通过基因突变或水平基因转移获得对抗菌药物的抵抗能力。常见机制：
- 产生β-内酰胺酶或改变药物靶点（PBP、核糖体）  
- 通过外排泵（efflux pump）主动排出药物  
- 改变膜通透性或获取替代代谢途径  
预防耐药策略：
1. 合理用药  
   - 抗生素应在细菌学诊断和药敏指导下使用，首选窄谱药  
   - 遵医嘱完成整个疗程，不随意停药或加量  
2. 加强感染控制  
   - 医院严格手卫生、环境和设备消毒、隔离策略  
   - 社区与长护机构推广免洗手消毒、接种疫苗  
3. 抗生素管理（ASP）  
   - 定期监测耐药谱、动态调整用药指南  
   - 多学科团队评估处方合理性  
4. 科普教育  
   - 提高公众对耐药危害和合理用药的认识  
   - 减少非处方抗生素使用和农牧食品中滥用抗生素  
5. 研发新药和替代疗法  
   - 推动新抗生素、抗菌肽、噬菌体、抗体疗法等研究与应用。"""
    },
    {
        "question": "抗菌药物治疗中为什么要监测血药浓度？",
        "answer": """
血药浓度监测（TDM, Therapeutic Drug Monitoring）能确保药物在体内既达有效浓度又不至于中毒。主要目的：
1. 保证疗效  
   - 维持血药浓度高于病原菌的最低抑菌浓度（MIC），避免亚疗程用药导致耐药  
2. 降低毒性  
   - 对肾毒性（氨基糖苷类、万古霉素）或耳毒性药物（庆大霉素）进行剂量调整  
3. 个体化给药  
   - 根据患者年龄、体重、肾肝功能、液体分布、合并用药情况精确调整剂量与给药间隔  
4. 指导特殊人群用药  
   - 重症监护、烧伤患者、大手术后、妊娠、儿童和老年人常需密切监测  
5. 评估依从性  
   - 判断患者是否按时按量给药，帮助临床做进一步用药管理决策。"""
    },
    {
        "question": "疫苗是如何激发免疫保护的？",
        "answer": """
疫苗模拟天然感染，激活机体先天与适应性免疫系统，建立免疫记忆并提供长期保护。关键步骤：
1. 抗原呈递  
   - 树突状细胞或巨噬细胞摄取、加工疫苗抗原并在二级淋巴器官表达MHC分子呈递给T细胞  
2. 激活T细胞  
   - CD4+辅助性T细胞分化（Th1、Th2、Tfh），分泌细胞因子促进B细胞成熟；  
   - CD8+细胞毒性T细胞识别并清除被感染的细胞  
3. 激活B细胞与抗体生成  
   - B细胞在Tfh细胞帮助下分化为浆细胞，产生特异性中和抗体；  
   - 体液免疫阻断病原体入侵和扩散  
4. 建立免疫记忆  
   - 产生记忆B细胞和记忆T细胞，遇到真实病原快速应答  
5. 强化剂/加强针  
   - 部分疫苗需多剂程或定期加强以维持高水平保护  
不同类型疫苗（灭活、减毒、亚单位、核酸、载体）有各自优势与注意事项，应依国家免疫规划及个体健康状况接种。"""
    }
])



qa_dict.extend([
    {
        "question": "如何计算两点之间的欧氏距离（L2 距离）？",
        "answer": '''
使用欧氏距离公式：

distance = sqrt(Σ_i (x_i - y_i)^2)

下面是一个纯 Python 实现，支持任意维度的向量：

```python
import math
from typing import Sequence

def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """计算等长向量 a 和 b 之间的 L2 距离。"""
    if len(a) != len(b):
        raise ValueError("向量长度必须相同")
    sq_sum = 0.0
    for ai, bi in zip(a, b):
        diff = ai - bi
        sq_sum += diff * diff
    return math.sqrt(sq_sum)

# 示例
if __name__ == "__main__":
    p1 = (1.0, 2.0, 3.0)
    p2 = (4.0, 6.0, 8.0)
    print(f"L2 距离: {l2_distance(p1, p2):.4f}")  # ≈7.0711
```

适用场景：图像特征匹配、聚类、最近邻搜索等。
'''
    },
    {
        "question": "如何计算向量之间的余弦相似度？",
        "answer": '''
余弦相似度衡量两个向量在方向上的相似程度，公式：

cosine_similarity = (a · b) / (||a|| * ||b||)

取值范围在 [-1, 1] 之间，越接近 1 表示越相似。下面给出 NumPy 实现：

```python
import numpy as np
from typing import Sequence

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """计算向量 a 和 b 之间的余弦相似度。"""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("输入必须是一维向量")
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("输入向量不应全为 0")
    return dot / (norm_a * norm_b)

# 示例
if __name__ == "__main__":
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    sim = cosine_similarity(v1, v2)
    print(f"余弦相似度: {sim:.4f}")  # ≈0.9746
```
'''
    },
    {
        "question": "如何使用 OpenCV + Dlib 对两张人脸图片进行基本对比？",
        "answer": '''
基本流程：
1. 人脸检测  
2. 关键点定位  
3. 提取 128 维人脸嵌入向量  
4. 计算 L2 距离或余弦相似度  

完整示例代码：

```python
import cv2
import dlib
import numpy as np
from typing import Optional

# 模型文件路径，需要提前下载
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL = "dlib_face_recognition_resnet_model_v1.dat"

# 初始化
_detector = dlib.get_frontal_face_detector()
_shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
_face_rec_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)

def get_face_embedding(image_path: str) -> Optional[np.ndarray]:
    """读取图片并返回第一张人脸的 128D 嵌入向量，未检测到人脸返回 None。"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _detector(gray, 1)
    if not faces:
        return None
    # 使用第一个检测到的人脸
    shape = _shape_predictor(gray, faces[0])
    embedding = _face_rec_model.compute_face_descriptor(img, shape)
    return np.array(embedding, dtype=float)

if __name__ == "__main__":
    emb1 = get_face_embedding("face1.jpg")
    emb2 = get_face_embedding("face2.jpg")
    if emb1 is None or emb2 is None:
        print("未检测到人脸或图片读取失败")
    else:
        l2_dist = np.linalg.norm(emb1 - emb2)
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"L2 距离: {l2_dist:.4f}")
        print(f"余弦相似度: {cos_sim:.4f}")
```
'''
    },
    {
        "question": "如何计算网络传输与磁盘 I/O 的吞吐率？",
        "answer": '''
吞吐率 = 传输的数据量 / 时间，常用单位 MB/s（Megabytes per second） 或 Mbps（Megabits per second）。

下面使用 psutil 库监测指定时间间隔内的网卡和磁盘吞吐率：

```python
import psutil
import time
from typing import Optional, Tuple

_prev_net = None  # 上一次采样的网络快照和时间
_prev_disk = None  # 上一次采样的磁盘快照和时间

def _now() -> float:
    """高精度时间戳。"""
    return time.perf_counter()

def net_io_speed(interval: Optional[float] = 1.0) -> Tuple[float, float]:
    """
    返回上行、下行吞吐率，单位 MB/s。
    若 interval 不为 None，则阻塞 interval 秒后测量；
    否则基于上次快照非阻塞测量。
    """
    global _prev_net
    t1 = _now()
    counters1 = psutil.net_io_counters()

    if interval is not None or _prev_net is None:
        time.sleep(interval or 0)
        t0 = t1
        counters0 = counters1
        t1 = _now()
        counters1 = psutil.net_io_counters()
    else:
        counters0, t0 = _prev_net

    sent_delta = counters1.bytes_sent - counters0.bytes_sent
    recv_delta = counters1.bytes_recv - counters0.bytes_recv
    dt = t1 - t0 if t1 - t0 > 0 else 1e-6

    _prev_net = (counters1, t1)
    up = sent_delta / dt / (1024**2)
    down = recv_delta / dt / (1024**2)
    return up, down

def disk_io_speed(interval: Optional[float] = 1.0) -> Tuple[float, float]:
    """
    返回读、写吞吐率，单位 MB/s。interval 同上。
    """
    global _prev_disk
    t1 = _now()
    counters1 = psutil.disk_io_counters()

    if interval is not None or _prev_disk is None:
        time.sleep(interval or 0)
        t0 = t1
        counters0 = counters1
        t1 = _now()
        counters1 = psutil.disk_io_counters()
    else:
        counters0, t0 = _prev_disk

    read_delta = counters1.read_bytes - counters0.read_bytes
    write_delta = counters1.write_bytes - counters0.write_bytes
    dt = t1 - t0 if t1 - t0 > 0 else 1e-6

    _prev_disk = (counters1, t1)
    rd = read_delta / dt / (1024**2)
    wr = write_delta / dt / (1024**2)
    return rd, wr

if __name__ == "__main__":
    up, down = net_io_speed(interval=1.0)
    rd, wr = disk_io_speed(interval=1.0)
    print(f"网络 ↑{up:.2f} MB/s, ↓{down:.2f} MB/s")
    print(f"磁盘 读{rd:.2f} MB/s, 写{wr:.2f} MB/s")

    net_io_speed(interval=1.0)
    disk_io_speed(interval=1.0)

    print("\\n持续监测（每 0.5s 打印一次）：")
    try:
        while True:
            time.sleep(0.5)
            up, down = net_io_speed(interval=None)
            rd, wr = disk_io_speed(interval=None)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] 网络 ↑{up:.2f} MB/s, ↓{down:.2f} MB/s | 磁盘 读{rd:.2f} MB/s, 写{wr:.2f} MB/s")
    except KeyboardInterrupt:
        print("监测结束。")
```
'''
    },
    {
        "question": "如何在 Python 中高效计算二维点集的 L2 距离矩阵？",
        "answer": '''
给定点集 A (m×d) 和 B (n×d)，返回形状为 (m, n) 的距离矩阵 D，其中

D[i, j] = ||A[i] - B[j]||_2.

推荐使用 NumPy 广播和矢量化操作：

```python
import numpy as np
from typing import Sequence

def pairwise_l2_np_fast(
    A: Sequence[Sequence[float]],
    B: Sequence[Sequence[float]]
) -> np.ndarray:
    """更高效地计算 A 和 B 之间的成对 L2 距离矩阵。"""
    A = np.asarray(A, dtype=np.float64)  # shape: (m, d)
    B = np.asarray(B, dtype=np.float64)  # shape: (n, d)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("输入应为同维度的二维点集")

    A_sq = np.einsum('ij,ij->i', A, A)
    B_sq = np.einsum('ij,ij->i', B, B)
    cross = A @ B.T
    d2 = A_sq[:, None] + B_sq[None, :] - 2 * cross
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)

if __name__ == "__main__":
    A = [[0, 0], [1, 1], [2, 2]]
    B = [[1, 0], [2, 1]]
    D = pairwise_l2_np_fast(A, B)
    print(D)
    # 预期输出：
    # [[1.         2.23606798]
    #  [1.         1.        ]
    #  [2.23606798 1.        ]]
```
'''
    }
])







```python
qa_dict.extend([
    {
        "question": "什么是 KMP 算法？如何高效实现字符串匹配？",
        "answer": '''
KMP（Knuth–Morris–Pratt）算法是一种高效的字符串匹配算法，避免了在匹配失败时回溯主串指针，从而保证时间复杂度为 O(n + m)。  
核心在于预处理模式串，构造最长相等前后缀数组（lps）来利用已经匹配过的信息。

```python
from typing import List

def kmp_prefix(pattern: str) -> List[int]:
    """构造模式串的最长前缀后缀表 lps。"""
    lps = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j
    return lps

def kmp_search(text: str, pattern: str) -> List[int]:
    """返回 pattern 在 text 中所有匹配的起始索引。"""
    if not pattern:
        return []
    lps = kmp_prefix(pattern)
    result = []
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            result.append(i - j + 1)
            j = lps[j - 1]
    return result

if __name__ == "__main__":
    text = "ababcabcabababd"
    pattern = "ababd"
    positions = kmp_search(text, pattern)
    print(f"匹配位置：{positions}")  # [10]
```
'''
    },
    {
        "question": "如何计算两个字符串的最长公共子序列（LCS）？",
        "answer": '''
最长公共子序列（Longest Common Subsequence, LCS）指两个序列中最长的公共子序列，不要求连续，但要保持相对顺序。

```python
from typing import Tuple

def lcs_length(s1: str, s2: str) -> Tuple[int, str]:
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    lcs_chars = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs_chars.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    lcs_chars.reverse()
    return dp[m][n], "".join(lcs_chars)

if __name__ == "__main__":
    s1 = "ABCBDAB"
    s2 = "BDCABA"
    length, subseq = lcs_length(s1, s2)
    print(f"LCS长度: {length}, 一个LCS: {subseq}")
    # 输出: LCS长度: 4, 一个LCS: BCBA
```
'''
    },
    {
        "question": "如何计算两个字符串之间的编辑距离（Levenshtein距离）？",
        "answer": '''
编辑距离是指将一个字符串转换成另一个字符串所需的最少单字符编辑操作数，包括插入、删除和替换。

```python
def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + 1
                )
    return dp[m][n]

if __name__ == "__main__":
    w1 = "kitten"
    w2 = "sitting"
    dist = levenshtein_distance(w1, w2)
    print(f"编辑距离: {dist}")  # 3
```
'''
    },
    {
        "question": "如何用滑动窗口算法寻找字符串中不重复的最长子串？",
        "answer": '''
滑动窗口用于寻找字符串中最长无重复字符的子串，复杂度 O(n)。

```python
def length_of_longest_substring(s: str) -> int:
    char_index = {}
    left = 0
    max_len = 0
    for right, c in enumerate(s):
        if c in char_index and char_index[c] >= left:
            left = char_index[c] + 1
        char_index[c] = right
        max_len = max(max_len, right - left + 1)
    return max_len

if __name__ == "__main__":
    s = "abcabcbb"
    print(f"最长无重复子串长度: {length_of_longest_substring(s)}")  # 3
```
'''
    },
    {
        "question": "如何用动态规划实现0-1背包问题？",
        "answer": '''
0-1背包问题：给定物品重量和价值，求在背包容量限制下能取得的最大价值。

```python
from typing import List, Tuple

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i] -1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]

if __name__ == "__main__":
    w = [2, 1, 3, 2]
    v = [12, 10, 20, 15]
    cap = 5
    print(f"最大价值: {knapsack_01(w, v, cap)}") # 37
```
'''
    },
    {
        "question": "如何实现快速幂算法计算大整数的乘方取模？",
        "answer": '''
快速幂用于高效计算 (x^n) % mod，时间复杂度 O(log n)。

```python
def fast_pow(x: int, n: int, mod: int) -> int:
    result = 1 % mod
    base = x % mod
    while n > 0:
        if n & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        n >>= 1
    return result

if __name__ == "__main__":
    x = 2
    n = 10
    mod = 1000
    print(f"{x}^{n} mod {mod} = {fast_pow(x, n, mod)}")  # 24
```
'''
    },
    {
        "question": "如何用二分查找算法查找有序数组中的目标值？",
        "answer": '''
二分查找时间复杂度为 O(log n)，适用于有序数组。

```python
from typing import List, Optional

def binary_search(arr: List[int], target: int) -> Optional[int]:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return None

if __name__ == "__main__":
    nums = [1,3,5,7,9]
    t = 5
    idx = binary_search(nums, t)
    print(f"目标 {t} 的索引: {idx}")  # 2
```
'''
    },
    {
        "question": "Trie树（字典树）是什么，如何用它实现字符串前缀查询？",
        "answer": '''
Trie树是用于高效存储和查询字符串集合的数据结构，支持快速的前缀搜索。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True

if __name__ == "__main__":
    trie = Trie()
    trie.insert("apple")
    print(trie.search("apple"))   # True
    print(trie.search("app"))     # False
    print(trie.starts_with("app"))# True
    trie.insert("app")
    print(trie.search("app"))     # True
```
'''
    },
    {
        "question": "Dijkstra算法如何计算带权图的最短路径？",
        "answer": '''
Dijkstra 算法用于计算单源最短路径，时间复杂度 O(E log V)。

```python
import heapq
from typing import List, Tuple

def dijkstra(graph: List[List[Tuple[int, int]]], start: int) -> List[float]:
    """
    graph是邻接表，graph[u] = [(v, w), ...]
    返回 start 到各点的最短距离。
    """
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist

if __name__ == "__main__":
    graph = [
        [(1, 2), (2, 5)],
        [(2, 1), (3, 4)],
        [(3, 1)],
        []
    ]
    start = 0
    distances = dijkstra(graph, start)
    print(f"从 {start} 出发的最短路径: {distances}")
    # [0, 2, 3, 4]
```
'''
    }
])
```



qa_dict.extend([
    {
        "question": "什么是余弦相似度？如何用Python计算？",
        "answer": '''
余弦相似度用于衡量两个向量方向的相似度，计算两个向量夹角的余弦值，范围[-1,1]，值越大，表示越相似，常用于文本和推荐系统中。

```python
from typing import List
import math

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a*b for a,b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# 示例
if __name__ == "__main__":
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    sim = cosine_similarity(v1, v2)
    print(f"余弦相似度: {sim:.4f}")  # 输出约0.9746
```
'''
    },
    {
        "question": "Bellman-Ford算法如何计算最短路径？适用场景是什么？",
        "answer": '''
Bellman-Ford算法用于计算带负权边的单源最短路径，可检测负权环。其原理是不断松弛各边，最多执行|V|-1轮松弛。

```python
from typing import List, Tuple

def bellman_ford(edges: List[Tuple[int, int, int]], n: int, start: int) -> List[float]:
    dist = [float('inf')] * n
    dist[start] = 0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # 负权环检测
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return []  # 存在负权环

    return dist

if __name__ == "__main__":
    edges = [
        (0, 1, 6), (0, 2, 7), (1, 2, 8), (1, 3, 5),
        (1, 4, -4), (2, 3, -3), (2, 4, 9), (3, 1, -2),
        (4, 0, 2), (4, 3, 7)
    ]
    dist = bellman_ford(edges, 5, 0)
    if dist:
        print("起点0到各点最短距离:", dist)
    else:
        print("检测到负权环，无法计算最短路径。")
```
'''
    },
    {
        "question": "Floyd-Warshall算法怎样求任意两点的最短路径？",
        "answer": '''
Floyd-Warshall算法通过动态规划，逐步尝试所有中间节点更新任意两点间最短路径，适合稠密图，时间复杂度O(n³)。

```python
from typing import List

def floyd_warshall(graph: List[List[float]]) -> List[List[float]]:
    n = len(graph)
    dist = [row[:] for row in graph]  # 深拷贝，避免修改原图

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

if __name__ == "__main__":
    INF = float('inf')
    graph = [
        [0, 3, INF, 7],
        [8, 0, 2, INF],
        [5, INF, 0, 1],
        [2, INF, INF, 0]
    ]
    dist = floyd_warshall(graph)
    print("任意两点间最短路径矩阵:")
    for row in dist:
        print(row)
```
'''
    },
    {
        "question": "如何用Trie树实现单词插入和计数？",
        "answer": '''
Trie树是一种用于字符串快速查找的树形结构，可高效支持前缀查询及单词计数。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0  # 此节点结尾单词出现次数

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            node = node.children.setdefault(c, TrieNode())
        node.count += 1

    def count_word(self, word: str) -> int:
        node = self.root
        for c in word:
            if c not in node.children:
                return 0
            node = node.children[c]
        return node.count

if __name__ == "__main__":
    trie = Trie()
    words = ["apple", "app", "apple", "apply"]
    for w in words:
        trie.insert(w)
    print(trie.count_word("apple"))  # 2
    print(trie.count_word("app"))    # 1
    print(trie.count_word("ap"))     # 0
```
'''
    },
    {
        "question": "堆排序Heap Sort如何实现？",
        "answer": '''
堆排序是利用堆结构实现的排序算法，将数组视为二叉堆，先建堆，再依次取出堆顶元素放到末尾。

```python
from typing import List

def heapify(arr: List[int], n: int, i: int) -> None:
    largest = i
    l, r = 2*i + 1, 2*i + 2

    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr: List[int]) -> None:
    n = len(arr)
    # 构建最大堆
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    # 依次取出最大值
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

if __name__ == "__main__":
    data = [4, 10, 3, 5, 1]
    heap_sort(data)
    print(data)  # [1, 3, 4, 5, 10]
```
'''
    },
    {
        "question": "如何用Manacher算法求最长回文子串长度？",
        "answer": '''
Manacher算法通过对字符串插入分隔符，统一处理奇偶长度回文，实现O(n)时间复杂度求最长回文子串长度。

```python
def manacher(s: str) -> int:
    # 插入#处理奇偶回文统一
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n
    center = right = 0
    max_len = 0

    for i in range(n):
        if i < right:
            p[i] = min(right - i, p[2*center - i])
        else:
            p[i] = 0
        # 中心扩展
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        # 更新回文中心和右边界
        if i + p[i] > right:
            center, right = i, i + p[i]
        max_len = max(max_len, p[i])
    return max_len

if __name__ == "__main__":
    s = "abacdfgdcaba"
    length = manacher(s)
    print(f"最长回文子串长度: {length}")  # 输出3，如"aba"
```
'''
    }
])
