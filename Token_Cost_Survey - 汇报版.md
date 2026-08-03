# 大语言模型token消耗与成本优化研究综述

> 摘　要　大语言模型（Large Language Model，LLM）从问答工具发展为可调用工具、执行长期任务的智能体（Agent）之后，token 消耗成为制约其规模化部署的主要成本因素。本文围绕 token 消耗问题，以“观测空间—动作—优化方式”为统一框架对 65 篇论文进行梳理：观测空间描述按何种粒度观测 token 消耗（共八个层级，从单个 token 到整个多智能体系统），动作描述用观测到的东西做了什么控制决策，优化方式描述具体用什么方法实现 token 优化（共十条路线）。分析表明，token 尚无统一分类标准，按视角可划分为产生阶段、生命周期功能、计费身份、功能类型与信息密度五个维度；各粒度上的观测—动作配对在第 3 章逐级归纳；监测以调用级四类计费记账为基础，结合阶段聚合与跨运行方差统计；优化方式按对象可划分为十条路线。输入 token 主导总成本、token 削减不等于成本削减、评估必须采用预算匹配比较，是当前研究公认的三条经验；跨平台成本标准化与计费透明度仍是待解决的主要问题。

> 关键词　大语言模型；token消耗；成本优化；统计粒度；缓存计费

---

## 1　引　言

大语言模型在目标检测、自然语言处理、语音识别等领域应用成效卓然，推动了人工智能的发展。随着模型能力的增强，其应用形态由单轮问答逐步演化为能够调用外部工具、自主执行多步任务的智能体系统。在智能体场景下，模型需要反复进行推理、工具调用与自我纠错，每一次循环都直接消耗token；多智能体系统还需在输入、通信与输出三个阶段间传递token。这一变化使token消耗成为最突出的成本问题。

从数据上看，OpenRouter 平台的每周token处理量在 15 个月内从 0.4 万亿增长至 27 万亿，增长约 68 倍[1]；智能体型编码任务一次运行平均消耗约 195 万token、花费 6.1 美元，同一任务的不同运行之间消耗差异最高可达 10 倍[4]。进一步的分析表明，编码智能体的输入token即使在启用缓存的情况下仍主导总成本，代码评审一个阶段即占全部token消耗的 59.4%[5]；隐藏推理token常为可见答案token的 10 倍以上，占计费token的 90% 以上，用户为其付费却无法观测[11]。上述事实说明，token成本问题并非单纯由模型价格引起，而是消耗去向、度量口径与优化手段三方面共同作用的结果。

围绕这一问题，本文对 2024 年至 2026 年间 65 篇相关论文进行梳理，全文以“观测空间—动作—优化方式”为统一框架组织。三个概念贯穿全文：观测空间描述按何种粒度、观测哪些信息（第 3 章）；动作描述用观测到的东西做了什么控制决策（第 3 章各粒度配对）；优化方式描述具体用什么方法实现 token 优化（第 5 章，十条路线）。第 2 节介绍 token 的分类方式（观测的对象基础）；第 4 节介绍监测方法，回答“怎么观测”；第 6 节给出主要发现与待解决问题，第 7 节总结全文。

---

## 2　token的分类方式

目前学术界尚未形成统一的token分类体系，多篇工作从不同角度提出了分类方案。本文将其归纳为五个维度，并单独介绍特殊token类型。各维度的分类视角、代表工作与用途如表 1 所示。

表 1　token分类的五维总览

| 分类维度 | 划分依据 | 代表工作 | 主要用途 |
|----------|----------|----------|----------|
| 产生阶段 | 自回归推理的过程 | ARES[23]、ToneCost[16] | 理解各类token如何计费 |
| 生命周期功能 | 智能体场景中的经济角色 | Token Economics[1] | 解释token花在哪些环节 |
| 计费身份 | 缓存读写与输出的身份 | CAPC[18]、TokenReduction[6] | 计算真实账单金额 |
| 功能类型 | token的语义效用 | SkillReducer[14] | 判断哪些内容可以压缩 |
| 信息密度 | 数据的表示构成 | ONTO[55] | 定位格式开销的来源 |

### 2.1　按产生阶段分类

按产生阶段分类是最基础的分类方式，由 Transformer 自回归推理的过程决定，将token划分为输入、输出、思考与缓存四类，各类别的含义与计费特征如表 2 所示。

表 2　按产生阶段分类

| 类别 | 含义 | 计费特征 |
|------|------|----------|
| 输入token | 进入模型的全部文本：系统提示、查询、上下文、工具结果 | 按输入价计费，影响首token延迟 |
| 输出token | 模型逐个生成的文本 | 按输出价计费，通常是输入的 2–5 倍 |
| 思考token | 推理模型给出答案前的内部推理过程 | 按输出价计费，但对用户不可见 |
| 缓存token | 多个请求共享的前缀，KV 缓存可复用 | 享受折扣价 |

输入与输出token的计费差异决定了输出token的单位成本远高于输入token[4][23]；思考token的不可见性引出了计费透明性问题（详见第 4 节）；缓存token的折扣定价是后续各项优化方式成立的基础。

### 2.2　按生命周期功能分类

Token Economics 综述从经济学角度出发，将token视为经济基本单元，兼具生产要素、交换媒介与记账单位三种角色，并按生命周期功能将智能体场景下的token划分为五类[1]，如表 3 所示。

表 3　按生命周期功能分类

| 类别 | 经济学角色 | 举例 |
|------|-----------|------|
| 输入token | 原材料 | 用户提示词 |
| 推理token | 中间产品 | 思维链推理过程 |
| 通信token | 中间产品 | 多智能体之间的消息传递 |
| 外部token | 中间产品 | RAG/API 检索结果 |
| 输出token | 总产出 | 最终生成内容 |

该综述还给出了 token 的 shadow price 公式：一个 token 的真实单价等于 API 计费单价与等待时间的机会成本之和。该公式表明，成本分析不能只看单价，还需计入用户的等待时间价值。多智能体场景中，智能体反复传递完整上下文带来的通信开销（论文称之为“通信税”，输入输出比约为 2:1 至 3:1）是成本结构的核心特征[1]。

### 2.3　按计费身份分类

同一批输入token，其计费身份不同则单价不同。ARES 给出了标准化的四类计费公式[23]：每次调用的成本等于非缓存输入、缓存读取、缓存创建与输出四类token的数量分别乘以各自单价后的和。各计费身份的费率相对值如表 4 所示。

表 4　计费身份与费率

| 计费身份 | 相对费率 | 说明 |
|----------|----------|------|
| 非缓存输入 | 1.0 | 原价 |
| 缓存读取 | 约 0.1 | 折扣价 |
| 缓存创建 | 约 1.25 | 写溢价 |
| 输出 | 2–5 | 输入价的数倍 |

计费身份对成本分析具有决定性影响。缓存读取与缓存创建的价差达 12.5 倍[6][18]；真实账单中，缓存创建与缓存读取合计约占编码智能体账单的 80%[6]；复用预计算 KV 缓存比从头执行预填充便宜 8.6 至 49.7 倍。此外，隐藏推理token的账目可被恶意服务商虚增最多 1469%，计费必须绑定到服务商无法控制的证据上[12]。

### 2.4　按功能类型分类

按功能类型分类着眼于token的语义效用。SkillReducer 对 55,315 个公开智能体技能的分析表明，技能正文可划分为五类[14]，各类别的占比与压缩难度如表 5 所示。

表 5　按功能类型分类（技能正文）

| 功能类别 | 占比 | 能否压缩 | 说明 |
|----------|------|----------|------|
| 核心指令 | 约 38.5% | 很难 | Agent 必须遵守的可执行指令 |
| 背景知识 | 约 40.7% | 容易 | 解释性内容、原理说明 |
| 示例 | 约 12.9% | 一般 | 代码片段、输入输出对 |
| 模板 | 约 7.6% | 容易 | 可复制的样板代码 |
| 冗余 | 约 0.3% | 最容易 | 重复内容，可直接删除 |

结合其他智能体系统的分析，功能类型还可细分为系统token、动作token、观察token、记忆token、规划token与元认知token六类，分别对应系统提示、工具调用语法、工具返回结果、检索历史、推理计划与自检纠错[14][56][54][32]。

### 2.5　按信息密度分类

按信息密度分类从数据表示视角出发。ONTO 对结构化数据的token组成进行了消融分析，将 JSON 数据中的token按角色拆开统计[55]，结果如表 6 所示。

表 6　按信息密度分类（JSON token构成）

| 组成类别 | JSON 中占比 | 优化空间 |
|----------|-------------|----------|
| 字段名/键 | 52.6% | 最大（键消除贡献超过全部毛节省） |
| 标点符号 | 23.2% | 次大 |
| 数值内容 | 17.9% | 不可压缩（这是信息本身） |
| 结构缩进 | 0% | 层级结构要付出的代价 |
| 空白字符 | 6.3% | 格式开销 |

### 2.6　特殊token类型

除上述五类分类维度外，部分工作还设计了具有特定功能的特殊token，如表 7 所示。

表 7　特殊token类型

| 特殊token | 说明 | 代表工作 |
|----------|------|----------|
| 控制token | 触发特定行为的专用token（如 $\tau_{off}$ 触发模型切换） | PyroDash[52] |
| 路由token | 决定下一步由哪个专家或模型处理 | PiERN[53] |
| 预算token | 编码目标计算预算的元token | SmartVL[17] |
| 压缩token | 代替原文的紧凑表示 | DynamicLongContext[29] |
| 进度检查token | 结构化的自评信号（reason + value） | AgentInfer[54] |

综合上述分析，token分类尚不存在统一标准，实际应用中可按需选用。智能体场景下，推荐将生命周期五分类与计费四类组合使用：前者解释token的功能结构，后者决定真实账单。

## 3　观测空间：token 消耗的统计粒度

研究者统计token消耗时所采用的单位各不相同。本文按统计单位的大小将其排列为一条从细到粗的谱系，构成观测空间的分级：粒度越细，控制越精细，但实现也越复杂；粒度越粗，越贴近实际计费口径，但无法定位消耗的具体去向。每一级既规定了在这一尺度上观测什么、如何统计，也界定了能据此做出什么控制动作（动作即"用观测空间做了什么"）。该谱系共包含八个层级，如表 8 所示；各粒度的观测—动作配对总表见 3.10 节。

表 8　观测空间的八级粒度谱系

| 层级 | 统计单位 | 回答的问题 |
|------|----------|-----------|
| token级 | 一个token | 每个token是否值得生成 |
| 步骤与动作级 | 一次推理步骤或动作 | 该步骤是否值得继续推理 |
| 调用级 | 一次 LLM API 调用 | 单次请求的计费金额 |
| 轮次级 | 一次交互轮次 | 多轮对话消耗的增长趋势 |
| 智能体级 | 一个智能体或角色 | 各角色的消耗差异 |
| 阶段级 | 一个任务阶段 | 消耗在任务阶段间的分布 |
| 会话与轨迹级 | 一个完整任务 | 单任务的总消耗 |
| 系统级 | 整个多智能体系统 | 跨智能体的通信与协调开销 |

### 3.1　token级

token级以单个token为统计与决策单位，路由或压缩决策直接嵌入自回归生成过程，每生成一个token即做出一次决策，是当前最前沿但实现最复杂的粒度。该层级的代表工作、决策内容与机制如表 9 所示。

表 9　token级代表工作

| 代表工作 | 决策内容与机制 |
|----------|----------------|
| PyroDash[52] | 每个token决定是否切换到 LLM，控制token $\tau_{off}$ 出现在自回归流中 |
| PiERN[53] | 路由器读取隐藏状态，决定每个token由 LLM 还是计算专家生成 |
| SmartVL[17] | 用 Gumbel-sigmoid 逐个决定视觉补丁是否保留 |

统计方式为在自回归循环内部根据隐藏状态或额外预测器逐位置决策，收益是获得更平滑、更可控的质量与成本曲线。

该粒度上的动作：用观测到的内部状态判断每个 token 由谁生成——难题交接大模型[52]、交给计算专家[53]；按预算决定视觉补丁去留[17]。

### 3.2　步骤与动作级

步骤与动作级以一次推理步骤或动作为单位，比token级大一层，控制点设在步骤边界，多用于思维链（Chain-of-Thought，CoT）优化与智能体动作控制。代表工作如表 10 所示。

表 10　步骤与动作级代表工作

| 代表工作 | 统计口径与决策方式 |
|----------|---------------------|
| OS-Pruner[24] | 统计每步的token数与准确率，仅在段落边界做最优停止决策 |
| TokenSqueeze[27] | 把消耗拆成步骤数与每步token数，按难度选深度并在步骤内精炼语言 |
| D-CoT[26] | 以控制标签划定的思考段为单位，统计每段token数与空答率 |
| LAR[56] | 以动作为单位统计有效决策长度，检测转移等价片段并学成潜在动作 |
| ARES[23] | 按每次调用的推理深度与token数统计，用耐心计数器驱动自适应努力度 |

统计方式为将消耗拆分为“推理步骤数与每步平均token数”的乘积，优化目标是仅当继续推理能带来收益时才继续。

该粒度上的动作：用“继续推理的期望收益 vs token 代价”决定是否停止[24]；用停滞检测决定是否升级努力度[23]；用难度决定推理深度[27]；用步级信号判断是否浪费并干预[64]；用进度决定保留哪些交互[35]；用熵决定把哪些片段学成潜在动作[56]。

### 3.3　调用级

调用级以一次 LLM API 调用为单位，是计费的最小记账单元，计费、账单核查与成本预测均按此口径进行。该层级区分不同计费类别的token，代表工作如表 11 所示。

表 11　调用级代表工作

| 代表工作 | 统计类别 |
|----------|----------|
| Coding Agents[4] | 分非缓存输入、输出、缓存创建、缓存读取四类记账 |
| InvisibleTokens[11] | 区分可见答案token与隐藏推理token |
| Tokenomics[5] | 分输入、输出、推理三类记账 |
| TokenReduction[6] | 在四类计费token之外，还统计不可归因残差 |
| CAPC[18] | 按四类计费，命中率同时是调用次数与前缀大小的函数 |
| TokenPilot[19] | 把输入token分解为缓存命中与未命中两类 |

部分工作不区分计费类别，仅统计每个请求的输入与输出token总数，适合研究输入格式、语气与路由策略对单次请求消耗的影响，如 ToneCost[16]、ONTO[55] 与 LatencyAwareRouting[43]。此外，近年出现三种口径变体：按每任务聚合token后以固定价表折算美元（HarnessEffect[50]）、以模型调用价格比而非token数作为成本单位（ConformalCascade[46]）、以方法为单题产生的全部完成token作为成本（SampleMore[7]）。这些变体的共同特点是不再只看token数量，而是考察“token数、单价与调用次数”的乘积结构。

该粒度上的动作：用命中率与交叉阈值决定压缩策略与压缩比[18]；用访问频率与命中率决定保留/驱逐[19]；用困惑度删低信息 token[20][21]；用 NAP 一致性决定采纳哪个压缩[22]；用便宜模型质量决定是否升级[47]；用预算与消耗决定路由[45][46]；用 token 消耗做编译期预算保证[61]。

### 3.4　轮次级

轮次级以一次交互轮次为单位，监控多轮之间消耗的增长趋势，多用于对话与记忆检索类系统。代表工作如表 12 所示。

表 12　轮次级代表工作

| 代表工作 | 统计指标 |
|----------|----------|
| RCR-Router[42] | 每轮token消耗 |
| SimpleMem[31] | 每轮检索深度 $k \in [3, 20]$ |
| AMA[30] | 每轮检索次数 $K_r$ |
| MAGE[28] | 每轮上下文增长 |

统计方式为监控多轮之间消耗的增长速度，防止超线性膨胀；优化目标是使消耗增长为对数级或常数级，而非线性级。

该粒度上的动作：用查询意图决定检索深度与层级[30][31]；用依赖关系驱逐已完成步骤[36]；用 load-bearing 评分决定保留哪些块[37]。

### 3.5　智能体级

智能体级以单个智能体或角色为单位，是多智能体系统最主要的观测层级：为每个角色设置预算、统计实际消耗并识别差异。代表工作如表 13 所示。

表 13　智能体级代表工作

| 代表工作 | 统计指标 |
|----------|----------|
| RCR-Router[42] | 每智能体的token预算 $B_i$ |
| PSMAS[48] | 每智能体的激活状态（激活/空闲） |
| AgentInfer[54] | 大小模型之间的token分配 |

统计方式为累加所有轮次、所有智能体的消耗，优化目标是在保持任务成功率的同时使总和最小。

该粒度上的动作：用角色/阶段/新近度评分决定注入哪些记忆[42]；用扫描相位决定哪些智能体激活[48]；用进度信号决定大小模型分工[54]。

### 3.6　阶段级

阶段级以软件开发生命周期阶段或执行轨迹的时间分区为单位，是编码智能体特有的统计方式，揭示消耗在各阶段的结构性分布。代表工作如表 14 所示。

表 14　阶段级代表工作

| 代表工作 | 阶段划分 |
|----------|----------|
| Tokenomics[5] | ChatDev 内部阶段映射为 SDLC 六阶段（设计、编码、补全、评审、测试、文档） |
| Coding Agents[4] | 轨迹按时间五等分（早期到后期） |

阶段级统计最重要的发现是：智能体型软件工程的主要成本不在初始代码生成，而在自动化的精炼与验证，仅代码评审一个阶段即占约 59.4% 的token[5]。

该粒度上的动作：用阶段分布识别主要成本，据此在评审前插入人工检查点、压缩评审[5]。

### 3.7　会话与轨迹级

会话与轨迹级以完整会话或运行轨迹为单位，累计整个任务的所有消耗，是几乎所有优化工作默认的报告口径。代表工作如表 15 所示。

表 15　会话与轨迹级代表工作

| 代表工作 | 统计指标 |
|----------|----------|
| D-MEM[32] | 全会话 API token总量 |
| GAMER[33] | 全任务token消耗均值 |
| DynaGraph[49] | 全轨迹计算量 |
| HarnessEffect[50] | 每任务总token数，按固定价表折算美元 |

统计方式为累计整个任务的所有消耗并归一化为单位任务效率指标，如每百万token完成任务数；优化目标是让每个token产生最大的价值。

该粒度上的动作：用子目标边界决定压缩哪些段落[28]；用成功率与成本反馈决定启用哪些技能[39][40]；用临界阈值与 shadow price 决定每查询预算与放弃[60][62][63]；用内部状态决定是否中止[59]；用轨迹质量决定是否触发监督[65]。

### 3.8　系统级

系统级观测的对象是整个多智能体系统，而不是某个单独的任务、轮次或 Agent。在多智能体应用里，多个 Agent 要相互发送消息来协作——传递中间结果、请求别人帮忙、相互辩论答案——这些跨 Agent 的通信本身就要消耗 token，而且通信量会随 Agent 数量和轮次快速增长。系统级就是统计"各 Agent 之间到底传递了多少 token、通信结构长什么样"，回答"系统的通信与协调开销是多少"。它与智能体级的区别是：智能体级看单个 Agent 花了多少，系统级看 Agent 与 Agent 之间的往来花了多少。

代表工作为 SVR-MAD[51]：它观测各 Agent 的答案在同行辩论挑战之后有多少"存活"下来（用辩论结果作为证据，而不是相信 Agent 自报的置信度，因为自报置信度在难题上正确率只有 5–34%）。该粒度上的动作：用观测到的答案存活率，决定逐轮砍掉哪些 Agent 之间的通信、只保留"经受住挑战"的 Agent，从而把通信 token 降低 61%，详见 3.10 节。

### 3.9　总览与规律

综合八个层级的分析，可以归纳出三个规律，如表 16 所示。

表 16　观测空间的三个规律

| 规律 | 内容 |
|------|------|
| 监控粒度随优化对象而定 | 优化思维链的工作监控到步骤级，细粒度路由的工作监控到token级 |
| 计费口径固定在调用级 | 无论上游如何优化，最终均以一次调用的四类计费token核算成本 |
| 报告口径固定在会话与轨迹级 | 优化效果最终以整个任务的总消耗下降幅度呈现 |

由于同任务跨运行存在 10 倍方差[4]，会话与轨迹级必须统计多次运行而非单次运行。除上述八个层级外，近年还出现两个新口径：峰值token以每个回合的峰值上下文压力为指标，对应 KV 缓存压力，与累计消耗互补[41]；成功调整成本（Cost per Success，CPS）将失败运行的消耗计入分子，防止“快速失败显得便宜”的假象[6]。

### 3.10　各观测粒度上的观测—动作配对总表

上文定义了观测空间的分级，本节把“观测”与“动作”配对：每一行给出该粒度上观测什么信息，以及用这些观测（观测空间）做出了哪些控制动作，如表 17 所示。这里区分两个概念——动作回答“用观测空间做了什么”，优化方式回答“具体用什么方法实现优化”，后者（各路线策略）见第 5 章。

表 17　各观测粒度上的观测—动作配对

| 观测粒度 | 观测内容（代表） | 该粒度上的动作：用观测做了什么（代表论文） |
|----------|------------------|--------------------------------|
| token级 | 每个token的生成决策与内部状态（PyroDash[52]、PiERN[53]） | 用内部状态判断该token由谁生成：难题交接大模型（[52]）、交给计算专家（[53]）；按预算决定视觉补丁去留（[17]） |
| 步骤与动作级 | 每步token数、段落边界、准确率与推理努力度（Ares[23]、OS-Pruner[24]、TokenSqueeze[27]） | 用“继续推理的期望收益 vs token代价”决定是否停止（[24]）；用停滞检测决定是否升级努力度（[23]）；用难度决定推理深度（[27]）；用步级信号判断是否浪费并干预（[64]）；用进度决定保留哪些交互（[35]）；用熵决定把哪些片段学成潜在动作（[56]） |
| 调用级 | 四类计费token、缓存命中、单价（Coding Agents[4]、CAPC[18]、TokenPilot[19]） | 用命中率与交叉阈值决定压缩策略与压缩比（[18]）；用访问频率与命中率决定保留/驱逐（[19]）；用困惑度决定删哪些token（[20][21]）；用NAP一致性决定采纳哪个压缩（[22]）；用便宜模型质量决定是否升级（[47]）；用预算与消耗决定路由（[45][46]）；用token消耗做编译期预算保证（[61]） |
| 轮次级 | 每轮检索次数与上下文增长（AMA[30]、SimpleMem[31]） | 用查询意图决定检索深度与层级（[30][31]）；用依赖关系决定驱逐哪些步骤（[36]）；用load-bearing评分决定保留哪些块（[37]） |
| 智能体级 | 每智能体token预算与激活状态（RCR-Router[42]、PSMAS[48]） | 用角色/阶段/新近度评分决定注入哪些记忆（[42]）；用扫描相位决定哪些智能体激活（[48]）；用进度信号决定大小模型分工（[54]） |
| 阶段级 | 各阶段token分布（Tokenomics[5]） | 用阶段分布识别主要成本，据此在评审前插入检查点、压缩评审（[5]） |
| 会话与轨迹级 | 全任务总消耗与峰值（D-MEM[32]、HarnessEffect[50]） | 用子目标边界决定压缩哪些段落（[28]）；用成功率与成本反馈决定启用哪些技能（[39][40]）；用临界阈值与shadow price决定每查询预算与放弃（[60][62][63]）；用内部状态决定是否中止（[59]）；用轨迹质量决定是否触发监督（[65]） |
| 系统级 | 跨智能体通信token（SVR-MAD[51]） | 用答案存活率决定剪掉哪些Agent通信（[51]） |

整体上，粒度越细，可做的动作越精细（token级的逐token决策、步骤级的提前中止），相应要求观测越及时；粒度越粗，动作越接近成本与质量的整体权衡（任务级预算配置、系统级通信剪枝），相应要求观测覆盖整个任务或系统。训练时手段（SlimSearcher[57]、TokenSqueeze[27]的训练成分、SmartVL[17]、PyroDash[52]）因需重训模型，不在运行时观测—动作配对的讨论范围内，其定位见 5.7 节说明。

## 4　token消耗的监测方法

### 4.1　计费口径

token消耗的监测以四类计费token为底层记账语言。按 Claude API 计费规则，单次调用的成本可表示为：

$$
\text{cost} = t_{\text{unc}} \cdot p_{\text{in}} + t_{\text{out}} \cdot p_{\text{out}} + t_{\text{cw}} \cdot 1.25 \, p_{\text{in}} + t_{\text{cr}} \cdot 0.1 \, p_{\text{in}}
$$

其中，$t_{\text{unc}}$、$t_{\text{out}}$、$t_{\text{cw}}$、$t_{\text{cr}}$ 分别为非缓存输入、输出、缓存创建与缓存读取的token数，$p_{\text{in}}$、$p_{\text{out}}$ 分别为输入与输出单价；缓存创建按名义输入价的约 1.25 倍计费，缓存读取按约 0.1 倍计费[4][23]。采用该口径重建单次运行的成本，中位误差约为 1%，说明计费口径可被正确还原[6]。

### 4.2　实证监测的三类方法

现有实证监测方法可分为三类：框架埋点、账单倒推与外部推断，各类方法的做法、代表工作与关键发现如表 18 所示。

表 18　实证监测的三类方法

| 方法 | 做法 | 代表工作 | 关键发现 |
|------|------|----------|----------|
| 框架埋点 | 在框架中加记录点，记下每次调用的token数与计费类别，再按任务阶段汇总 | Tokenomics[5]、Coding Agents[4] | 代码评审占 59.4% 的token，输入token占 53.9%，同一任务不同运行相差 10 倍 |
| 账单倒推 | 用服务商返回的真实计费金额重建成本构成 | TokenReduction[6] | 缓存创建与读取合计约占账单的 80%，且token削减并不预测成本 |
| 外部推断 | 服务商不公开内部信息时，从可见信息推断被隐藏的用量 | InvisibleTokens[11]、TokenInflation[12] | 隐藏推理token占计费的 90% 以上，且这类核查方法本身也能被绕过 |

第一类做法是给智能体框架加埋点，让框架在每次调用时自动记录token数据：Tokenomics 修改 ChatDev 框架，用 GPT-5 运行 30 个开发任务并记录完整执行轨迹[5]；Coding Agents 在 SWE-bench Verified 上进行 500 个实例、每实例 4 次独立运行的实验[4]。

第二类做法以服务商返回的真实账单金额为数据来源，从账单倒推成本构成：TokenReduction 的主实验包含 2908 次实际计费的成对运行，并提出 L1–L8 分层证据标准，从组件级压缩率到每次成功执行的成本逐级验证[6]。

第三类做法面向不公开内部细节的商业服务，这类服务只返回最终答案，用户看不到中间过程，只能从可见信息推断被隐藏的用量：InvisibleTokens 提出承诺、预测、行为与签名四类核查策略，并组织为执行、安全记录、用户核查三层框架[11]；TokenInflation 从攻击者角度对三个近期核查框架做系统性拆解，结论是问题不在具体核查工具，而在“证据由被核查方自己提供”本身[12]。

### 4.3　监测中的问题

监测过程中存在四个值得注意的问题，如表 19 所示。

表 19　监测中的问题

| 问题 | 内容 | 代表工作 |
|------|------|----------|
| token削减不等于成本削减 | 削减 38.4% 的工具输出token反而对应 6.7% 的成本上升 | TokenReduction[6] |
| 等成本比较才是公平比较 | 36 个比较中无方法显著优于重复采样，预算匹配比较应成为标准协议 | SampleMore[7] |
| 计费可能被虚报 | 恶意服务商可将账单虚增最多 1469%，账单核查须绑定到服务商无法控制的证据 | TokenInflation[12] |
| 压缩可能误伤工具链 | 检索摘要喂给期望原始输出的命令会静默产生错误答案 | TokenReduction[6] |
| 量化可能使token膨胀 | INT3/INT4量化使推理token增加（INT3最高 +292%），per-token加速被总token增加抵消 | Quantization Inflates[8] |
| 等预算对比才公平（强化） | 增强方案须与获得相同token预算的基线对比；单/多智能体对比须对齐思考token | Budget-Constrained Web[9]、SAS vs MAS[10] |

综合而言，监测的正确做法是“调用级记账、阶段级聚合、会话级报告、跨运行方差统计”，并始终以成功调整成本而非原始token数作为决策指标。

## 5　优化方式：token 消耗的具体优化方法

第 3 章定义了观测空间并给出各粒度上的动作（用观测空间做了什么）。本章回答"具体用什么方法实现 token 优化"，即优化方式：按优化对象可划分为十条路线，各路线互有侧重，实际系统中常组合使用。表 20 给出各路线及其代表策略，并标注每条路线作用的观测粒度，与第 3 章的观测—动作配对对应。

表 20　优化方式十路线总览

| 路线 | 优化对象 | 代表策略 | 作用粒度 |
|------|-----------|----------|---------|
| 上下文与输入优化 | 减少进入模型的输入token | 技能精简、缓存感知压缩、预算约束训练 | 调用级（请求内容）、步骤与动作级 |
| 推理链优化 | 减少推理模型的思考token | 最优停止、自适应深度、结构化推理 | 步骤与动作级 |
| 记忆管理优化 | 降低重复检索与上下文膨胀 | 执行状态树、语义压缩、多巴胺门控 | 轮次级、会话与轨迹级 |
| 路由与调度 | 降低选错模型造成的成本差额 | 角色感知路由、预算路由、候选集级联 | 调用级 |
| 多智能体协调 | 消除智能体之间的冗余协作 | 相位调度、动态重配置、运行框架优化 | 智能体级、系统级 |
| 模型协作与分工 | 避免简单任务使用大模型 | token级迁移、token级路由、层次化协作 | token级、智能体级 |
| 数据格式与表示 | 减少格式与结构带来的开销 | 列式格式、动作重参数化 | 调用级、步骤与动作级 |
| 训练时效率塑造 | 消除训练目标中的浪费 | 级联奖励门控、成本感知训练 | 训练时（需重训模型） |
| 执行粒度控制 | 管理调用结构与中止时机 | 质量门控、提前中止、预算分配 | 调用级、步骤级、会话与轨迹级 |
| 记忆与技能可执行化 | 减少现场解读文本的开销 | 程序化记忆、边界契约编译 | 会话与轨迹级 |

### 5.1　上下文与输入优化

该路线以减少进入模型的输入token或提高每个token的信息量为目标，代表策略如表 21 所示。

表 21　上下文与输入优化策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 技能精简 | 核心指令保持常驻，背景知识与示例改为按需加载 | 描述压缩 48%、正文压缩 39% | SkillReducer[14] |
| 代码压缩 | 依次去除空白、注释与缩进，并对短标识符做带映射表的重命名 | 输入减少 42%（精度 −12pp） | Minification[15] |
| 工具菜单过滤 | 每一步只向智能体暴露下一步因果上必要的工具 | token减少约 98%，成功率提升 53.6pp | ToolMenuBench[13] |
| 缓存感知压缩 | 让所有查询共用同一个压缩前缀，以保住缓存命中 | 对不做压缩的纯缓存节省 89.6% | CAPC[18] |
| 前缀稳定化 | 把易变字段替换为静态占位符，并把工具定义移出前缀区 | 命中率 38.7% 提升至 79.2% | TokenPilot[19] |
| 预算约束训练 | 联合控制视觉token数量、模型深度与宽度，三者共享同一个预算 | 50% 预算下七基准平均精度高 7.8pp | SmartVL[17] |
| 语气选择 | 任务内容不变时只改提示语气，选出“准确率不降、token更省”的语气 | 同一任务仅换语气，输出token差异最高达 44.3% | ToneCost[16] |
| prompt压缩 | 用小型语言模型评估每个token的困惑度，删除低信息token、保留高信息token | 最高 20× 压缩、性能基本保持 | LLMLingua[20] |
| 长上下文压缩 | 按关键信息密度与位置压缩，针对长上下文场景 | 4× token减少下问答提升 21.4% | LongLLMLingua[21] |
| 动作保持压缩 | 压缩工具返回结果，并验证压缩是否改变下一步动作 | 总token减少约 33%，任务能力持平 | CoACT[22] |

输入token主导成本这一结论由三条实证支撑：Tokenomics 中占 53.9%[5]、Coding Agents 中即使有缓存仍是输入主导[4]、Minification 中源码占修复提示约 92%[15]。需要指出，查询级压缩（每个查询前缀不同）在缓存计费下可能产生负收益，比不压缩贵 40.1%[18]。

### 5.2　推理链优化

该路线以控制推理模型的思考token为目标，避免答案已得出后仍反复验证的“过度思考”，代表策略如表 22 所示。

表 22　推理链优化策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 最优停止 | 只在继续推理的期望收益超过token代价时继续生成 | 长度减少 20% 至 60% | OS-Pruner[24] |
| 自适应深度 | 简单题目采用短思维链，难题保留长链，并对步骤语言做精炼 | token减少约 50% | TokenSqueeze[27] |
| 结构化推理 | 用控制标签组织思考结构，并通过训练让模型内化该结构 | token减少 64.7%，准确率提升 9.9pp | D-CoT[26] |
| 推理努力度自适应 | 从中等推理强度起步，只在进展停滞时升级到高强度 | 仅用 12% token获得更好质量 | ARES[23] |
| 预算感知搜索 | 让剩余预算驱动树搜索，预算越少时越趋于保守 | 5 次调用达到基线 20 次精度 | BAVT[25] |
| 等成本负结果 | 采用预算匹配比较协议，在相同token成本下对比各方法 | 36 个比较中无方法显著更优 | SampleMore[7] |

需要指出，等成本下自我反思类方法可能不如多次采样[7]。

### 5.3　记忆管理优化

该路线在信息不丢失的前提下降低重复检索与上下文膨胀的成本；Agentic Context Management[3] 把上下文管理组织为设计记忆形态、写入、划定检索范围、预取、压缩固化五个环节，代表策略如表 23 所示。

表 23　记忆管理优化策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 执行状态树 | 按子目标边界压缩已完成段落，并用分支隔离错误轨迹 | token减少 55.1% | MAGE[28] |
| 语义无损压缩 | 在写入时压缩对话，并根据查询意图动态决定检索深度 | token减少约 30 倍 | SimpleMem[31] |
| 多巴胺门控 | 按“实际结果与预期的差距”（奖励预测误差）判断哪些对话值得触发记忆进化 | token减少 80% 以上 | D-MEM[32] |
| 动作中心图 | 用图结构存储动作价值，记忆读写全程不调用 LLM | token减少约 50% | GAMER[33] |
| 程序化记忆 | 日志只追加不总结，读取时用 grep 或 Python 检索 | 计费token减少 4.2 至 5.8 倍 | PRO-LONG[38] |
| 复发才固化 | 只有语义相似的交互反复出现时才调用 LLM 固化记忆 | 构建token减少 87.3% | RecMem[34] |
| 边界压缩 | 每块文本按 $\alpha{:}1$ 比例压成记忆token，推理时只召回相关块 | 上下文 7K 扩展至 1.75M | DynamicLongContext[29] |
| 峰值削减 | 把压缩与回查封装成工具，由智能体自行决定何时压缩 | 峰值token减少 20% | ACM[41] |
| 分层记忆 | 建原文、事实知识、片段摘要三层记忆，按检索需求路由到合适粒度 | token节省约 80% | AMA[30] |
| 依赖图驱逐 | 维护执行依赖图，优先驱逐“已完成且效果已持久化”的步骤 | 80M token连续会话精度无退化 | CWL[36] |
| load-bearing驱逐 | 识别“删除会导致后续失败”的承重历史块，用背包选择保留 | 峰值上下文-52%，匹配全量历史准确率 | LRE[37] |
| 进度预测保留 | 每步预测相对进度与“是否值得记住”，只保留值得记住的交互 | 步数-26.9%，任务完成率 81.0% | PABU[35] |

记忆管理需要在四个设计点上做出选择：存储粒度（原子事实、语义块、图节点或层次结构）、检索方式（相似度、意图感知、状态树路径或强化学习门控）、压缩时机（写入时、子目标完成时、异步或语义复发时）与更新策略（全量更新、条件触发或冲突驱动）。“何时该花费 LLM 成本固化记忆”是独立的触发时机设计点：RecMem 要求语义复发[34]，D-MEM 在奖励预测误差高时触发[32]，ACM 将触发权交给模型本身[41]。

### 5.4　路由与调度

该路线将请求分配给最合适的模型或实例，以平衡质量、成本与时延。路由问题可形式化为带预算约束的选择问题：选择组件 M，使匹配质量最高，同时调用成本不超过预算 B[2]。代表策略如表 24 所示。

表 24　路由与调度策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 角色感知路由 | 按角色相关性、阶段与新近度打分，在预算内取分数最高的若干条 | token减少 11% 至 47% | RCR-Router[42] |
| 延迟感知路由 | 把精度、成本与时延放进同一个路由决策 | 效用提升最多 40% | LatencyAwareRouting[43] |
| 工作负载级预算 | 把剩余预算均摊到剩余查询，允许放弃低价值查询 | 最紧预算下高 14% | WISERouter[45] |
| 候选集级联 | 用候选答案集合的大小决定是否升级，精度保证不依赖数据分布 | 期望成本节省 43% | ConformalCascade[46] |
| 用户可控路由 | 对两个分类器做参数加权插值，让用户连续调节精度与成本 | 接近全检索精度、成本约一半 | Flare-Aug[44] |
| 成本感知级联 | 按“便宜→昂贵”逐级升级模型，成本与质量联合优化 | 以 98% 成本匹配 GPT-4 性能 | FrugalGPT[47] |

按决策时机，路由可分为生成前路由与生成后级联两类：前者便宜但可能选错，后者可靠但每查询需生成多次[2]。

### 5.5　多智能体协调与模型协作

该路线通过拓扑设计、激活调度与上下文共享减少多智能体系统的冗余消耗，代表策略如表 25 所示。

表 25　多智能体协调策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 相位调度 | 用扫描指针控制智能体激活，空闲智能体只接收压缩摘要 | token减少 27.3% | PSMAS[48] |
| 动态重配置 | 评估器监控执行置信度，局部缺口插补丁节点、严重断裂重建子图 | token减少 68.6% | DynaGraph[49] |
| 运行框架优化 | 在运行框架内统一实施前缀稳定、渐进压缩、上下文卸载与零token等待 | 成本每任务减少 41% | HarnessEffect[50] |
| 通信剪枝 | 用贝叶斯后验证据逐轮剪掉“未经受同行挑战”的Agent | 通信token-61%，准确率不变或提高 | SVR-MAD[51] |

最简单的多智能体实现让所有智能体同时激活并接收完整累积上下文，总消耗为 $O(n \times L)$，广播式架构下实际为平方级膨胀[48]。

模型协作与大小模型分工方面，代表策略如表 26 所示。

表 26　模型协作与大小模型分工策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| token级迁移 | 小模型在生成中输出 $\tau_{off}$ token，把剩余推理交给大模型续写 | $\lambda=0.6$ 时成本降低 96.4% | PyroDash[52] |
| token级路由 | 让 LLM 与高精度计算专家按token交替生成 | 延迟与能耗降低 1 至 2 个数量级 | PiERN[53] |
| 层次化协作 | 由大模型负责规划、小模型负责执行 | 端到端加速 1.3 倍 | AgentInfer[54] |

### 5.6　数据格式与表示优化

该路线把一部分 token 开销归因于表示方式本身：同样的信息用不同格式存放、同一个动作用不同粒度表示，token 数量可以相差数倍。列式格式从输入侧消除格式冗余，动作重参数化从动作表示侧缩短决策序列，代表策略如表 27 所示。

表 27　数据格式与表示优化策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 列式格式 | 字段名只声明一次，数据按行排列，用缩进代替嵌套括号 | 输入token减少 46% 至 51% | ONTO[55] |
| 动作重参数化 | 找出“转移等价”的低熵动作片段，学成潜在动作token，把逐token决策变成逐潜在动作决策 | 动作token减少 2.9% 至 27.1% | LAR[56] |

### 5.7　训练时效率塑造

该路线将效率直接写入训练目标：监督微调阶段筛选数据，强化学习阶段设计奖励，代表策略如表 28 所示。需要说明的是，本路线靠训练实现效率目标，要求能够重训模型，与已部署智能体的运行时优化是互补关系；其中 SlimSearcher 与 PyroDash 训练的对象是智能体本身（web 智能体少调工具、小模型学会在推理中途求助大模型），TokenSqueeze 则属于重训通用推理模型。只关心已部署智能体监测与运行时优化的读者，可以跳过本节。

表 28　训练时效率塑造策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 级联奖励门控 | 让正确性、工具成本与生成长度三个信号相乘作为奖励 | 工具轮次减少 17% 至 58%，精度反升 | SlimSearcher[57] |
| 成本感知训练 | 把奖励设为正确性减去 $\lambda$ 倍归一化成本 | 成本降低 96.4% | PyroDash[52] |
| 长度感知 DPO | 在偏好信号中放大“赢家更短”的权重 | token减少约 50% | TokenSqueeze[27] |

### 5.8　执行粒度控制与技能可执行化

执行粒度控制指运行时动态决定多智能体管道的调用结构：合并还是拆分调用，以及何时中止注定失败的轨迹，代表策略如表 29 所示。

表 29　执行粒度控制策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 质量门控复合执行 | 用组合分数选择执行模式，质量跌破底线时回退到更细的模式 | 输入token减少 51% | AgentCapsules[58] |
| 自适应升级阶梯 | 默认单遍执行，验证失败时才逐级升级重试方式 | token不到最省基线的一半 | FlowEvo[39] |
| 召回受控中止 | 用模型内部信号训练出的探测分类器逐轮把关，提前中止注定失败的轨迹，并全局搜索各轮的召回预算 | 90% 召回下节省 60.2% token | Doomed from the Start[59] |
| 按shadow price分配 | 让每个查询的token用到“边际收益等于统一shadow price”为止，收益不足的查询直接放弃 | 256 token预算下精度提升最高 24.0pp | Shadow Price[60] |
| 等成本采样优先 | 在固定预算下用多次采样替代自我反思 | 无方法显著更优 | SampleMore[7] |
| 编译时预算保证 | 用 Rust affine type 在编译期保证预算不可复制、不可双花 | 超支 30/30→0/30 | Token Budgets[61] |
| 任务级预算配置 | 按任务难度动态选择工具集、prompt模板、token预算与模型 | 推理准确率+31.3% | ARC[62] |
| 双层博弈预算控制 | 上层设质量目标与成本激励，下层选执行参数 | token成本-17.4% | Stackelberg[63] |
| 实时浪费检测与干预 | 运行时用LLM-free信号检测浪费，触发强制切换策略或要求证据 | 失败run中58.1% token在警告后仍消耗，干预后降至30.4% | Early Diagnosis[64] |
| 零开销运行时监督 | LLM-free过滤器在交互点检查信号，触发时启动Supervisor | token-29.68% | SupervisorAgent[65] |

需要指出，每一轮单独保证“放行 98% 的成功轨迹”，合在一起并不能保证整个任务（两道 98% 的门叠加，全局放行率只有 96%），必须做全局统计[59]。

记忆与技能的可执行化把文本记忆与技能编译成可直接调用的组件，代表策略如表 30 所示。

表 30　记忆与技能可执行化策略

| 策略 | 机制 | 效果 | 代表 |
|------|------|------|------|
| 无损程序化记忆 | 日志只追加不总结，读取时用 grep 或 Python 检索 | token减少 4.2 至 5.8 倍 | PRO-LONG[38] |
| 边界契约编译 | 把技能编译为最小可执行接口，运行时按需披露细节 | 解决阶段token减少 57.44% | SkillSmith[40] |
| 工作流转技能编译 | 把成功轨迹编译为可调用技能并入库，后续任务直接复用 | token不到最省基线的一半 | FlowEvo[39] |

## 6　主要发现与待解决的问题

### 6.1　跨路线的三条经验

综合上述分析，可以归纳出跨路线的三条经验，如表 31 所示。

表 31　跨路线的三条经验

| 经验 | 内容 | 依据 |
|------|------|------|
| 输入token主导成本 | 编码智能体的输入token占账单主体，即使有缓存也是如此 | Tokenomics[5]、Minification[15]、TokenReduction[6] |
| 运行框架是最大的成本杠杆 | 仅更换运行框架即省 41% 成本，优于更换模型（最多 36%） | HarnessEffect[50] |
| 评估必须预算匹配 | 36 个等成本比较中无方法显著优于重复采样，证据需分层验证 | SampleMore[7]、TokenReduction[6]、Budget-Constrained Web[9]、SAS vs MAS[10] |

其一，输入token主导成本。编码智能体的输入token占账单主体（即使有缓存）：Tokenomics 中占 53.9%[5]，源码占修复提示约 92%[15]。对账单的进一步拆解将这一结论修正为“进入缓存前缀的输入token主导”，真实账单中缓存创建与读取合计约占 80%[6]。因此，优化优先级应放在输入压缩与缓存，而非输出长度控制。

其二，运行框架是最大的成本杠杆。仅更换运行框架（前缀稳定、渐进压缩、上下文卸载、零token等待、失败处理）即可使每任务成本减少 41%，优于更换模型（最多 36%）[50]。压缩与清除内容时必须保持前缀稳定、不破坏缓存命中，否则优化会反噬：查询级压缩比不压缩贵 40.1%[18]，削减 38.4% token反而多花 6.7% 成本[6]。

其三，评估必须采用预算匹配比较。“方法增益”可能只是“多花了钱”的假象，36 个等成本比较中没有任何方法显著优于重复采样[7]。Budget-Constrained Web[9] 进一步要求任何增强方案必须与获得相同token预算的基线对比，SAS vs MAS[10] 要求单智能体与多智能体对比时对齐思考token预算，否则"提升"可能是预算假象。证据需分层验证：组件级压缩率、计费token、任务成功、每次成功执行的成本逐级确认[6]。

### 6.2　独立优化技术的融合趋势

现有研究呈现出独立优化技术向端到端框架集成的趋势，主要表现如表 32 所示。

表 32　独立优化技术的融合趋势

| 融合表现 | 代表工作 |
|----------|----------|
| 推理架构与系统调度协同 | AgentInfer 把协作、调度、投机解码与压缩四个组件整合进同一框架[54] |
| token与计算联合控制 | SmartVL 联合控制token数量、模型深度与模型宽度[17] |

### 6.3　待解决的问题

现有研究仍存在若干待解决的问题，主要包括以下八方面：

（1）跨平台成本标准化。各服务商的token定价与缓存折扣不同，缺少统一度量，缓存感知计费公式可作为起点[38]。

（2）质量与成本权衡的刻画。质量与成本此消彼长的边界（Pareto 前沿）尚不清楚，不同任务类型的最优token分配缺少理论刻画。

（3）长期运行token爆炸。智能体数百步任务的上下文线性增长尚未解决，智能体型编码任务输入输出比超过 150:1，是典型的通信开销[1]。

（4）token使用可预测性。模型无法预测自身使用量（Pearson 相关系数不超过 0.39），执行前预测相关系数低于 0.15[4]，预算控制需要独立于模型的机制，如对数刻度范围预测与预算警报。

（5）计费透明度与账单核查。隐藏推理token占计费 90% 以上且不可验证[11]，核对机制本身也可被绕过[12]，需要将计费绑定到服务商无法控制的证据，如可信执行环境认证、密码学证明与第三方确定性重执行。

（6）token消耗的固有随机性。同任务 10 倍方差[4]使单次成本估计不可靠，多次运行报告应成为标准做法。

（7）压缩与缓存的冲突。查询级压缩使缓存失效，压缩省下的成本被缓存写入抹掉[18]，“压缩比、命中率与轨迹长度”三者耦合，尚缺少统一框架。

（8）峰值与累计口径未统一。峰值token（对应 KV 压力）与累计消耗两个口径针对不同瓶颈，尚无换算关系[41]。

## 7　结　语

本文围绕大语言模型智能体场景下的 token 消耗问题，以“观测空间—动作—优化方式”为框架对 65 篇论文进行了系统梳理。token 分类尚未形成统一标准，生命周期五分类与计费四类是智能体场景的推荐组合；观测空间包含从 token 级到系统级的八个粒度层级，监控粒度随优化对象而定，计费口径固定在调用级，报告口径固定在会话级；各粒度上的观测—动作配对已在 3.10 节归纳，即用观测空间的信息做出控制动作；优化方式按对象可划分为十条路线，其中输入压缩、缓存感知与训练时效率塑造是当前收益最显著的三个方向；监测以调用级四类计费记账为基础，结合阶段聚合与跨运行方差统计，并须防范 token 削减不等于成本削减、等成本比较与计费虚报等问题。该框架为面向智能体系统的 token 监测与优化组件设计提供文献基础。

总之，token 消耗与成本优化研究已建立起从分类、观测、动作到优化方式的完整框架，但跨平台成本标准化、计费透明度与长期运行 token 爆炸等问题仍有待解决，这些问题将在未来相当长的时间内继续成为研究热点。

## 参考文献

[1] Chen Y, Chen J, et al. Token Economics for LLM Agents: A Dual-View Study from Computing and Economics. arXiv:2605.09104, 2026.

[2] Varangot-Reille C, Bouvard C, et al. Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in Large Language Model-Based Systems. arXiv:2502.00409, 2025.

[3] Dadhich G. Agentic Context Management: Solving Agent Memory and Cost by Treating Them as Lifecycle and Architecture Problems. arXiv:2607.21503, 2026.

[4] Anonymous. How Do Coding Agents Spend Your Money? Analyzing and Predicting Token Consumptions in Agentic Coding Tasks. ICLR 2026 under review.

[5] Salim M, Latendresse J, Khatoonabadi S H, Shihab E. Tokenomics: Quantifying Where Tokens Are Used in Agentic Software Engineering. MSR 2026.

[6] Weinberger S, Hozez A. Token Reduction Is Not Cost Reduction: An Empirical Study of End-to-End Efficiency in API-Based Coding Agents. arXiv:2607.12161, 2026.

[7] Mirzaei I. Sample More, Reflect Less: Self-Refine and Reflexion Lose to Repeated Sampling at Equal Token Cost, from 1.5B to 7B. arXiv:2607.28576, 2026.

[8] Lian X, Krichene W, Huang B, et al. Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models. arXiv:2606.25519, 2026.

[9] Hajimiri S, Aminbeidokhti M, Dolz J, et al. Are Online Skill and Memory Modules Always Worth Their Tokens? A Budget-Constrained Study of Web Agents. arXiv:2606.15017, 2026.

[10] Tran D, Kiela D. Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets. arXiv:2604.02460, 2026.

[11] Sun G, Wang Z, Zhao X, et al. Invisible Tokens, Visible Bills: The Urgent Need to Audit Hidden Operations in Opaque LLM Services. arXiv:2505.18471, 2025.

[12] Hoque S, Zhang J, Sun J, Suya F. Token Inflation: How Dishonest Providers Can Overcharge for Large Language Model Usage. arXiv:2605.30040, 2026.

[13] Babu R S, Iyer L G. ToolMenuBench: Benchmarking Tool-Menu Filtering Strategies for Reliable and Efficient LLM Agents. arXiv:2606.15508, 2026.

[14] Gao Y, et al. SkillReducer: Optimizing LLM Agent Skills for Token Efficiency. arXiv:2603.29919, 2026.

[15] Hrubec N, Cito J. Minification: Reducing Token Usage of State-in-Context Agents. ICPC 2026.

[16] Kumar A, Dobariya O. Understanding Tone-Dependent Inference Cost in Large Language Models. arXiv:2607.23915, 2026.

[17] Wang P, et al. SmartVL: Look Less, Think Faster — Joint Token-Compute Adaptation for Multimodal LLMs. ECCV 2026.

[18] Song Y. Cache-Aware Prompt Compression: A Two-Tier Cost Model for LLM API Caching. arXiv:2607.15516, 2026.

[19] Xu B, Xue Z, Chen D, et al. TokenPilot: Cache-Efficient Context Management for LLM Agents. arXiv:2606.17016, 2026.

[20] Jiang H, Wu Q, Lin C-Y, Yang Y, Qiu L. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP 2023. arXiv:2310.05736.

[21] Jiang H, Wu Q, Luo X, et al. LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression. ACL 2024. arXiv:2310.06839.

[22] Chen H, Zhu Y, Zhang Y, Li J. CoACT: Action-Preserving Observation Compression for Coding Agents. arXiv:2607.02911, 2026.

[23] Cuyckens S, et al. Ares: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents. arXiv:2607.27879, 2026.

[24] Ehab M, et al. OS-Pruner: Pruning Chains-of-Thought of Reasoning Models via Optimal Stopping. arXiv:2607.11089, 2026.

[25] Li Y, Deng W, Li J, Li X. BAVT: Spend Less, Reason Better — Budget-Aware Value Tree Search for LLM Agents. arXiv:2603.12634, 2026.

[26] Ubukata S. D-CoT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models. arXiv:2602.21786, 2026.

[27] Zhang Y, et al. TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs. NeurIPS 2025.

[28] Chen Y, et al. MAGE: Beyond Semantic Organization — Memory as Execution State Management for Long-Horizon Agents. arXiv:2606.06090, 2026.

[29] Chen Z, et al. Lychee: Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning. arXiv:2602.08382, 2026.

[30] Huang W, et al. AMA: Adaptive Memory via Multi-Agent Collaboration. arXiv:2601.20352, 2026.

[31] Liu J, et al. SimpleMem: Efficient Lifelong Memory for LLM Agents. arXiv:2601.02553, 2026.

[32] Song Y, Xin Q. D-MEM: Dopamine-Gated Agentic Memory via Reward Prediction Error Routing. arXiv:2603.14597, 2026.

[33] Zheng X, et al. GAMER: Bridging Inference-Time Scaling and Episodic Memory with Action-Centric Graphs. arXiv:2607.27415, 2026.

[34] Dai Z, Deng S, Guan S, et al. RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents. arXiv:2605.16045, 2026.

[35] Jiang H, Ge L, Cai H, Song R. PABU: Progress-Aware Belief Update for Efficient LLM Agents. arXiv:2602.09138, 2026.

[36] Semenov A, Dorofeev S. Beyond Compaction: Structured Context Eviction for Long-Horizon Agents. arXiv:2606.11213, 2026.

[37] Lia N J, Mazumder A. Learning What Not to Forget: Long-Horizon Agent Memory from a Few Kilobytes of Learning. arXiv:2606.20954, 2026.

[38] Fox A, Wang J, Rosu P, Dhingra B. PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning. arXiv:2607.20064, 2026.

[39] Ren Z, Yue L, et al. FlowEvo: Self-Evolving Agents through the Co-Evolution of Workflows and Executable Skills. arXiv:2607.21596, 2026.

[40] Xu D, Chen Z, et al. SkillSmith: Compiling Agent Skills into Boundary-Guided Runtime Interfaces. arXiv:2605.15215, 2026.

[41] Li X, Ming R, Chu M, et al. ACM: Agentic Context Management for Long Horizon Tasks. arXiv:2607.23809, 2026.

[42] Liu J, et al. RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory. arXiv:2508.04903, 2025.

[43] Patel S, et al. LatencyAwareRouting: Beyond Accuracy and Cost — Latency-Aware LLM Query Routing for Dynamic Workloads. arXiv:2607.18253, 2026.

[44] Su J, Healey J, Nakov P, Cardie C. Flare-Aug: Fast or Better? Balancing Accuracy and Cost in Retrieval-Augmented Generation with Flexible User Control. arXiv:2502.12145, 2025.

[45] Li Y, Gao Z, Lakshmanan L V S. WISERouter: LLM Routing with Workload Budget Constraint. arXiv:2607.23765, 2026.

[46] Dou Y, Lian S, Li S. Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference. arXiv:2607.25018, 2026.

[47] Chen L, Zaharia M, Zou J. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. NeurIPS 2023. arXiv:2305.05176.

[48] Dubey M. PSMAS: Phase-Scheduled Multi-Agent Systems for Token-Efficient Coordination. arXiv:2604.17400, 2026.

[49] Guo Y, et al. DynaGraph: Lightweight Multi-Model Interaction Framework via Dynamic Topological Reconfiguration. arXiv:2605.29511, 2026.

[50] Sayed Ali M, Novik A, et al. The Harness Effect: How Orchestration Design Sets the Token Economics of Enterprise Agentic AI. arXiv:2607.06906, 2026.

[51] Jiang W, Shahout R, Li M, et al. SVR-MAD: A Bayesian-Inspired Framework for Posterior-Guided Multi-Agent Debate. arXiv:2605.23099, 2026.

[52] Lyu N, et al. PyroDash: Cost-Efficient Token-Level Small-Large Language Model Collaborative Inference. arXiv:2607.20327, 2026.

[53] Xiao H, et al. PiERN: Token-Level Routing for Integrating High-Precision Computation and Reasoning. arXiv:2509.18169, 2025.

[54] Lin W, et al. AgentInfer: Towards Efficient Agents — A Co-Design of Inference Architecture and System. arXiv:2512.18337, 2025.

[55] Deekeswar H. ONTO: A Token-Efficient Columnar Notation for LLM Input Optimization. arXiv:2604.17512, 2026.

[56] Huang W, et al. LAR: Latent Action Reparameterization for Efficient Agent Inference. arXiv:2605.18597, 2026.

[57] Xie Z, Wang J, Yang D, et al. SlimSearcher: Training Efficiency-Aware Web Agents via Adaptive Reward Gating. arXiv:2606.07074, 2026.

[58] Ray A. Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines. arXiv:2605.00410, 2026.

[59] Ruan K, Huang Z, Zhou Z, et al. Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade. arXiv:2607.06503, 2026.

[60] Wan X, Zhu S, Cai J, et al. The Shadow Price of Reasoning: Economic Perspective on Optimal Budget Allocation for LLMs. ICML 2026.

[61] Khan S. Token Budgets: An Empirical Catalog of 63 LLM-Agent Budget-Overrun Incidents, with an Affine-Typed Rust Mitigation as a Case Study. arXiv:2606.04056, 2026.

[62] Taparia A, Sagar S, Senanayake R. Learning to Configure Agentic AI Systems. arXiv:2602.11574, 2026.

[63] Wang B. A Stackelberg Framework for Resource-Aware LLM Agents: Learning, Repair, and Conditional Guarantees. arXiv:2606.23026, 2026.

[64] Li X, Yan W, Wu Y, et al. Early Diagnosis of Wasted Computation in Multi-Agent LLM Systems via Failure-Aware Observability. arXiv:2606.01365, 2026.

[65] Lin F, Chen S, Fang R, Wang H, Lin T. Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems. ICLR 2026. arXiv:2510.26585.
