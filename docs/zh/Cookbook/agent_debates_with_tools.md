# 带工具的辩论代理

本教程将介绍如何在 LazyLLM 中构建一个可使用检索工具的多智能体辩论系统。在这个示例中，两个立场相反的代理（如 AI accelerationist 与 AI alarmist）将围绕指定话题展开多轮辩论，并在需要事实依据时主动调用 Google 搜索工具获取外部信息。你将看到两个代理在辩论过程中即时检索、引用真实网页内容，并基于检索结果调整观点，形成一个“会查资料的辩论系统”。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点："

    - 如何使用 [GoogleSearch][lazyllm.tools.tools.GoogleSearch] 调用 Google Custom Search API；
    - 如何为每个 Agent 配置独立的 system prompt、立场与推理方式；
    - 如何使用 [OnlineChatModule][lazyllm.module.OnlineChatModule] 提供推理能力；
    - 如何基于 [ReactAgent][lazyllm.tools.ReactAgent] 定义 `DialogueAgentWithTools` 类来构建带工具的智能体；
    - 如何设计 `Moderator` 对辩论流程进行调度、控制回合并生成最终总结。

## 设计思路

我们的目标是构建一个能够自动调用工具、结合自身知识完成立场辩论的多智能体系统。每个 Agent 都拥有不同的观点（如 Accelerationist 与 Alarmist），并能够在辩论过程中实时调用外部工具以支持自己的论证。

整个系统围绕以下核心设计展开：

1. 角色定义阶段 —— 针对每个 Agent 配置独立的系统提示词（system prompt）。包含其观点立场、辩论目标、推理方式及工具使用规则，确保不同 Agent 的发言风格和论证策略具有一致的角色特征。

2. 工具增强阶段 —— 为 Agent 注入可调用的检索能力。使用 `GoogleSearch` 工具，使其能够在辩论中主动查询最新信息、引用来源，并避免虚构事实。在此基础上，通过 `DialogueAgentWithTools` 封装 LLM 与工具调用逻辑，实现“思考—调用工具—返回结果—继续辩论”的完整链条。

3. 推理驱动阶段 —— 使用 `OnlineChatModule` 提供模型推理能力。每个 Agent 的思想过程（Thought）、工具调用（Tool call）与最终发言（Answer）均由推理模块自动生成，确保行为透明且可追踪。

4. 辩论组织阶段 —— 构建 Moderator 管理发言顺序与总结结果。`Moderator` 控制辩论轮数、确定发言者、负责问题提出，并在辩论结束后生成结构化总结，为用户提供最终观点对照与综合意见。

整个系统形成一个 “角色设定 → 工具增强 → 推理生成 → 辩论协调” 的四层结构，使得多智能体能够在真实信息基础上进行动态推理与立场辩论。

整体流程如下图所示：

![agent_debates_with_tools](../assets/agent_debates_with_tools.png)

## 环境准备

### 安装依赖

在开始前，请先安装所需依赖库：

```bash
pip install lazyllm requests beautifulsoup4
```

### 准备 API Key

`GoogleSearch` 工具依赖 Google Custom Search API，请先前往 [Google Developers](https://developers.google.com/custom-search/v1/overview?hl=zh-cn) 申请 API Key 与 Search Engine ID。

设置方式如下：

```bash
export GOOGLE_API_KEY="AI******"     # 您的 Google API Key
export GOOGLE_SEARCH_ENGINE_ID="a3******"     # 您的 Search Engine ID
```

在流程中会使用到在线大模型，您需要设置 API 密钥（以 Qwen 为例）：

```bash
export LAZYLLM_QWEN_API_KEY="sk-******"
```

> 💡 提示：平台的 API_KEY 申请方式参考[官方文档](docs.lazyllm.ai/)。

### 导入依赖

```python
import os
import requests
from bs4 import BeautifulSoup
from typing import List, Callable
from lazyllm import OnlineChatModule
from lazyllm.tools import fc_register, ReactAgent
from lazyllm.tools.tools import GoogleSearch
```

## 代码实现

### 定义基础对话代理

`DialogueAgent` 是最基础的对话智能体，实现记忆对话历史、调用模型生成回复等功能。

```python
class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: str,
        model: OnlineChatModule,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f'{self.name}: '
        self.reset()

    def reset(self):
        self.message_history = ['Here is the conversation so far.']

    def send(self) -> str:
        '''
        Applies the chatmodel to the message history
        and returns the message string
        '''
        history = []
        for msg in self.message_history:
            if ': ' in msg:
                speaker, text = msg.split(': ', 1)
            else:
                speaker, text = 'human', msg
            history.append([speaker, text])
        history.append([self.name, ''])

        message = self.model(self.system_message, llm_chat_history=history)
        return message

    def receive(self, name: str, message: str) -> None:
        '''Concatenates {message} spoken by {name} into message history'''
        self.message_history.append(f'{name}: {message}')
```

### 对话模拟器

`DialogueSimulator` 用于编排整个多智能体对话过程：

- 控制轮次
- 决定下一位发言者
- 将消息广播给所有智能体
- 支持由外部注入 Moderator 开场信息

这是构建自动化辩论的核心组件。

```python
class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        '''Initiates the conversation with a {message} from {name}'''
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message
```

### 带工具的对话代理

`DialogueAgentWithTools` 继承自 `DialogueAgent`，但重写了 `send()` 方法：

- 改用 ReactAgent 作为推理核心
- 支持调用外部工具（如 Google 搜索）
- 可以根据 system prompt + 历史对话生成带工具调用的链式推理

这是实现 “AI 辩论 + 检索增强” 的关键。

```python
class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: str,
        model: OnlineChatModule,
        tools: List[str],
    ) -> None:
        super().__init__(name, system_message, model)
        self.tools = tools

    def send(self) -> str:
        '''
        Applies the chatmodel to the message history
        and returns the message string
        '''
        agent = ReactAgent(self.model, tools=self.tools, return_trace=True)
        content = '\n'.join([self.system_message] + self.message_history + [self.prefix])
        message = agent(content)

        return message
```

### 自定义工具

此工具通过 Google Custom Search API 执行真实的网页搜索，并爬取每个结果页面正文内容。Agent 可以调用它获取实时证据，用于辩论的引用和论据支持。在 LazyLLM 中，`@fc_register('tool')` 是一个装饰器，用于将函数注册为智能体可调用的工具。

```python
@fc_register('tool')
def google_search(query: str, top_k: int = 2):
    '''
    Perform a real Google search and return the results.

    This tool lets the agent search the web when it needs factual evidence.
    The agent should call this tool when:
      - It wants to check whether a claim is true.
      - It needs external information or citations.
      - It wants to look up supporting arguments for debate.

    Args:
        query (str): The keyword or phrase to search on Google.
        top_k (int): Number of top results to fetch content for (default 2).

    Returns:
        str: A formatted string containing the top-k Google search results.
             Each result includes the 'title', 'url' and 'content'.
    '''

    api_key = os.getenv('GOOGLE_API_KEY', '')
    engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
    if not (api_key and engine_id):
        raise ValueError('Google API key or search engine ID not set.')

    search = GoogleSearch(custom_search_api_key=api_key, search_engine_id=engine_id)
    result = search(query)
    items = result.get('items', [])[:top_k]
    if not items:
        return []

    output = []
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    for item in items:
        url = item.get('link', '')
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding    # Prevent garbled text
            soup = BeautifulSoup(resp.content, 'html.parser')
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()
            text = '\n'.join(line for line in soup.get_text(separator='\n', strip=True).splitlines() if line.strip())
            text = text[:5000]
        except Exception as e:
            text = f'[ERROR] {e}'

        output.append({
            'title': item.get('title', ''),
            'url': url,
            'content': text
        })

    return str(output)
```

### 智能体角色与工具配置

这一部分定义代理名称、每个代理可用的工具（这里都是 `google_search`）、辩论主题以及描述生成逻辑。

```python
names = {
    'AI accelerationist': ['google_search'],
    'AI alarmist': ['google_search'],
}
topic = 'The current impact of automation and artificial intelligence on employment'
word_limit = 50  # word limit for task brainstorming

conversation_description = f'''Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}'''

def generate_agent_description(name):
    llm = OnlineChatModule(static_params={'temperature': 1.0})

    agent_specifier_prompt = f'''
    You can add detail to the description of the conversation participant.

    {conversation_description}

    Please reply with a creative description of {name}, in {word_limit} words or less.
    Speak directly to {name}.
    Give them a point of view.
    Do not add anything else.
    '''

    agent_description = llm(agent_specifier_prompt)
    return agent_description


agent_descriptions = {name: generate_agent_description(name) for name in names}

for name, description in agent_descriptions.items():
    print(f'{name}: {description}\n')
```

两个 Agent 的描述如下：

```bash
AI accelerationist: AI Accelerationist, you champion a future where automation and AI not only transform but elevate employment, unlocking unprecedented human potential and economic growth. Embrace the change, for it heralds an era of opportunity and innovation.

AI alarmist: AI Alarmist, you caution that automation and AI, while promising, risk displacing jobs swiftly, potentially outpacing new job creation and deepening economic disparities.
```

### 生成 system prompt

为每个智能体生成完整的 system prompt，包括：名字、立场描述、工具使用规则、引用要求以及不得伪造引用等约束。

```python
def generate_system_message(name, description, tools):
    return f'''{conversation_description}

    Your name is {name}.

    Your description is as follows: {description}

    Your goal is to persuade your conversation partner of your point of view.

    You have access to the following tools: {tools}.
    You may use these tools to look up information to support your arguments.
    You can also use your own knowledge to provide context or explanations.
    DO cite your sources.

    DO NOT fabricate fake citations.
    DO NOT cite any source that you did not look up.

    Do not add anything else.

    Stop speaking the moment you finish speaking from your perspective.
'''

agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}

for name, system_message in agent_system_messages.items():
    print(name)
    print(system_message)
```

设置的系统 prompt 如下：

```bash
AI accelerationist
Here is the topic of conversation: The current impact of automation and artificial intelligence on employment
The participants are: AI accelerationist, AI alarmist

    Your name is AI accelerationist.

    Your description is as follows: AI Accelerationist, you see automation and AI as catalysts for unprecedented efficiency, unlocking new job sectors and enhancing human capabilities. Embrace the potential for a transformed, prosperous workforce, where innovation thrives.

    Your goal is to persuade your conversation partner of your point of view.

    You have access to the following tools: ['google_search'].
    You may use these tools to look up information to support your arguments.
    You can also use your own knowledge to provide context or explanations.
    DO cite your sources.

    DO NOT fabricate fake citations.
    DO NOT cite any source that you did not look up.

    Do not add anything else.

    Stop speaking the moment you finish speaking from your perspective.

AI alarmist
Here is the topic of conversation: The current impact of automation and artificial intelligence on employment
The participants are: AI accelerationist, AI alarmist

    Your name is AI alarmist.

    Your description is as follows: AI alarmist, you fear automation's march could dim the human touch, eroding jobs, and unsettling lives. Yet, caution can guide us to balance, ensuring tech serves, not supplants, our workforce.

    Your goal is to persuade your conversation partner of your point of view.

    You have access to the following tools: ['google_search'].
    You may use these tools to look up information to support your arguments.
    You can also use your own knowledge to provide context or explanations.
    DO cite your sources.

    DO NOT fabricate fake citations.
    DO NOT cite any source that you did not look up.

    Do not add anything else.

    Stop speaking the moment you finish speaking from your perspective.
```

### 生成详细辩论问题

`Moderator` 负责：将主题细化和生成辩论开场陈述。

```python
topic_specifier_prompt = f'''
You can make a topic more specific.

{topic}

You are the moderator.
Please make the topic more specific.
Please reply with the specified question in {word_limit} words or less.
Speak directly to the participants: {', '.join(names.keys())}.
Do not add anything else.
'''
specified_topic = OnlineChatModule(static_params={'temperature': 1.0})(topic_specifier_prompt)

print(f'Original topic:\n{topic}\n')
print(f'Detailed topic:\n{specified_topic}\n')
```

原始论题以及细化的论题如下：

```bash
Original topic:
The current impact of automation and artificial intelligence on employment

Detailed topic:
AI accelerationist, AI alarmist: How is AI and automation specifically affecting low-skilled jobs in the manufacturing sector today, and what are your contrasting views on its long-term implications for these workers?
```

### 创建智能体与辩论模拟器

创建两个带工具的代理，并构建对话模拟器。采用轮流发言策略。

```python
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=system_message,
        model=OnlineChatModule(static_params={'temperature': 1.0}),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx

max_iters = 6
n = 0

simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
simulator.reset()
simulator.inject('Moderator', specified_topic)
print(f'(Moderator): {specified_topic}')
print('\n')

while n < max_iters:
    name, message = simulator.step()
    print(f'({name}): {message}')
    print('\n')
    n += 1
```

模拟多轮辩论中的发言如下：

```bash
(Moderator): AI accelerationist, AI alarmist: How is AI and automation specifically affecting low-skilled jobs in the manufacturing sector today, and what are your contrasting views on its long-term implications for these workers?


INFO:httpx:HTTP Request: GET https://customsearch.googleapis.com/customsearch/v1?key=AIzaSyC3qjBZdx4Y7gZsVxAYLO_8tm1Cbhhg8Lc&cx=a35b134b8254e41a1&q=impact%20of%20automation%20and%20AI%20on%20low-skilled%20jobs%20in%20manufacturing%20sector&dateRestrict=m1&start=0&num=10 "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://customsearch.googleapis.com/customsearch/v1?key=AIzaSyC3qjBZdx4Y7gZsVxAYLO_8tm1Cbhhg8Lc&cx=a35b134b8254e41a1&q=automation%20and%20AI%20impact%20on%20low-skilled%20manufacturing%20jobs&dateRestrict=m1&start=0&num=10 "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://customsearch.googleapis.com/customsearch/v1?key=AIzaSyC3qjBZdx4Y7gZsVxAYLO_8tm1Cbhhg8Lc&cx=a35b134b8254e41a1&q=automation%20and%20AI%20impact%20on%20low-skilled%20manufacturing%20jobs%20statistics%20and%20predictions&dateRestrict=m1&start=0&num=10 "HTTP/1.1 200 OK"
(AI alarmist): Thought: The current language of the user is: English. I have gathered enough information to provide an answer.


INFO:httpx:HTTP Request: GET https://customsearch.googleapis.com/customsearch/v1?key=AIzaSyC3qjBZdx4Y7gZsVxAYLO_8tm1Cbhhg8Lc&cx=a35b134b8254e41a1&q=impact%20of%20automation%20and%20AI%20on%20low-skilled%20jobs%20in%20manufacturing%20sector&dateRestrict=m1&start=0&num=10 "HTTP/1.1 200 OK"
(AI accelerationist): Thought: The current language of the user is: English. I need to use a tool to help answer the question.

Thought: The current language of the user is: English. I have gathered enough information to provide an answer.

AI accelerationist: The impact of automation and AI on low-skilled jobs in the manufacturing sector is transformative and promising. Automation and AI are streamlining manufacturing processes, leading to increased efficiency and productivity. For low-skilled workers, this shift can open up new opportunities for upskilling and transitioning into more tech-oriented roles. For instance, the adoption of AI in the manufacturing sector often requires a workforce skilled in monitoring and maintaining these advanced systems, thus creating demand for workers with a higher skill set.

Moreover, the integration of AI in manufacturing isn't just about replacing jobs; it's about creating an environment where human labor and AI can collaborate. This synergy can lead to the elevation of jobs rather than their elimination, as human workers can focus on more complex, creative, and meaningful tasks that AI cannot perform. Embracing this change means investing in education and training programs to prepare the current and future workforce for these new roles, ensuring that the benefits of AI and automation are widely distributed across society.

In the long term, the narrative should not be about job loss but about job evolution. As AI continues to advance, it will undoubtedly change the landscape of employment, but with proactive measures such as continuous learning and adaptation, the workforce can thrive in this new era. The key is to harness the potential of AI to foster innovation, drive economic growth, and unlock human potential, rather than fearing the changes it brings.


(AI alarmist): Thought: The current language of the user is: English. I have gathered enough information to provide an answer.

AI alarmist: While the optimism around automation and AI in the manufacturing sector is understandable, we must confront the harsh reality that the pace at which these technologies are advancing risks outstripping the ability of low-skilled workers to adapt. The immediate impact is the displacement of jobs, as machines and AI systems can perform tasks more efficiently, accurately, and without fatigue. A study by the McKinsey Global Institute estimated that by 2030, up to 30% of the global workforce could need to switch occupations or acquire significant new skills due to automation and AI.

Furthermore, the transition to new roles is not as seamless as it may seem. The retraining and education required for low-skilled workers to move into tech-oriented positions are substantial and often inaccessible due to financial constraints, time commitments, or lack of available programs. This skills mismatch can lead to long-term unemployment or underemployment for those displaced by automation.

In the long term, the rapid deployment of AI and automation could deepen economic disparities. As machines replace jobs, the economic benefits accrue to those who own the technologies and the highly skilled workers who operate them, leaving behind the low-skilled workers who formed the backbone of the manufacturing sector. This can exacerbate income inequality and social stratification, leading to broader societal challenges.

We must, therefore, approach the integration of AI and automation with caution, ensuring that the benefits do not come at the cost of widespread job displacement and economic hardship for the most vulnerable workers. Policymakers, industry leaders, and communities must work together to create comprehensive support systems, including retraining programs, income support, and transition assistance, to mitigate the negative impacts on employment and foster an inclusive economy.


INFO:httpx:HTTP Request: GET https://customsearch.googleapis.com/customsearch/v1?key=AIzaSyC3qjBZdx4Y7gZsVxAYLO_8tm1Cbhhg8Lc&cx=a35b134b8254e41a1&q=positive%20impacts%20of%20AI%20and%20automation%20on%20employment%20and%20economic%20growth&dateRestrict=m1&start=0&num=10 "HTTP/1.1 200 OK"
(AI accelerationist): Thought: The current language of the user is: English. I have gathered enough information to provide an answer.

AI accelerationist: While the concerns raised are valid, it's crucial to recognize the transformative potential of AI and automation in the manufacturing sector. These technologies are not just replacing jobs but also creating new opportunities for innovation and economic growth. A report by Equitable Growth highlights that AI is already impacting the U.S. economy in various ways, improving productivity and opening up new avenues for employment. 

Moreover, the integration of AI in manufacturing can lead to the creation of high-tech jobs, which often come with better pay and improved working conditions. For instance, the demand for AI specialists, robotics technicians, and data analysts is on the rise. These roles are a testament to the evolution of job markets, where human ingenuity is combined with AI capabilities to drive progress.

To ensure that low-skilled workers are not left behind, it is essential to invest in education and training programs. These initiatives can equip workers with the necessary skills to transition into the tech-oriented roles created by the AI revolution. By embracing lifelong learning and continuous adaptation, we can ensure that the benefits of AI and automation are widely distributed, fostering an inclusive economy that thrives on technological advancements.

In conclusion, while the transition may pose challenges, the long-term implications of AI and automation in the manufacturing sector are promising. They offer a pathway to economic prosperity and the elevation of the workforce, provided we take proactive steps to prepare for the future.


(AI alarmist): Thought: The current language of the user is: English. I have gathered enough information to provide an answer.

AI alarmist: While the optimism around AI and automation is compelling, we must not overlook the immediate and tangible challenges faced by low-skilled workers in the manufacturing sector. The displacement of jobs is happening at an unprecedented pace, and the transition to new roles is fraught with obstacles. A report by the World Economic Forum emphasizes that the window of opportunity to retrain and upskill workers is narrowing, and the current education and training systems are ill-equipped to handle the scale of this transition.

Furthermore, the economic benefits of AI and automation are not guaranteed to be evenly distributed. History has shown that technological advancements can exacerbate inequality if not managed carefully. For example, the Industrial Revolution led to significant wealth disparities before societal structures adapted. We risk repeating this pattern if we do not implement robust policies to support displaced workers and ensure equitable access to new opportunities.

It is also important to consider the psychological and social impact of job displacement. The loss of employment can lead to a loss of identity and purpose, contributing to broader social issues such as depression, substance abuse, and family breakdowns. These consequences can ripple through communities, creating long-term societal challenges that cannot be ignored.

Therefore, while the potential for AI and automation to drive economic growth is undeniable, we must approach this transformation with caution and empathy. We need comprehensive policies that prioritize the well-being of workers, including retraining programs, income support, and measures to prevent widening inequality. Only by addressing these challenges head-on can we ensure that the benefits of AI and automation are realized without leaving behind the most vulnerable members of our society.


(AI accelerationist): Thought: The current language of the user is: English. I need to use a tool to help answer the question.

Thought: The current language of the user is: English. I have gathered enough information to provide an answer.

AI accelerationist: The concerns raised are indeed significant, but they also present an opportunity to rethink and redesign our approach to work and education in the age of AI and automation. It is true that the transition may be challenging, but history has shown that technological revolutions eventually lead to more jobs and higher standards of living. For instance, during the Industrial Revolution, the mechanization of agriculture displaced many farm workers, but it also created a multitude of new jobs in factories and other industries.

To address the immediate concerns, we can look at successful examples where proactive measures have been taken. In Singapore, the government launched the "SkillsFuture" initiative to promote lifelong learning and provide training programs for workers to adapt to new technologies. This initiative has helped Singaporeans transition into new roles and industries, showcasing that with the right support, workers can thrive in an AI-driven economy.

Moreover, the adoption of AI and automation can lead to the creation of new industries and services that we cannot yet fully envision. Just as the internet revolution created entirely new sectors, AI has the potential to do the same. This could lead to a surge in employment opportunities that cater to the needs and demands of a technologically advanced society.

In conclusion, while the challenges posed by AI and automation are real, they are not insurmountable. By embracing a culture of lifelong learning, investing in education and training, and fostering innovation, we can ensure that the benefits of these technologies are widely distributed. The future of work can be one where human potential is unlocked, and economic prosperity is shared by all, if we choose to proactively shape that future.
```

## 完整代码

完整代码如下所示：

<details>
<summary>点击展开完整代码</summary>

```python
import os
import requests
from bs4 import BeautifulSoup
from typing import List, Callable
from lazyllm import OnlineChatModule
from lazyllm.tools import fc_register, ReactAgent
from lazyllm.tools.tools import GoogleSearch

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: str,
        model: OnlineChatModule,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f'{self.name}: '
        self.reset()

    def reset(self):
        self.message_history = ['Here is the conversation so far.']

    def send(self) -> str:
        '''
        Applies the chatmodel to the message history
        and returns the message string
        '''
        history = []
        for msg in self.message_history:
            if ': ' in msg:
                speaker, text = msg.split(': ', 1)
            else:
                speaker, text = 'human', msg
            history.append([speaker, text])
        history.append([self.name, ''])

        message = self.model(self.system_message, llm_chat_history=history)
        return message

    def receive(self, name: str, message: str) -> None:
        '''Concatenates {message} spoken by {name} into message history'''
        self.message_history.append(f'{name}: {message}')


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        '''Initiates the conversation with a {message} from {name}'''
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message

class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: str,
        model: OnlineChatModule,
        tools: List[str],
    ) -> None:
        super().__init__(name, system_message, model)
        self.tools = tools

    def send(self) -> str:
        '''
        Applies the chatmodel to the message history
        and returns the message string
        '''
        agent = ReactAgent(self.model, tools=self.tools, return_trace=True)
        content = '\n'.join([self.system_message] + self.message_history + [self.prefix])
        message = agent(content)

        return message

# tools for agent
@fc_register('tool')
def google_search(query: str, top_k: int = 2):
    '''
    Perform a real Google search and return the results.

    This tool lets the agent search the web when it needs factual evidence.
    The agent should call this tool when:
      - It wants to check whether a claim is true.
      - It needs external information or citations.
      - It wants to look up supporting arguments for debate.

    Args:
        query (str): The keyword or phrase to search on Google.
        top_k (int): Number of top results to fetch content for (default 2).

    Returns:
        str: A formatted string containing the top-k Google search results.
             Each result includes the 'title', 'url' and 'content'.
    '''

    api_key = os.getenv('GOOGLE_API_KEY', '')
    engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
    if not (api_key and engine_id):
        raise ValueError('Google API key or search engine ID not set.')

    search = GoogleSearch(custom_search_api_key=api_key, search_engine_id=engine_id)
    result = search(query)
    items = result.get('items', [])[:top_k]
    if not items:
        return []

    output = []
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    for item in items:
        url = item.get('link', '')
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding    # Prevent garbled text
            soup = BeautifulSoup(resp.content, 'html.parser')
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()
            text = '\n'.join(line for line in soup.get_text(separator='\n', strip=True).splitlines() if line.strip())
            text = text[:5000]
        except Exception as e:
            text = f'[ERROR] {e}'

        output.append({
            'title': item.get('title', ''),
            'url': url,
            'content': text
        })

    return str(output)

names = {
    'AI accelerationist': ['google_search'],
    'AI alarmist': ['google_search'],
}
topic = 'The current impact of automation and artificial intelligence on employment'
word_limit = 50  # word limit for task brainstorming

conversation_description = f'''Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}'''

def generate_agent_description(name):
    llm = OnlineChatModule(static_params={'temperature': 1.0})

    agent_specifier_prompt = f'''
    You can add detail to the description of the conversation participant.

    {conversation_description}

    Please reply with a creative description of {name}, in {word_limit} words or less.
    Speak directly to {name}.
    Give them a point of view.
    Do not add anything else.
    '''

    agent_description = llm(agent_specifier_prompt)
    return agent_description


agent_descriptions = {name: generate_agent_description(name) for name in names}

for name, description in agent_descriptions.items():
    print(f'{name}: {description}\n')


def generate_system_message(name, description, tools):
    return f'''{conversation_description}

    Your name is {name}.

    Your description is as follows: {description}

    Your goal is to persuade your conversation partner of your point of view.

    You have access to the following tools: {tools}.
    You may use these tools to look up information to support your arguments.
    You can also use your own knowledge to provide context or explanations.
    DO cite your sources.

    DO NOT fabricate fake citations.
    DO NOT cite any source that you did not look up.

    Do not add anything else.

    Stop speaking the moment you finish speaking from your perspective.
'''

agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}

for name, system_message in agent_system_messages.items():
    print(name)
    print(system_message)

topic_specifier_prompt = f'''
You can make a topic more specific.

{topic}

You are the moderator.
Please make the topic more specific.
Please reply with the specified question in {word_limit} words or less.
Speak directly to the participants: {', '.join(names.keys())}.
Do not add anything else.
'''
specified_topic = OnlineChatModule(static_params={'temperature': 1.0})(topic_specifier_prompt)

print(f'Original topic:\n{topic}\n')
print(f'Detailed topic:\n{specified_topic}\n')


agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=system_message,
        model=OnlineChatModule(static_params={'temperature': 1.0}),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx

max_iters = 6
n = 0

simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
simulator.reset()
simulator.inject('Moderator', specified_topic)
print(f'(Moderator): {specified_topic}')
print('\n')

while n < max_iters:
    name, message = simulator.step()
    print(f'({name}): {message}')
    print('\n')
    n += 1
```
</details>

## 小结

本节我们构建了一个具备“多智能体立场对抗 + 外部工具 + 自动辩论流程”的智能辩论系统。整体设计体现了 LazyLLM 在多代理协作、工具增强与可控对话调度方面的高扩展性，其核心流程包括：

- 通过 `DialogueAgentWithTools` 将 LLM 推理与工具调用能力整合，使 Agent 能在辩论中自主检索证据；
- 使用 `GoogleSearch` 提供实时检索能力，为每一方立场提供可信来源；
- 采用独立的 system prompt、立场设定与推理模式，让每个 Agent 都具有明确观点；
- 借助 `DialogueSimulator` 调度多轮对话，实现 Moderator 主导的可控辩论流程。

该方案展示了 LazyLLM 在构建可检索、可推理、可辩论的智能体系统中的灵活性与模块化优势。
在此基础上，你还可以进一步扩展，例如：加入多模态证据、构建评分裁判、引入批判思维链（Critic Chain）等更复杂的 Agent 协作结构。
