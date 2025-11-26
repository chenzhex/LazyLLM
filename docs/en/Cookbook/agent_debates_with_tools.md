# Agent Debate with Tools

This tutorial demonstrates how to build a multi-agent debate system in LazyLLM where agents can actively use tools. In this example, two agents with opposing viewpoints (e.g., an AI accelerationist and an AI alarmist) engage in a multi-round debate on a specified topic. Whenever factual evidence is needed, the agents will autonomously call the Google Search tool to fetch external information.

You will see how each agent performs on-the-fly retrieval, cites real web content, and adjusts its arguments based on the retrieved evidence — forming a *research-capable debate system.*

!!! abstract "In this section, you will learn the following key concepts of LazyLLM:"

    - How to use [GoogleSearch][lazyllm.tools.tools.GoogleSearch] to call the Google Custom Search API;
    - How to configure independent system prompts, stances, and reasoning behaviors for each agent;
    - How to provide reasoning capability using [OnlineChatModule][lazyllm.module.OnlineChatModule];
    - How to define a `DialogueAgentWithTools` class using [ReactAgent][lazyllm.tools.ReactAgent] to build tool-enabled agents;
    - How to design a `Moderator` to orchestrate the debate, control rounds, and generate the final summary.

Here is the polished English translation, keeping the technical tone and structure clear and professional:

## Design Overview

Our goal is to build a multi-agent debate system in which each agent can automatically invoke external tools and combine retrieved evidence with its own knowledge to defend its stance. Each agent has a distinct point of view (e.g., Accelerationist vs. Alarmist) and is able to call retrieval tools in real time to support its arguments during the debate.

The entire system is structured around four core design stages:

1. Role Specification — Each agent is assigned an independent system prompt, which defines its stance, debate objectives, reasoning style, and rules for tool usage. This ensures that different agents maintain consistent speaking styles and argumentation strategies aligned with their roles.

2. Tool Augmentation — Agents are equipped with retrieval capabilities. By integrating the `GoogleSearch` tool, agents can proactively query up-to-date information, cite sources, and avoid fabricating facts. Building on this, the `DialogueAgentWithTools` class encapsulates the logic for LLM reasoning and tool invocation, enabling a full loop of *thinking → calling a tool → receiving results → continuing the debate*.

3. Inference-Driven Reasoning — The `OnlineChatModule` provides the inference engine for each agent. The agent’s thought process (Thought), tool invocation (Tool call), and final message (Answer) are all produced by this module, ensuring transparent and traceable behavior.

4. Debate Orchestration — A Moderator coordinates turn-taking and produces the final summary. The `Moderator` controls the debate rounds, selects speakers, introduces the refined topic, and generates a structured conclusion at the end, offering users a synthesized comparison of viewpoints.

Together, these components form a four-layer architecture of *Role Definition → Tool Augmentation → LLM Reasoning → Debate Coordination*, allowing multiple agents to engage in dynamic, evidence-based debate grounded in real-world information.

The overall workflow is illustrated below:

![agent_debates_with_tools](../assets/agent_debates_with_tools.png)

## Environment Setup

### Install Dependencies

Before getting started, please install the required libraries:

```bash
pip install lazyllm requests beautifulsoup4
```

### Prepare API Key

The `GoogleSearch` tool relies on the Google Custom Search API.
Please apply for an API Key and a Search Engine ID via
[Google Developers](https://developers.google.com/custom-search/v1/overview?hl=zh-cn).

Set them as environment variables:

```bash
export GOOGLE_API_KEY="AI******"     # Your Google API Key
export GOOGLE_SEARCH_ENGINE_ID="a3******"     # Your Search Engine ID
```

Since this workflow uses an online LLM, you also need to set the API key (Qwen as an example):

```bash
export LAZYLLM_QWEN_API_KEY="sk-******"
```

> 💡 Tip: For instructions on how to apply for platform API keys, refer to the [official documentation](docs.lazyllm.ai/).

### Import Dependencies

```python
import os
import requests
from bs4 import BeautifulSoup
from typing import List, Callable
from lazyllm import OnlineChatModule
from lazyllm.tools import fc_register, ReactAgent
from lazyllm.tools.tools import GoogleSearch
```

## Code Implementation

### Define the Base Dialogue Agent

`DialogueAgent` is the fundamental dialogue agent, responsible for storing conversation history and invoking the model to generate responses.

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

### Dialogue Simulator

`DialogueSimulator` is used to orchestrate the entire multi-agent conversation process:

- Controls the dialogue rounds
- Determines the next speaker
- Broadcasts messages to all agents
- Supports injecting the Moderator’s opening message

This is the core component for building an automated debate system.

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

### Tool-Enhanced Dialogue Agent

`DialogueAgentWithTools` inherits from `DialogueAgent` but overrides the `send()` method:

- Uses `ReactAgent` as the reasoning core
- Supports calling external tools (such as Google Search)
- Generates chain-of-thought reasoning with tool use based on the system prompt and conversation history

This is the key component for enabling *AI debate + retrieval augmentation*.

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

### Custom Tool

This tool performs real web searches via the Google Custom Search API and crawls the main textual content of each result page. Agents can call it to obtain real-time evidence for citations and argument support during the debate. In LazyLLM, `@fc_register('tool')` is a decorator used to register a function as a tool that can be called by an agent.


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

### Agent Roles and Tool Configuration

This section defines the agent names, the tools available to each agent (here, all are `google_search`), the debate topic, and the logic for generating agent descriptions.

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

The descriptions of the two agents are as follows:

```bash
AI accelerationist: AI Accelerationist, you champion a future where automation and AI not only transform but elevate employment, unlocking unprecedented human potential and economic growth. Embrace the change, for it heralds an era of opportunity and innovation.

AI alarmist: AI Alarmist, you caution that automation and AI, while promising, risk displacing jobs swiftly, potentially outpacing new job creation and deepening economic disparities.
```

### Generating System Prompts

Generate a complete system prompt for each agent, including: name, stance description, tool usage rules, citation requirements, and constraints such as not fabricating references.

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

The configured system prompts are as follows:

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

### Generate Detailed Debate Questions

The `Moderator` is responsible for refining the topic and generating the opening statement for the debate.

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

The original topic and the refined topic are as follows:

```bash
Original topic:
The current impact of automation and artificial intelligence on employment

Detailed topic:
AI accelerationist, AI alarmist: How is AI and automation specifically affecting low-skilled jobs in the manufacturing sector today, and what are your contrasting views on its long-term implications for these workers?
```

### Creating Agents and the Dialogue Simulator

Create two agents with tool capabilities and set up the dialogue simulator. A turn-taking strategy is used for speaking.

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

The speeches in the multi-round debate simulation are as follows:

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

## Full Code

The complete code is as follows:

<details>
<summary>Click to expand the full code</summary>

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

## Summary

In this section, we built an intelligent debate system featuring “multi-agent opposing stances + external tools + automated debate flow.” The overall design demonstrates LazyLLM’s high extensibility in multi-agent collaboration, tool augmentation, and controllable dialogue scheduling. The core workflow includes:

- Integrating LLM reasoning with tool invocation through `DialogueAgentWithTools`, enabling agents to autonomously retrieve evidence during debates;
- Using `GoogleSearch` to provide real-time retrieval, supplying each stance with credible sources;
- Assigning independent system prompts, stance definitions, and reasoning modes, ensuring each agent has a clear point of view;
- Leveraging `DialogueSimulator` to orchestrate multi-round conversations, allowing a Moderator to guide a controlled debate process.

This approach showcases LazyLLM’s flexibility and modular advantages in building intelligent agent systems capable of retrieval, reasoning, and debate. On this foundation, further extensions are possible, such as incorporating multimodal evidence, implementing scoring judges, or introducing Critic Chains for more complex agent collaboration.
