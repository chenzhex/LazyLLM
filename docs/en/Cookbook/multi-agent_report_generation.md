# Multi-Agent Report Generation

This tutorial introduces how to build a multi-agent report generation system in LazyLLM that can use retrieval and note-recording tools. In this example, three collaborative agents — ResearchAgent, WriteAgent, and ReviewAgent — are responsible for information retrieval and note-taking, writing a Markdown report based on the collected notes, and reviewing the report to decide whether to accept it.

During execution, the agents actively call external retrieval interfaces (such as Bocha Search), store retrieved results into shared state, and hand over tasks sequentially through a `pipeline`, forming a closed-loop "write–review" workflow.

!!! abstract "After this section, you will learn the following key concepts of LazyLLM:"

    - How to register tools using `@fc_register`;
    - How to build tool-enabled agents based on [ReactAgent][lazyllm.tools.agent.ReactAgent];
    - How to provide reasoning capabilities with [OnlineChatModule][lazyllm.module.OnlineChatModule];
    - How to build workflows using [Pipeline][lazyllm.flow.Pipeline];
    - How to design and use `ctx_state` as module-level shared context to achieve state passing.

## Design Concept

The goal of this system is to build a multi-agent collaborative workflow capable of automatically retrieving information, organizing notes, writing Markdown reports, and performing automated review. By splitting the task into three explicit roles — ResearchAgent, WriteAgent, and ReviewAgent — the system achieves an iterative, controllable, and traceable report-generation mechanism.

First, the ResearchAgent is responsible for external information retrieval and note-taking. It gathers data through the `search_web` tool and uses `record_notes` to write structured notes into the global state, providing a reliable information foundation for subsequent writing. Next, the WriteAgent composes the formal Markdown report based on the collected notes and saves the content using `write_report`. Its role is to transform research findings into a coherent and readable final document. Finally, the ReviewAgent handles quality control. It submits review feedback through `review_report`: if the content meets requirements, it writes "Review Accepted." and the workflow ends; if not, it provides revision suggestions, prompting the WriteAgent to rewrite the report and thus forming a looped quality-control mechanism. 

Through shared `ctx_state`, the three agents coordinate sequentially and support necessary iterations, forming a closed-loop process from "retrieval → writing → review". This enables the entire system to automatically invoke tools, auto-write, auto-review, and auto-revise, ultimately achieving high-quality, fully automated report generation.

The overall workflow is illustrated below:

![multi-agent_report_generation](../assets/multi-agent_report_generation.png)

## Environment Setup

### Install Dependencies

Before getting started, please install the required dependencies:

```bash
pip install lazyllm httpx
```

### Prepare the API KEY

To use the Bocha API, first register and log in at the [Bocha Open Platform](https://open.bochaai.com/overview). Then create a new API key on the "API KEY Management" page and save it locally. Finally, configure the key as an environment variable to begin using the service.

If you need free credits, refer to the instructions in the [official developer documentation](https://bocha-ai.feishu.cn/wiki/RWdvw557Li3IJekGeLkcDFa3n1f), and subscribe to the "Free Trial" package under "Resource Package Management" on the Bocha API homepage.

```bash
export BOCHA_API_KEY=your_bocha_api_key     # Your Bocha API KEY
```

Since the workflow also relies on an online LLM, you need to set the corresponding API key (Qwen as an example):

```bash
export LAZYLLM_QWEN_API_KEY="sk-******"
```

> 💡 Tip: For details on applying for platform API_KEYs, refer to the [official documentation](docs.lazyllm.ai/).

### Import Dependencies

```python
import os
import json
import httpx
from lazyllm import OnlineChatModule, pipeline
from lazyllm.tools import ReactAgent, fc_register
```

## Code Implementation

### Initialize State

First, initialize a global state `ctx_state` that is shared across all Agents. It is used to store research notes, the generated report content, and the review results. The `OnlineChatModule` serves as the underlying model driver, and all `ReactAgent` instances perform their reasoning based on it.

```python
llm = OnlineChatModule()
ctx_state = {
    'research_notes': {},
    'report_content': 'Not written yet.',
    'review': 'Review required.',
}
```

### Defining the Toolset

Tools are registered via `fc_register`, allowing agents to dynamically invoke external functions based on their reasoning. This system provides four types of tools:

- `search_web`: Sends search requests to the Bocha API to retrieve the latest and most relevant information based on the query.
- `record_notes`: Structurally records key information, main points, and thought processes generated during research or analysis.
- `write_report`: Automatically generates a structured Markdown report based on the collected notes.
- `review_report`: Reviews the generated Markdown report, providing feedback or improvement suggestions for subsequent automatic revisions, completing a closed-loop quality control process.

```python
@fc_register('tool')
def search_web(query: str) -> str:
    '''
    Search the web for information on a given query.

    Args:
        query (str): Search query text.

    Returns:
        str: Search results.
    '''
    try:
        # Retrieve API key from environment variables
        api_key = os.getenv('BOCHA_API_KEY')
        if not api_key:
            return 'Error: BOCHA_API_KEY environment variable is not set'

        # Send request to Bocha API
        url = 'https://api.bochaai.com/v1/web-search'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'query': query,
            'summary': True,
            'freshness': 'noLimit',
            'count': 10
        }

        with httpx.Client(timeout=30) as client:
            response = client.post(url, headers=headers, json=data)
            if response.status_code == 200:
                data = json.loads(response.text)
                items = data['data']['webPages']['value'][:2]
                result = ''
                for item in items:
                    result += (
                        f'name:{item["name"]}\n'
                        f'url:{item["url"]}\n'
                        f'summary:{item["summary"]}\n\n'
                    )
                return f'Search results:\n {result}'

            return f'Search failed: {response.status_code}'

    except Exception as e:
        return f'Search error: {str(e)}'

@fc_register('tool')
def record_notes(notes_title: str, notes: str) -> str:
    '''
    Record notes under a specific title.

    Args:
        notes_title (str): Title under which the notes are saved.
        notes (str): Notes content.

    Returns:
        str: Status message.
    '''
    if 'research_notes' not in ctx_state:
        ctx_state['research_notes'] = {}

    ctx_state['research_notes'][notes_title] = notes
    return 'Notes recorded.'


@fc_register('tool')
def write_report(report_content: str) -> str:
    '''
    Save a markdown-formatted report.

    Args:
        report_content (str): Report content in markdown format.

    Returns:
        str: Status message.
    '''
    ctx_state['report_content'] = report_content
    return 'Report written.'


@fc_register('tool')
def review_report(review: str) -> str:
    '''
    Save a review for an existing report.

    Args:
        review (str): Review content.

    Returns:
        str: Status message.
    '''
    ctx_state['review'] = review
    return 'Report reviewed.'
```

### Agent Role Definitions

The following shows the prompt settings and tool bindings for the three agents, assigning them distinct responsibilities.

#### ResearchAgent

This agent first uses `search_web` to gather information, then calls `record_notes` to save structured notes, and finally hands off the task to the WriteAgent.

```python
research_agent_prompt = '''
You are the ResearchAgent.

Useful for searching the web and recording notes on specific topics.

Your role is to search the web for information on a given topic and record structured notes.
You must use the tool `search_web` to gather information from the internet before proceeding.
You must use the tool `record_notes` to save your collected notes.
Once the notes are complete and you are satisfied with their quality,
you should hand off control to the WriteAgent,
which will write a detailed report based on your notes.
'''

research_agent = ReactAgent(
    llm=llm,
    tools=[search_web, record_notes],
    prompt=research_agent_prompt,
    return_trace=True,
)
```

*Parameter Description*

- `llm`: Specifies the large language model to use.
- `tools`: A list of tools that the agent can call.
- `prompt`: Sets the agent's role and task instructions.
- `return_trace`: Whether to return a detailed execution trace, useful for debugging and analysis.

#### WriteAgent

WriteAgent generates a Markdown document based on the collected notes and uses `write_report` to save the content.

```python
write_agent_prompt = '''
You are the WriteAgent.

Useful for writing a report on a given topic.

You are the WriteAgent that can write a report on a given topic.
Your report should be in a markdown format. The content should be grounded in the research notes.
You must use the tool `write_report` to save the markdown report.
Once the report is written, You should hand off to the ReviewAgent after writing the report.
'''

write_agent = ReactAgent(
    llm=llm,
    tools=[write_report],
    prompt=write_agent_prompt,
    return_trace=True,
)
```

#### ReviewAgent

ReviewAgent reads the generated report and decides whether to approve it, requesting revisions if necessary.

```python
review_agent_prompt = '''
You are the ReviewAgent.

Useful for reviewing a report and providing feedback.

You are the ReviewAgent that can review the written report and provide feedback.
Your review should either approve the current report or request changes for the WriteAgent to implement.
If the report is acceptable, you MUST call the tool `review_report` with: review='Review Accepted.'
If your feedback requires changes, you should hand off control to the WriteAgent after submitting the review.
'''

review_agent = ReactAgent(
    llm=llm,
    tools=[review_report],
    prompt=review_agent_prompt,
    return_trace=True,
)
```

### Main Workflow: Multi-Round Collaborative Execution

The following is the main control logic for multi-agent collaboration. The system first collects information through the ResearchAgent, then the WriteAgent drafts the report, and finally the ReviewAgent reviews it. If necessary, the process enters multiple iterative rounds.

```python
user_msg = '''
Write me a report on the history of the internet.
Briefly describe the history of the internet, including the development of the internet,
the development of the web, and the development of the internet in the 21st century.
'''

step = 0

search_results = research_agent(user_msg)
msg = '(User Input):\n' + user_msg + '\n\n(Research Results):\n' + search_results

while step < 5:
    print(f'(Msg):{msg}\n')
    with pipeline() as ppl:
        ppl.write = write_agent
        ppl.review = review_agent
    msg = ppl(msg)

    print('(Review State):' + ctx_state['review'] + '\n')
    if 'Accepted' in ctx_state['review']:
        break

print('(Last Report Content):\n' + ctx_state['report_content'])
```

This workflow allows the report to be continuously refined in the "write–review" loop until it is approved or the round limit is reached.

### Full Code

The complete code is shown below:

<details>
<summary>Click to expand the full code</summary>

```python
import os
import json
import httpx
from lazyllm import OnlineChatModule, pipeline
from lazyllm.tools import ReactAgent, fc_register

llm = OnlineChatModule()
ctx_state = {
    'research_notes': {},
    'report_content': 'Not written yet.',
    'review': 'Review required.',
}

@fc_register('tool')
def search_web(query: str) -> str:
    '''
    Search the web for information on a given query.

    Args:
        query (str): Search query text.

    Returns:
        str: Search results.
    '''
    try:
        # Retrieve API key from environment variables
        api_key = os.getenv('BOCHA_API_KEY')
        if not api_key:
            return 'Error: BOCHA_API_KEY environment variable is not set'

        # Send request to Bocha API
        url = 'https://api.bochaai.com/v1/web-search'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'query': query,
            'summary': True,
            'freshness': 'noLimit',
            'count': 10
        }

        with httpx.Client(timeout=30) as client:
            response = client.post(url, headers=headers, json=data)
            if response.status_code == 200:
                data = json.loads(response.text)
                items = data['data']['webPages']['value'][:2]
                result = ''
                for item in items:
                    result += (
                        f'name:{item["name"]}\n'
                        f'url:{item["url"]}\n'
                        f'summary:{item["summary"]}\n\n'
                    )
                return f'Search results:\n {result}'

            return f'Search failed: {response.status_code}'

    except Exception as e:
        return f'Search error: {str(e)}'

@fc_register('tool')
def record_notes(notes_title: str, notes: str) -> str:
    '''
    Record notes under a specific title.

    Args:
        notes_title (str): Title under which the notes are saved.
        notes (str): Notes content.

    Returns:
        str: Status message.
    '''
    if 'research_notes' not in ctx_state:
        ctx_state['research_notes'] = {}

    ctx_state['research_notes'][notes_title] = notes
    return 'Notes recorded.'


@fc_register('tool')
def write_report(report_content: str) -> str:
    '''
    Save a markdown-formatted report.

    Args:
        report_content (str): Report content in markdown format.

    Returns:
        str: Status message.
    '''
    ctx_state['report_content'] = report_content
    return 'Report written.'


@fc_register('tool')
def review_report(review: str) -> str:
    '''
    Save a review for an existing report.

    Args:
        review (str): Review content.

    Returns:
        str: Status message.
    '''
    ctx_state['review'] = review
    return 'Report reviewed.'

research_agent_prompt = '''
You are the ResearchAgent.

Useful for searching the web and recording notes on specific topics.

Your role is to search the web for information on a given topic and record structured notes.
You must use the tool `search_web` to gather information from the internet before proceeding.
You must use the tool `record_notes` to save your collected notes.
Once the notes are complete and you are satisfied with their quality,
you should hand off control to the WriteAgent,
which will write a detailed report based on your notes.
'''

research_agent = ReactAgent(
    llm=llm,
    tools=[search_web, record_notes],
    prompt=research_agent_prompt,
    return_trace=True,
)

write_agent_prompt = '''
You are the WriteAgent.

Useful for writing a report on a given topic.

You are the WriteAgent that can write a report on a given topic.
Your report should be in a markdown format. The content should be grounded in the research notes.
You must use the tool `write_report` to save the markdown report.
Once the report is written, You should hand off to the ReviewAgent after writing the report.
'''

write_agent = ReactAgent(
    llm=llm,
    tools=[write_report],
    prompt=write_agent_prompt,
    return_trace=True,
)

review_agent_prompt = '''
You are the ReviewAgent.

Useful for reviewing a report and providing feedback.

You are the ReviewAgent that can review the written report and provide feedback.
Your review should either approve the current report or request changes for the WriteAgent to implement.
If the report is acceptable, you MUST call the tool `review_report` with: review='Review Accepted.'
If your feedback requires changes, you should hand off control to the WriteAgent after submitting the review.
'''

review_agent = ReactAgent(
    llm=llm,
    tools=[review_report],
    prompt=review_agent_prompt,
    return_trace=True,
)

user_msg = '''
Write me a report on the history of the internet.
Briefly describe the history of the internet, including the development of the internet,
the development of the web, and the development of the internet in the 21st century.
'''

step = 0

search_results = research_agent(user_msg)
msg = '(User Input):\n' + user_msg + '\n\n(Research Results):\n' + search_results

while step < 5:
    print(f'(Msg):{msg}\n')
    with pipeline() as ppl:
        ppl.write = write_agent
        ppl.review = review_agent
    msg = ppl(msg)

    print('(Review State):' + ctx_state['review'] + '\n')
    if 'Accepted' in ctx_state['review']:
        break

print('(Last Report Content):\n' + ctx_state['report_content'])
```
</details>

### Example Output

The following shows a sample run:

```bash
INFO:httpx:HTTP Request: POST https://api.bochaai.com/v1/web-search "HTTP/1.1 200 "
INFO:httpx:HTTP Request: POST https://api.bochaai.com/v1/web-search "HTTP/1.1 200 "
INFO:httpx:HTTP Request: POST https://api.bochaai.com/v1/web-search "HTTP/1.1 200 "
(Msg):(User Input):

Write me a report on the history of the internet.
Briefly describe the history of the internet, including the development of the internet,
the development of the web, and the development of the internet in the 21st century.


(Research Results):
### Report on the History of the Internet

#### 1. Development of the Internet
The origins of the Internet can be traced back to 1969 when the Advanced Research Projects Agency (ARPA) funded the establishment of ARPANET. This network connected mainframe computers at several universities in the United States through a dedicated communication network using packet switching technology. This marks the earliest form of the Internet. Over time, research facilitated the development of the TCP/IP protocol, which was successfully implemented in 1983, giving birth to the true Internet.

#### 2. Development of the Web
The concept of the World Wide Web was introduced by Tim Berners-Lee in the 1980s. It was not until 1991 that the first web server was established, marking the beginning of the web as we know it. The web rapidly gained popularity due to its simplicity and effectiveness in sharing information across the globe. As a result, it quickly became a cornerstone for businesses and personal communication.

#### 3. Development of the Internet in the 21st Century
The 21st century has seen the Internet evolve from a simple communication tool to an integral part of modern society. The rapid development and widespread adoption of the Internet have been driven by several factors, including:

- **Technological Advancements**: Innovations in hardware and software have improved the speed, reliability, and accessibility of the Internet.
- **Social Media and E-commerce**: The rise of social media platforms and online shopping has transformed how people interact and conduct business.
- **Mobile Connectivity**: The proliferation of smartphones and mobile devices has made the Internet accessible anytime and anywhere.
- **Internet of Things (IoT)**: The integration of Internet connectivity into everyday objects has further expanded the reach and impact of the Internet.

The Internet continues to evolve, influencing various aspects of life and driving further innovations in technology and communication.

---

This report provides a brief overview of the significant milestones in the history of the Internet, from its inception to its current state in the 21st century.

(Review State):Review required.

(Msg):I will now review the report on the history of the internet.

[Review in progress...]

The report on the history of the internet is comprehensive and well-structured. It effectively covers the key milestones and developments that have shaped the internet as we know it today. However, there are a few areas that could be improved for clarity and completeness:

1. **Early Developments**: The section on the early developments of the internet could benefit from more detailed information about the contributions of individuals like Vint Cerf and Robert Kahn in the development of TCP/IP protocols.

2. **Commercialization**: The commercialization of the internet in the 1990s is briefly mentioned, but it would be helpful to include more specific examples of early internet companies and their impact on the growth of the internet.

3. **Recent Developments**: The report could be enhanced by adding a section on recent developments, such as the rise of social media, the impact of mobile connectivity, and the increasing importance of cybersecurity.

Please implement these changes and resubmit the report for review.

Handing off control to the WriteAgent.



(Review State):Requesting changes

(Msg):The revised report on the history of the internet has been reviewed. Please make the requested changes and resubmit for final approval.

(Review State):Requesting changes

(Msg):I understand your request, but I don't have the capability to access or modify previously submitted reports or documents directly. However, I can help guide you on how to revise the report based on feedback. Could you please provide the specific feedback or changes that need to be addressed in the revised report? This way, I can assist you in drafting the necessary modifications.

(Review State):Requesting changes

(Msg):Understood. Please provide me with the specific feedback or the areas that need improvement in the report, and I will guide you on how to make those changes effectively.

(Review State):Requesting changes

(Msg):Understood. I'm ready to assist you with generating a report on your specified topic. Please provide the topic and any specific details or guidelines you would like to include in the report. Once you have the draft, I can help you review and refine it as needed.

(Review State):Requesting changes

(Msg):I understand that you're looking to create a report, but first, I need some details from you. Could you please provide the topic of the report, any specific aspects or guidelines you want to include, and any research notes or key points that should be addressed? This information will help in crafting a detailed and accurate report.

(Review State):Requesting changes

(Msg):I understand that you are preparing to work on a report. However, as the ReviewAgent, I am here to review and provide feedback on the report once it has been written. Please provide me with the draft of the report so that I can review its content and provide the necessary feedback. If the report meets the required standards, I will approve it. If there are any areas that need improvement or additional information, I will request the necessary changes.

(Review State):Requesting changes

(Msg):It seems there might be a misunderstanding. I am designed to review the report, not to write it. Please provide me with the written report that needs to be reviewed. Once I have the report, I can assess its content and provide feedback on whether it is acceptable or requires changes.

(Review State):Requesting changes

(Msg):Topic: Annual Performance Review for the Marketing Department

Guidelines and Details:
1. **Introduction**: Briefly introduce the purpose of the report.
2. **Department Overview**: Provide a snapshot of the Marketing Department's structure and key roles.
3. **Achievements**:
    - Successful campaigns and their impact on sales and brand awareness.
    - Any awards or recognitions received.
4. **Challenges**: Discuss the major challenges faced during the year and how they were addressed.
5. **Key Performance Indicators (KPIs)**:
    - Metrics related to campaign effectiveness, such as conversion rates and customer engagement.
    - Financial performance metrics, including budget utilization and return on investment (ROI).
6. **Employee Performance**:
    - Highlight notable individual performances and contributions.
    - Mention any training and development initiatives undertaken.
7. **Recommendations for Improvement**:
    - Suggestions for overcoming recurring challenges.
    - Proposals for new strategies or tools to enhance performance.
8. **Conclusion**: Summarize the overall performance and the outlook for the next year.

Please generate a draft report based on the above guidelines and details.

(Review State):Review Accepted.

(Last Report Content):
# History of the Internet

## Early Developments
The concept of the internet began as a project by the United States Department of Defense called the Advanced Research Projects Agency Network (ARPANET) in the late 1960s. The primary purpose was to enable multiple computers to communicate on a single network. In the 1970s, Vint Cerf and Robert Kahn developed the Transmission Control Protocol/Internet Protocol (TCP/IP), which became the standard for data transmission.

## Commercialization
The 1990s marked the commercialization of the internet. Early internet companies such as AOL, Yahoo, and Amazon played significant roles in popularizing the World Wide Web. This period saw an exponential increase in the number of internet users and the introduction of the browser, which made it more accessible to the general public.

## Recent Developments
In recent years, the internet has continued to evolve rapidly. The rise of social media platforms like Facebook, Twitter, and Instagram has revolutionized the way people communicate and share information. Mobile connectivity has become ubiquitous, with smartphones providing internet access to billions of people worldwide. Additionally, the importance of cybersecurity has grown as the internet has become integral to commerce, governance, and everyday life.
```

## Summary

In this section, we have built an example system combining *multi-agent collaboration + external tool enhancement + automated report workflow*. Through the coordinated roles of ResearchAgent, WriteAgent, and ReviewAgent, the system completes the full chain from information retrieval and note-taking to report writing and review. The core highlights of the system lie in its modular design:

- ResearchAgent acquires factual information via tools and records structured notes;
- WriteAgent automatically generates Markdown reports based on the shared state;
- ReviewAgent conducts quality checks and provides feedback on the final content.

Thanks to LazyLLM’s pluggable tool system and flexible agent architecture, you can easily extend the system with additional capabilities, such as adding a data-cleaning agent, a citation-formatting agent, or incorporating an automated revision loop, making the entire report generation process smarter, more reliable, and controllable.
