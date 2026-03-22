import json
import time
from typing import AsyncGenerator, Any

from local_model import generate, answer_question
from tools.search import web_search, web_fetch
from tools.code import run_python
from tools.weather import get_weather
from tools.calculator import calculate

MAX_ITERATIONS = 8


# ── Simple intent router ──────────────────────────────────────────────────────
# Since our small GPT can't do function calling, we route tool use
# by detecting keywords in the user's message — then use our trained
# model to synthesize a final answer from the tool results.

def detect_intent(message: str) -> list:
    """
    Detect which tools to use based on the user's message.
    Returns a list of (tool_name, args) tuples in order.
    """
    msg = message.lower()
    tools_to_use = []

    # Weather
    weather_words = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cold', 'hot', 'climate']
    if any(w in msg for w in weather_words):
        # Extract location — look for "in <city>"
        import re
        location_match = re.search(r'\bin\s+([A-Za-z\s,]+?)(?:\?|$|today|now|currently|this)', message, re.IGNORECASE)
        location = location_match.group(1).strip() if location_match else 'Delhi'
        tools_to_use.append(('get_weather', {'location': location}))

    # Math / calculate
    calc_words = ['calculate', 'compute', 'what is', 'how much', 'solve', '=', 'plus', 'minus',
                  'multiply', 'divide', 'percent', '%', 'sqrt', 'square root', 'power']
    import re
    has_numbers = bool(re.search(r'\d', message))
    if has_numbers and any(w in msg for w in calc_words):
        # Try to extract expression
        expr_match = re.search(r'[\d\s\+\-\*\/\^\(\)\.]+', message)
        if expr_match:
            tools_to_use.append(('calculate', {'expression': expr_match.group().strip()}))

    # Code
    code_words = ['write code', 'python', 'script', 'function', 'program', 'algorithm',
                  'sort', 'fibonacci', 'factorial', 'generate', 'implement', 'code to']
    if any(w in msg for w in code_words):
        tools_to_use.append(('run_python', {'message': message}))

    # Web search — for anything factual, current, or research-based
    search_words = ['who', 'what', 'when', 'where', 'why', 'how', 'latest', 'recent',
                    'news', 'paper', 'research', 'find', 'search', 'look up', 'top',
                    'best', 'current', 'today', 'this month', 'this year']
    skip_search = any(t[0] in ['get_weather', 'calculate'] for t in tools_to_use)
    if not skip_search and any(w in msg for w in search_words):
        tools_to_use.append(('web_search', {'query': message, 'num': 5}))

    return tools_to_use


def build_code_from_request(request: str) -> str:
    """Generate Python code based on the user's request using simple templates."""
    req = request.lower()

    if 'fibonacci' in req:
        return '''
def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

nums = fibonacci(100)
print(f"First 100 Fibonacci numbers:")
for i, n in enumerate(nums):
    print(f"  F({i+1}) = {n}")
'''
    elif 'factorial' in req:
        return '''
import math
for i in range(1, 21):
    print(f"{i}! = {math.factorial(i)}")
'''
    elif 'sort' in req:
        return '''
import random
data = [random.randint(1, 100) for _ in range(20)]
print(f"Original: {data}")
print(f"Sorted:   {sorted(data)}")
print(f"Reversed: {sorted(data, reverse=True)}")
'''
    elif 'prime' in req:
        return '''
def sieve(n):
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            for j in range(i*i, n+1, i):
                primes[j] = False
    return [i for i, p in enumerate(primes) if p]

print(f"Primes up to 100: {sieve(100)}")
'''
    else:
        # Use the local model to attempt code generation
        code = generate(f"# Python code to {request}\n", max_tokens=200, temperature=0.6)
        return code if code.strip() else 'print("Hello from AgentOS!")'


# ── Main agent loop ───────────────────────────────────────────────────────────
async def run_agent(
    user_message: str,
    conversation_history: list = None,
    file_store: dict = None
) -> AsyncGenerator[dict, None]:
    """
    Hybrid agent:
    1. Detects intent → calls real tools (search, weather, calculator, code)
    2. Feeds tool results to your trained GPT to synthesize a final answer
    3. Falls back to pure GPT generation for creative/conversational tasks
    """
    if conversation_history is None:
        conversation_history = []

    iteration = 0
    tool_results_text = []
    tools_to_use = detect_intent(user_message)

    # ── Tool execution phase ──────────────────────────────────────────────────
    for tool_name, tool_args in tools_to_use:
        iteration += 1

        # Emit thinking
        thinking_map = {
            'web_search': f'I need current information. Let me search the web for: "{user_message}"',
            'get_weather': f'The user wants weather info. Fetching live data...',
            'calculate':   f'I need to compute something. Running the calculation...',
            'run_python':  f'This needs code. Writing and running Python...',
        }
        yield {
            'type': 'thinking',
            'text': thinking_map.get(tool_name, f'Using {tool_name}...'),
            'iteration': iteration
        }

        yield {
            'type': 'tool_call',
            'tool': tool_name,
            'inputs': tool_args,
            'iteration': iteration
        }

        t0 = time.time()

        # Execute tool
        try:
            if tool_name == 'web_search':
                result = await web_search(tool_args['query'], tool_args.get('num', 5))
                # Format for model context
                snippets = result.get('results', [])[:4]
                summary = '\n'.join([f"- {r['title']}: {r['snippet']}" for r in snippets])
                tool_results_text.append(f"Web search results:\n{summary}")

            elif tool_name == 'get_weather':
                result = await get_weather(tool_args['location'], tool_args.get('units', 'metric'))
                if result.get('current'):
                    cur = result['current']
                    summary = (f"{result['location']}: {cur['temp']}, {cur['description']}, "
                               f"humidity {cur['humidity']}, wind {cur['wind_speed']}")
                    tool_results_text.append(f"Weather data: {summary}")
                else:
                    result = {'error': result.get('error', 'Unknown error')}

            elif tool_name == 'calculate':
                result = calculate(tool_args['expression'])
                if result.get('success'):
                    tool_results_text.append(f"Calculation: {tool_args['expression']} = {result['result']}")
                else:
                    result = {'error': result.get('error')}

            elif tool_name == 'run_python':
                code = build_code_from_request(user_message)
                tool_args = {'code': code}
                result = run_python(code)
                if result.get('stdout'):
                    tool_results_text.append(f"Code output:\n{result['stdout'][:500]}")
                else:
                    tool_results_text.append(f"Code executed. {result.get('stderr', '')[:200]}")

            elif tool_name == 'read_file':
                if file_store and tool_args.get('filename') in file_store:
                    from tools.files import dispatch_file
                    result = dispatch_file(file_store[tool_args['filename']], tool_args['filename'])
                    tool_results_text.append(f"File content preview:\n{str(result)[:500]}")
                else:
                    result = {'error': 'File not found'}

            else:
                result = {'error': f'Unknown tool: {tool_name}'}

        except Exception as e:
            result = {'error': str(e)}

        elapsed = round(time.time() - t0, 2)

        yield {
            'type': 'tool_result',
            'tool': tool_name,
            'result': result,
            'elapsed': elapsed,
            'iteration': iteration
        }

    # ── Answer synthesis with your trained model ──────────────────────────────
    iteration += 1

    yield {
        'type': 'thinking',
        'text': 'Synthesizing answer from gathered information...',
        'iteration': iteration
    }

    # Build prompt for your GPT
    if tool_results_text:
        # Combine tool results with the question as context
        context = '\n\n'.join(tool_results_text)
        prompt = f"Based on the following information:\n{context}\n\nQuestion: {user_message}\nAnswer:"
        response = generate(prompt, max_tokens=350, temperature=0.7, top_k=35)
    else:
        # Pure generation for conversational / creative tasks
        response = answer_question(user_message)

    # Clean up response
    response = response.strip()
    if not response or len(response) < 10:
        response = generate(user_message + '\n', max_tokens=300, temperature=0.85)

    yield {'type': 'final_answer', 'text': response}
    yield {'type': 'done', 'iterations': iteration}