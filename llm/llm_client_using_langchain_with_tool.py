import json
import os
from dotenv import load_dotenv
from typing import Callable, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, HumanMessage
import json
import re
load_dotenv()


from langchain_groq import ChatGroq


class LLMClientTool:
    # Gemini
    # _base_llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     google_api_key=os.getenv("GEMINI_API_KEY"),
    # )


    # Grok
    _base_llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Best general-purpose Groq model
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    _tools: Dict[str, Callable] = {}

    MAX_STEPS = 5   # ← equivalent of max_turns

    SYSTEM_PROMPT = """
    You are a tool-using assistant.

    RULES:
    - NEVER call a tool inside another tool's arguments.
    - Call ONLY ONE tool at a time.
    - If multiple steps are required, call tools SEQUENTIALLY.
    - Wait for tool output before deciding the next step.
    - Tool arguments MUST be valid JSON.
    """

    # ----------------------------------
    # Register tool
    # ----------------------------------
    @classmethod
    def add_tool(cls, tool_fn: Callable):
        if not hasattr(tool_fn, "name"):
            raise ValueError("Tool must be decorated with @tool")
        cls._tools[tool_fn.name] = tool_fn

    # ----------------------------------
    # Ask with multi-tool planning
    # ----------------------------------
    @classmethod
    def ask(cls, user_prompt: str) -> str:
        llm = cls._base_llm.bind_tools(list(cls._tools.values()))

        messages = [
            HumanMessage(content=cls.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        for step in range(cls.MAX_STEPS):
            response = llm.invoke(messages)

            # Append model response FIRST
            messages.append(response)

            # No tools → final answer
            if not response.tool_calls:
                return response.content

            # Execute tools sequentially
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})

                if tool_name not in cls._tools:
                    raise RuntimeError(f"Tool '{tool_name}' not registered")

                tool_fn = cls._tools[tool_name]

                try:
                    tool_result = tool_fn.invoke(tool_args)
                except Exception as e:
                    tool_result = f"ERROR executing {tool_name}: {e}"

                # Append tool result
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        return "Stopped after maximum tool execution steps."



    """
    User Prompt
       ↓
    PLANNER (LLM)
       ↓
    Structured Plan (steps)
       ↓
    EXECUTOR (Python)
       ↓
    Tool Results
       ↓
    VERIFIER (LLM)
       ↓
    Final Answer
    
    
    
    
    1️⃣ Core Principles (VERY IMPORTANT)
    Rule 1: LLM plans, Python executes
    Rule 2: One tool call = one step
    Rule 3: Tool output is immutable input to next step
    Rule 4: LLM NEVER nests tools
    Rule 5: Max steps enforced by code
    
    These rules eliminate 90% of tool-calling failures.
    """



    """ 
    🧠 PHASE 1 — PLANNER
    Goal
    
    Convert natural language → explicit step plan
    
    Example prompt
    
    “Create a QR code with logo and write weather note”
    
    Planner output (JSON, not execution)
    {
      "steps": [
        {
          "tool": "generate_qr_code",
          "args": {
            "data": "www.deeplearning.com",
            "image_path": "dl_logo.jpg"
          }
        },
        {
          "tool": "get_weather_from_ip",
          "args": {}
        },
        {
          "tool": "write_txt_file",
          "args": {
            "file_path": "weather_note.txt",
            "content_from_previous_step": true
          }
        }
      ]
    }
    
    """


    @classmethod
    def extract_json(cls, text: str) -> dict:
        """
        Extract the first JSON object from an LLM response.
        Handles markdown fences and extra text.
        """
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in response")

        json_str = match.group(0)
        return json.loads(json_str)



    PLANNER_PROMPT = """
    You are a planning assistant.

    You MUST return ONLY valid JSON.
    DO NOT include explanations, markdown, or extra text.

    JSON format:
    {
      "steps": [
        {
          "tool": "tool_name",
          "args": { "key": "value" }
        }
      ]
    }

    Rules:
    - One tool per step
    - No nested tool calls
    - Use previous outputs implicitly (executor will handle it).
    - one key in args should be 'content'. 'content' should be set to true if previous result is required in current step. 
    """

    @classmethod
    def plan_steps(cls, llm, user_prompt: str, retries: int = 2) -> list[dict]:
        last_error = None

        for attempt in range(retries + 1):
            response = llm.invoke([
                HumanMessage(content=cls.PLANNER_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            raw = response.content.strip()

            try:
                plan = cls.extract_json(raw)

                if "steps" not in plan:
                    raise ValueError("Missing 'steps' key")

                return plan["steps"]

            except Exception as e:
                last_error = e

        raise RuntimeError(
            f"Planner failed after {retries + 1} attempts. Last error: {last_error}"
        )








    """
    ⚙️ PHASE 2 — EXECUTOR (MOST IMPORTANT)
    Goal
    
    Execute tools safely, sequentially, deterministically
    
    Executor Rules
    
    Enforce MAX_STEPS
    
    Validate tool exists
    
    Catch tool errors
    
    Store intermediate outputs
    
    NEVER let LLM execute tools directly
    """
    @classmethod
    def execute_plan(cls, tools: dict, steps: list[dict], max_steps=5):
        results = {}

        for i, step in enumerate(steps):
            if i >= max_steps:
                raise RuntimeError("Max execution steps exceeded")

            tool_name = step["tool"]
            args = step.get("args", {})

            if tool_name not in tools:
                raise RuntimeError(f"Unknown tool: {tool_name}")

            # Resolve dependency
            if args.pop("content", False):
                args["content"] = results[i - 1]

            try:
                result = tools[tool_name].invoke(args)
            except Exception as e:
                result = f"ERROR: {e}"

            results[i] = result

        return results











    VERIFIER_PROMPT = """
    You are a response generator.
    Given the tool execution results, produce a final answer for the user.
    """

    @classmethod
    def summarize(cls, llm, user_prompt, execution_results):
        response = llm.invoke([
            HumanMessage(content=cls.VERIFIER_PROMPT),
            HumanMessage(content=f"User request: {user_prompt}"),
            HumanMessage(content=f"Tool results: {execution_results}")
        ])
        return response.content

    @classmethod
    def robust_ask(cls, user_prompt, tools=None):
        tools = cls._tools
        llm = cls._base_llm.bind_tools(list(cls._tools.values()))
        plan = cls.plan_steps(llm, user_prompt)
        results = cls.execute_plan(tools, plan)
        return cls.summarize(llm, user_prompt, results)
