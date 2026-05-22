import json
import re
import ast
import os
from openai import OpenAI


def load_sap_system_prompt_text() -> str:
    default_path = "llm_interface/template/template_SAP_system.txt"
    system_prompt_path = os.getenv("SAP_SYSTEM_PROMPT_PATH", default_path)
    if not os.path.exists(system_prompt_path):
        system_prompt_path = default_path
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        return " ".join(f.readlines())


def LLM_SAP(prompts_list, llm='GPT', key='', llm_model=None):
    if isinstance(prompts_list, str):
        prompts_list = [prompts_list]
    if llm == 'Zephyr':
        result = LLM_SAP_batch_Zephyr(prompts_list, llm_model)
    elif llm == 'GPT':
        result = LLM_SAP_batch_gpt(prompts_list, key)
    return result

# Load the Zephyr model once and reuse it
def load_Zephyr_pipeline():
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    model_id = "HuggingFaceH4/zephyr-7b-beta"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Zephyr prefers specific generation parameters to stay aligned
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=512,  # you can tune this
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    return pipe
    

def LLM_SAP_batch_Zephyr(prompts_list, llm_model):
    print("### run LLM_SAP_batch with zephyr-7b-beta###")

    # Load templates
    with open('llm_interface/template/template_SAP_system_short.txt', 'r') as f:
        template_system = ' '.join(f.readlines())

    with open('llm_interface/template/template_SAP_user.txt', 'r') as f:
        template_user = ' '.join(f.readlines())

    numbered_prompts = [f"### Input {i + 1}: {p}\n### Output:" for i, p in enumerate(prompts_list)]
    prompt_user = template_user + "\n\n" + "\n\n".join(numbered_prompts)
    full_prompt = template_system + "\n\n" + prompt_user

    # Load Zephyr
    if llm_model is None:
        pipe = load_Zephyr_pipeline()
    else: 
        pipe = llm_model

    # zephyr
    # Run inference
    output = pipe(
        full_prompt,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        return_full_text=False
    )[0]["generated_text"]
    
    # Parse output
    print(f"output: {output}")
    parsed_outputs = parse_batched_llm_output(output, prompts_list)
    return parsed_outputs

def LLM_SAP_batch_gpt(prompts_list, key):
    print("### run LLM_SAP_batch with qwen/qwen3.5-35b-a3b ###")
    api_key = key or os.getenv("ROUTERAI_API_KEY", "")
    base_url = os.getenv("ROUTERAI_BASE_URL", "https://routerai.ru/api/v1")
    if not api_key:
        raise ValueError("Missing API key. Set ROUTERAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt_system = load_sap_system_prompt_text()

    with open('llm_interface/template/template_SAP_user.txt', 'r') as f:
        template_user=f.readlines()
        template_user=' '.join(template_user)

    numbered_prompts = [f"### Input {i + 1}: {p}\n### Output:" for i, p in enumerate(prompts_list)]
    prompt_user = template_user + "\n\n" + "\n\n".join(numbered_prompts)
    response = client.chat.completions.create(
        model="qwen/qwen3.5-35b-a3b",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
    )
    text = response.choices[0].message.content
    print(f"text: {text}")
    parsed_outputs = parse_batched_llm_output(text, prompts_list)

    return parsed_outputs


def parse_batched_llm_output(llm_output_text, original_prompts):
    """
    llm_output_text: raw string returned by the llm for multiple prompts
    original_prompts: list of the multiple original input strings
    """
    if llm_output_text is None:
        return [None for _ in original_prompts]
    if not isinstance(llm_output_text, str):
        llm_output_text = str(llm_output_text)
    if not llm_output_text.strip():
        return [None for _ in original_prompts]

    results = []

    # Preferred format: model echoes batched sections "### Input N:"
    outputs = [s.strip() for s in re.split(r"### Input \d+:\s*", llm_output_text) if s.strip()]

    if outputs:
        for i in range(len(original_prompts)):
            out = outputs[i] if i < len(outputs) else ""
            print(f"original_prompts: {original_prompts}")
            try:
                result = get_params_dict_SAP(out)
                results.append(result)
            except Exception as e:
                print(f"Failed to parse prompt {i+1}: {e}")
                results.append(None)
        return results

    # Fallback format: single response block without "### Input N:"
    # This commonly happens when only one prompt is requested.
    if len(original_prompts) == 1:
        print(f"original_prompts: {original_prompts}")
        parsed = get_params_dict_SAP(llm_output_text.strip())
        return [parsed]

    # Last-resort split by repeated explanation headers for multi-prompt output
    chunks = re.split(r"(?=a\.\s*Explanation:)", llm_output_text, flags=re.IGNORECASE)
    chunks = [c.strip() for c in chunks if c.strip()]
    for i in range(len(original_prompts)):
        chunk = chunks[i] if i < len(chunks) else ""
        print(f"original_prompts: {original_prompts}")
        try:
            result = get_params_dict_SAP(chunk)
            results.append(result)
        except Exception as e:
            print(f"Failed to parse prompt {i+1}: {e}")
            results.append(None)
    return results


def get_params_dict_SAP(response):
    """
    Parses the LLM output from SAP-style few-shot prompts.
    Cleans up Markdown-style code fences and returns a dict.
    """
    try:
        if response is None:
            return None
        if not isinstance(response, str):
            response = str(response)
        text = response.strip()
        if not text:
            return None

        explanation = ""
        dict_block = text

        if "a. Explanation:" in text and "b. Final dictionary:" in text:
            explanation = text.split("a. Explanation:", 1)[1].split("b. Final dictionary:", 1)[0].strip()
            dict_block = text.split("b. Final dictionary:", 1)[1].strip()
        elif "Explanation:" in text and "Final dictionary:" in text:
            explanation = text.split("Explanation:", 1)[1].split("Final dictionary:", 1)[0].strip()
            dict_block = text.split("Final dictionary:", 1)[1].strip()

        dict_str = re.sub(r"```[^\n]*\n?", "", dict_block).replace("```", "").strip()

        if "prompts_list" not in dict_str and "switch_prompts_steps" not in dict_str:
            brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not brace_match:
                return None
            dict_str = brace_match.group(0).strip()

        final_dict = ast.literal_eval(dict_str)
        if not isinstance(final_dict, dict):
            return None

        prompts = final_dict.get("prompts_list")
        switches = final_dict.get("switch_prompts_steps")
        if not isinstance(prompts, list) or not isinstance(switches, list):
            return None

        return {
            "explanation": explanation,
            "prompts_list": prompts,
            "switch_prompts_steps": switches,
        }

    except Exception as e:
        print("Parsing failed:", e)
        return None