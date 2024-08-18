# The Global LLM Challenge
Thise repository hosts code for the global LLM challenge

A first of its kind internet-scale study that aims to answer two questions: a) is there a point on a quality curve above which additional gains in LLM response quality don’t lead to increased user satisfaction? and b) is there a similar point on a quality curve below which user satisfaction is considered undesirable? 


The majority of LLM vendors are in a race with each other to one-up benchmarks like MMLU, MTBench, HellowSwag etc - designed and rated primarily by human experts. But as LLMs get deployed in applications for end users and productivity workers, there hasn’t been a clear effort to study the impact of marginal improvements in quality as perceived by every-day users (not experts). 


![Response Quality Curve Measurement](static/response-quality-curve.png)

## Table of Contents

- [High-level Setup](#high-level-setup)
- [Quality Curve Definition](#definitions)
- [Model Selection](#model-selection)
- [Task Selection](#task-selection)


## High-level setup
The experiment will be conducted via a single page web application. Users will see a greeting text, followed by a prompt and some context, and finally a randomly selected LLM response that they must rate on a likert scale of 1-5, or yes/no rating that matches the task represented in the prompt. Users will not be able to select prompts nor the LLM model that generates the response, all they will be required to do is to rate the LLM response. Users will be tracked via a cookie and be presented with another random triple (prompt, context, response) when they submit a response rating or when they refresh the page. For the same session, the user won’t be shown the same triple twice. 

In order for us to gather deeper insights from the study, we will need diversity in prompt-based tasks, diversity of task complexity, and diversity of response quality. These dimensions will help offer deeper insights into scenarios that impact user experience and satisfaction with LLM responses. In the following sections we define these dimensions in detail, starting with measuring and generating responses over a quality curve.

## Definitions
In order to run a study about human satisfaction with LLM responses, we must define quality and how to generate responses over a quality curve. There are several documented ways to measure the quality of an LLM response, including automatic scores like BLEU. ROGUE, human expert ratings, and recently LLM-assisted approaches that use few-shot and COT prompting techniques to evaluate an LLM response. Given budget and time constraints, we’ll use an LLM-assisted approach for high-quality and task-appropriate evaluation. For example, for summarization tasks we’ll use faithfulness (precision), completeness (recall), and measures like coherence and fluency. We’ll also use an LLM-assisted approach to create an aggregate measure of quality by applying a weighted measure of quality dimensions using NLP research that offers guidance on what matters most to users when evaluating text.

To build a response quality curve, we’ll use the size of the LLMs as a proxy for quality. In other words, we’ll rely on the scaling laws of LLMs that imply that smaller models lead to lower overall quality. While this won’t be precise in generating a smooth quality curve, it will match the practical spirit of our study - when using smaller models do I tradeoff satisfaction for cost? This will also be faster to implement as we’ll use one set of prompts and iterate over a set of LLMs that we have chosen as part of this study. For each response, we will use our LLM-assisted judge to evaluate responses based on the aggregated measure of quality defined above. 

## Model Selection
There isn’t any prior art that offers a reasonable framework to select models for our study. But we must select models of varying sizes that may offer insights into the two points on a quality curve, different licensing properties, and limit the number of models to reduce sparsity in collected samples. For these reasons, we limit OSS LLMs to Qwen 2-0.5B-Instruct, Qwen2-1.5B-Instruct, gemma-2-2B, Qwen2-7B-Instruct, Phi-3-small-128k-instruct, Qwen2-72B and Meta-Llama-3.1-70B. And for proprietary LLM models, we have limited our choices to Claude 3 Haiku, Claude 3.5 Sonnet, OpenAI GPT 3.5-Turbo and OpenAI GPT4-o. We have intentionally not chosen an OSS LLM with a 100B+ parameters, as the goal of the study isn’t to see which model type wins (OSS v. proprietary), but rather to gain a deeper understanding of user satisfaction with respect to size as a proxy for higher quality. 

## Tasks and Task Complexity
We are designing a study for end users, so we must be mindful of prompts they are likely to use so that we encourage them to read our study-generated prompts and rate the LLM responses. A survey (via ChatGPT, Google Search, and Gemini) revealed that users are most likely to engage an LLM for tasks like information extraction and summarization, creative tasks like writing a blog post or story, problem solving task like getting central ideas from a passage or writing business emails or brainstorming ideas to solve a problem at work/school. The following table captures a few examples of the prompts we will use in study. In total, we have generated a set of 30 prompts ranging in task complexity. The prompts and relevant context were generated by Gemini using COT reasoning techniques. For a complete list of prompts that we will use in this study, please review the Google Sheet here. 