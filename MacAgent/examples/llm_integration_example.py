#!/usr/bin/env python3
"""
Example script demonstrating the LLM integration system.

This script shows how to use the LLMConnector, PromptManager, and InstructionProcessor
classes to interact with language models.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any

# Add the MacAgent package to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MacAgent.src.intelligence import (
    LLMConnector, LLMProvider, ModelConfig, LLMResponse,
    PromptManager, PromptStrategy,
    InstructionProcessor, Instruction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    # Initialize the LLM connector
    logger.info("Initializing LLM Connector...")
    connector = LLMConnector(
        config_path="config/api_keys.json",
        cache_dir=".cache",
        enable_cache=True
    )
    
    # Initialize the prompt manager
    logger.info("Initializing Prompt Manager...")
    prompt_manager = PromptManager(
        templates_path="config/prompt_templates",
        default_model_token_limit=4096
    )
    
    # Initialize the instruction processor
    logger.info("Initializing Instruction Processor...")
    instruction_processor = InstructionProcessor(
        llm_connector=connector,
        default_model="gpt-4",
        default_provider=LLMProvider.OPENAI
    )
    
    # Example 1: Basic LLM generation with OpenAI
    logger.info("\n--- Example 1: Basic OpenAI Generation ---")
    
    # Configure the OpenAI model
    openai_config = ModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        max_tokens=500,
        temperature=0.7
    )
    
    # Set up the conversation
    openai_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a language model is in 2-3 sentences."}
    ]
    
    # Generate a response using OpenAI
    logger.info("Generating response from OpenAI...")
    openai_response = await connector.generate(openai_messages, openai_config)
    
    # Display the response
    logger.info(f"OpenAI Response: {openai_response.text}")
    logger.info(f"Model: {openai_response.model}")
    logger.info(f"Provider: {openai_response.provider.value}")
    logger.info(f"Token usage: {openai_response.token_usage}")
    logger.info(f"Latency: {openai_response.latency:.3f} seconds")
    
    # Example 2: Basic LLM generation with Anthropic
    logger.info("\n--- Example 2: Basic Anthropic Generation ---")
    
    # Configure the Anthropic model
    anthropic_config = ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        max_tokens=500,
        temperature=0.7
    )
    
    # Set up the conversation
    anthropic_messages = [
        {"role": "user", "content": "Explain what a language model is in 2-3 sentences."}
    ]
    
    # Generate a response using Anthropic
    logger.info("Generating response from Anthropic...")
    anthropic_response = await connector.generate(anthropic_messages, anthropic_config)
    
    # Display the response
    logger.info(f"Anthropic Response: {anthropic_response.text}")
    logger.info(f"Model: {anthropic_response.model}")
    logger.info(f"Provider: {anthropic_response.provider.value}")
    logger.info(f"Token usage: {anthropic_response.token_usage}")
    logger.info(f"Latency: {anthropic_response.latency:.3f} seconds")
    
    # Example 3: Using fallback from OpenAI to Anthropic
    logger.info("\n--- Example 3: Using Provider Fallback ---")
    
    # Set up the conversation
    fallback_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
    ]
    
    # Generate a response using the primary model with fallback
    logger.info("Generating response with fallback configuration...")
    fallback_response = await connector.generate(
        fallback_messages, 
        openai_config,
        fallback_configs=[anthropic_config]
    )
    
    # Display the response
    logger.info(f"Response: {fallback_response.text}")
    logger.info(f"Model: {fallback_response.model}")
    logger.info(f"Provider: {fallback_response.provider.value}")
    logger.info(f"Token usage: {fallback_response.token_usage}")
    logger.info(f"Latency: {fallback_response.latency:.3f} seconds")
    
    # Example 4: Using prompt templates
    logger.info("\n--- Example 4: Using Prompt Templates ---")
    
    # Render a prompt using a template
    template_name = "general_assistant"
    variables = {
        "task": "explain how LLMs work in simple terms",
        "context": "The user has no background in AI or machine learning"
    }
    
    # Get the formatted prompt
    logger.info(f"Rendering prompt using template: {template_name}")
    formatted_prompt = prompt_manager.render_prompt(
        template_name=template_name,
        variables=variables,
        model_name="gpt-4"
    )
    
    # Generate a response using the formatted prompt
    logger.info("Generating response with formatted prompt...")
    template_response = await connector.generate(formatted_prompt, openai_config)
    
    # Display the response
    logger.info(f"Response: {template_response.text}")
    
    # Example 5: Instruction processing
    logger.info("\n--- Example 5: Instruction Processing ---")
    
    # Parse a natural language instruction
    instruction_text = "Find all PDF files in my Documents folder older than 3 months"
    logger.info(f"Parsing instruction: {instruction_text}")
    
    instruction = await instruction_processor.parse_instruction(instruction_text)
    
    # Display the parsed instruction
    logger.info(f"Intent: {instruction.intent.value}")
    logger.info(f"Confidence: {instruction.confidence:.2f}")
    logger.info(f"Ambiguous: {instruction.ambiguous}")
    
    # Display parameters
    if instruction.parameters:
        logger.info("Parameters:")
        for name, param in instruction.parameters.items():
            logger.info(f"  - {name}: {param.value} (required: {param.required})")
    
    # If ambiguous, show disambiguation options
    if instruction.ambiguous and instruction.disambiguation_options:
        logger.info("Disambiguation options:")
        for i, option in enumerate(instruction.disambiguation_options):
            logger.info(f"  {i+1}. {option.get('rephrased', '')}")
        
        # Disambiguate using the first option
        logger.info("Disambiguating with the first option...")
        disambiguated = await instruction_processor.disambiguate(instruction, 0)
        logger.info(f"Disambiguated: {disambiguated.raw_text}")
        logger.info(f"New confidence: {disambiguated.confidence:.2f}")
    
    # Example 6: Task breakdown
    logger.info("\n--- Example 6: Task Breakdown ---")
    
    # Break down a complex task
    complex_task = "Create a monthly report of website analytics, summarize the key metrics, and email it to the team"
    logger.info(f"Breaking down complex task: {complex_task}")
    
    # Parse the instruction first
    complex_instruction = await instruction_processor.parse_instruction(complex_task)
    
    # Break it down
    breakdown = await instruction_processor.break_down_complex_task(complex_instruction)
    
    # Display the breakdown
    logger.info(f"Task description: {breakdown.description}")
    logger.info(f"Number of steps: {len(breakdown.steps)}")
    logger.info("Steps:")
    for i, step in enumerate(breakdown.steps):
        logger.info(f"  {i+1}. {step.instruction.raw_text}")
        logger.info(f"     Description: {step.description}")
        if step.dependencies:
            logger.info(f"     Dependencies: {step.dependencies}")
    
    # Clean up
    await connector.close()
    logger.info("Example completed successfully")


if __name__ == "__main__":
    asyncio.run(main())