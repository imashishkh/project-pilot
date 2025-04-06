#!/usr/bin/env python3
"""
Test script for the LLM integration system.
"""

import os
import sys
import asyncio
import logging

# Add the parent directory to the path to import the MacAgent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_openai_connection():
    """Test the OpenAI connection with a simple query."""
    logger.info("=== Testing OpenAI Connection ===")
    connector = LLMConnector(config_path="config/api_keys.json")
    
    # Configure the model
    config = ModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        max_tokens=100,
        temperature=0.7
    )
    
    # Set up a simple query
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how to rename a file on Mac in 2-3 sentences."}
    ]
    
    try:
        logger.info("Sending request to OpenAI...")
        response = await connector.generate(messages, config)
        
        logger.info("OpenAI Response:")
        logger.info(f"{response.text}")
        logger.info(f"Model: {response.model}")
        logger.info(f"Provider: {response.provider.value}")
        logger.info(f"Token usage: {response.token_usage}")
        logger.info(f"Latency: {response.latency:.3f} seconds")
    except Exception as e:
        logger.error(f"Error testing OpenAI connection: {e}")
    finally:
        await connector.close()


async def test_anthropic_connection():
    """Test the Anthropic connection with a simple query."""
    logger.info("\n=== Testing Anthropic Connection ===")
    connector = LLMConnector(config_path="config/api_keys.json")
    
    # Configure the model
    config = ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        max_tokens=100,
        temperature=0.7
    )
    
    # Set up a simple query
    messages = [
        {"role": "user", "content": "Explain how to rename a file on Mac in 2-3 sentences."}
    ]
    
    try:
        logger.info("Sending request to Anthropic...")
        response = await connector.generate(messages, config)
        
        logger.info("Anthropic Response:")
        logger.info(f"{response.text}")
        logger.info(f"Model: {response.model}")
        logger.info(f"Provider: {response.provider.value}")
        logger.info(f"Token usage: {response.token_usage}")
        logger.info(f"Latency: {response.latency:.3f} seconds")
    except Exception as e:
        logger.error(f"Error testing Anthropic connection: {e}")
    finally:
        await connector.close()


async def test_llm_connections():
    """Test both OpenAI and Anthropic connections."""
    await test_openai_connection()
    await test_anthropic_connection()


if __name__ == "__main__":
    asyncio.run(test_llm_connections()) 