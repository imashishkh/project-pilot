#!/usr/bin/env python3
"""
Test module for LLM integration functionality.

This module tests the connection and functionality of different LLM providers.
"""

import os
import json
import time
import pytest
import logging
import asyncio

# Import directly from MacAgent package or alternative paths
try:
    from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider, ModelConfig
except ModuleNotFoundError:
    # Try alternative import paths
    import sys
    from pathlib import Path
    # Add the project root to the path
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider, ModelConfig
    except ModuleNotFoundError:
        from src.intelligence.llm_connector import LLMConnector, LLMProvider, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture
def llm_connector():
    """Fixture providing an LLM connector instance."""
    # Create a connector
    connector = LLMConnector(config_path="config/api_keys.json")
    
    # Yield the connector for use in tests
    yield connector
    
    # Clean up after test
    asyncio.run(connector.close())


@pytest.mark.asyncio
@pytest.mark.intelligence
async def test_openai_connection(llm_connector):
    """Test the OpenAI connection with a simple query."""
    # Skip if OpenAI API key is not available
    try:
        with open("config/api_keys.json", "r") as f:
            config = json.load(f)
        if "OPENAI_API_KEY" not in config or not config["OPENAI_API_KEY"]:
            pytest.skip("OpenAI API key not configured")
    except (FileNotFoundError, json.JSONDecodeError):
        pytest.skip("API keys configuration not found or invalid")
    
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
    
    # Send the query and get response
    response = await llm_connector.generate(messages, config)
    
    # Verify response properties
    assert response is not None, "No response received from OpenAI"
    assert response.text, "Response text is empty"
    assert response.model, "Model name not returned"
    assert response.provider == LLMProvider.OPENAI, "Incorrect provider returned"
    assert response.token_usage, "Token usage not returned"
    assert response.latency > 0, "Invalid latency value"
    
    logger.info(f"OpenAI response: {response.text}")


@pytest.mark.asyncio
@pytest.mark.intelligence
async def test_anthropic_connection(llm_connector):
    """Test the Anthropic connection with a simple query."""
    # Skip if Anthropic API key is not available
    try:
        with open("config/api_keys.json", "r") as f:
            config = json.load(f)
        if "ANTHROPIC_API_KEY" not in config or not config["ANTHROPIC_API_KEY"]:
            pytest.skip("Anthropic API key not configured")
    except (FileNotFoundError, json.JSONDecodeError):
        pytest.skip("API keys configuration not found or invalid")
    
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
    
    # Send the query and get response
    response = await llm_connector.generate(messages, config)
    
    # Verify response properties
    assert response is not None, "No response received from Anthropic"
    assert response.text, "Response text is empty"
    assert response.model, "Model name not returned"
    assert response.provider == LLMProvider.ANTHROPIC, "Incorrect provider returned"
    assert response.token_usage, "Token usage not returned"
    assert response.latency > 0, "Invalid latency value"
    
    logger.info(f"Anthropic response: {response.text}")


@pytest.mark.asyncio
@pytest.mark.intelligence
async def test_caching(llm_connector):
    """Test that the LLM connector caches responses correctly."""
    # Skip if no API keys are available
    try:
        with open("config/api_keys.json", "r") as f:
            config = json.load(f)
        if (("OPENAI_API_KEY" not in config or not config["OPENAI_API_KEY"]) and 
            ("ANTHROPIC_API_KEY" not in config or not config["ANTHROPIC_API_KEY"])):
            pytest.skip("No API keys configured")
    except (FileNotFoundError, json.JSONDecodeError):
        pytest.skip("API keys configuration not found or invalid")
    
    # Determine which provider to use based on available keys
    provider = None
    model_name = None
    
    if "OPENAI_API_KEY" in config and config["OPENAI_API_KEY"]:
        provider = LLMProvider.OPENAI
        model_name = "gpt-4"
    elif "ANTHROPIC_API_KEY" in config and config["ANTHROPIC_API_KEY"]:
        provider = LLMProvider.ANTHROPIC
        model_name = "claude-3-opus-20240229"
    else:
        pytest.skip("No API keys configured")
    
    # Configure the model with caching enabled
    config = ModelConfig(
        provider=provider,
        model_name=model_name,
        max_tokens=100,
        temperature=0.0,  # Use 0 for deterministic responses
        enable_cache=True
    )
    
    # Set up a query
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    # First call - should hit the API
    start_time = time.time()
    response1 = await llm_connector.generate(messages, config)
    first_call_time = time.time() - start_time
    
    # Second call with same prompt - should hit the cache
    start_time = time.time()
    response2 = await llm_connector.generate(messages, config)
    second_call_time = time.time() - start_time
    
    # Verify cache hit
    assert response1.text == response2.text, "Cached response differs from original"
    assert second_call_time < first_call_time, "Second call not faster than first (cache may not be working)"
    
    logger.info(f"First call time: {first_call_time:.3f}s, Second call time: {second_call_time:.3f}s")
    logger.info(f"Cache speedup factor: {first_call_time/second_call_time:.1f}x")


if __name__ == "__main__":
    # For manual execution outside pytest
    async def run_tests():
        connector = LLMConnector(config_path="config/api_keys.json")
        try:
            print("Testing OpenAI connection...")
            await test_openai_connection(connector)
            
            print("\nTesting Anthropic connection...")
            await test_anthropic_connection(connector)
            
            print("\nTesting caching...")
            await test_caching(connector)
        finally:
            await connector.close()
    
    asyncio.run(run_tests()) 