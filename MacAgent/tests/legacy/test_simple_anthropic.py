#!/usr/bin/env python3
"""
Simple test script that directly tests the Anthropic API without using our custom classes.
"""

import os
import sys
import json
import asyncio
import logging
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_simple_anthropic():
    """Test Anthropic API directly."""
    # Load API key from config
    try:
        with open("config/api_keys.json", "r") as f:
            config = json.load(f)
            api_key = config.get("anthropic_api_key")
            
        if not api_key:
            logger.error("Anthropic API key not found in config")
            return
    except Exception as e:
        logger.error(f"Error loading API key: {e}")
        return
    
    # Set up headers and payload
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "Explain how to rename a file on Mac in 2-3 sentences."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    # Make the request
    url = "https://api.anthropic.com/v1/messages"
    
    try:
        async with aiohttp.ClientSession() as session:
            logger.info("Sending request to Anthropic...")
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract the response
                    text = data["content"][0]["text"]
                    
                    logger.info("Anthropic Response:")
                    logger.info(f"{text}")
                    logger.info(f"Model: {data['model']}")
                    if "usage" in data:
                        logger.info(f"Token usage: {data['usage']}")
                else:
                    error_info = await response.text()
                    logger.error(f"API error ({response.status}): {error_info}")
    except Exception as e:
        logger.error(f"Error testing Anthropic: {e}")


if __name__ == "__main__":
    asyncio.run(test_simple_anthropic()) 