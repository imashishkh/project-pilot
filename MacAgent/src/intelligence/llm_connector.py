"""
LLM Connector Module

This module provides a unified interface for connecting to various LLM providers,
managing API keys, request rate limiting, caching, and fallback mechanisms.
"""

import os
import json
import time
import logging
import hashlib
import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import aiohttp
import backoff
from diskcache import Cache

# Configure logging
logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for enums and dataclasses."""
    
    def default(self, obj):
        if isinstance(obj, Enum):
            return str(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # Add more providers as needed
    
    def __str__(self):
        return self.value
        
    def __repr__(self):
        return f"LLMProvider.{self.name}"
    
    def to_json(self):
        return self.value
    
    @classmethod
    def from_json(cls, data):
        return cls(data)


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    provider: LLMProvider
    model_name: str
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    timeout: float = 60.0  # Request timeout in seconds
    stream: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict with Enum values converted to their string values."""
        result = {}
        for field_name, field_value in asdict(self).items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        return result


@dataclass
class LLMResponse:
    """Response from an LLM request."""
    text: str
    model: str
    provider: LLMProvider
    token_usage: Dict[str, int]
    finish_reason: str
    raw_response: Any
    latency: float  # in seconds
    created_at: datetime = datetime.now()
    cached: bool = False


class RateLimiter:
    """
    Manages request rate limiting for different providers.
    
    Implements token bucket algorithm for rate limiting.
    """
    
    def __init__(self):
        """Initialize rate limiter with default limits."""
        self.limits = {
            LLMProvider.OPENAI: {
                "requests_per_minute": 60,  # Default for most tiers
                "tokens_per_minute": 90000,  # Default for most tiers
            },
            LLMProvider.ANTHROPIC: {
                "requests_per_minute": 50,  # Default for Anthropic
                "tokens_per_minute": 100000,  # Approximate default
            }
        }
        
        # Track usage
        self.usage = {provider: {
            "last_reset": datetime.now(),
            "requests_count": 0,
            "tokens_count": 0
        } for provider in self.limits}
        
        self.lock = asyncio.Lock()
    
    def update_limits(self, provider: LLMProvider, requests_per_minute: int = None, tokens_per_minute: int = None):
        """Update rate limits for a specific provider."""
        if provider not in self.limits:
            self.limits[provider] = {
                "requests_per_minute": 30,  # Conservative default
                "tokens_per_minute": 40000,  # Conservative default
            }
            
        if requests_per_minute is not None:
            self.limits[provider]["requests_per_minute"] = requests_per_minute
            
        if tokens_per_minute is not None:
            self.limits[provider]["tokens_per_minute"] = tokens_per_minute
    
    async def check_and_update(self, provider: LLMProvider, tokens: int = 0) -> bool:
        """
        Check if a request can be made and update usage.
        
        Args:
            provider: The LLM provider
            tokens: Estimated tokens for the request
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            # Reset counters if a minute has passed
            now = datetime.now()
            provider_usage = self.usage[provider]
            
            if (now - provider_usage["last_reset"]).total_seconds() >= 60:
                provider_usage["last_reset"] = now
                provider_usage["requests_count"] = 0
                provider_usage["tokens_count"] = 0
            
            # Check if within limits
            provider_limits = self.limits[provider]
            if (provider_usage["requests_count"] >= provider_limits["requests_per_minute"] or
                provider_usage["tokens_count"] + tokens >= provider_limits["tokens_per_minute"]):
                return False
            
            # Update usage
            provider_usage["requests_count"] += 1
            provider_usage["tokens_count"] += tokens
            
            return True
    
    async def wait_if_needed(self, provider: LLMProvider, tokens: int = 0) -> None:
        """
        Wait until a request can be made.
        
        Args:
            provider: The LLM provider
            tokens: Estimated tokens for the request
        """
        while not await self.check_and_update(provider, tokens):
            # Sleep a portion of a minute before checking again
            await asyncio.sleep(60 / self.limits[provider]["requests_per_minute"])


class ResponseCache:
    """
    Caches LLM responses to avoid redundant requests.
    """
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 86400):
        """
        Initialize the response cache.
        
        Args:
            cache_dir: Directory for cache storage
            ttl: Time-to-live for cache entries in seconds (default: 24 hours)
        """
        self.cache = Cache(cache_dir)
        self.ttl = ttl
        self.hit_count = 0
        self.miss_count = 0
    
    def _get_cache_key(self, model: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a request.
        
        Args:
            model: The model name
            messages: The conversation messages
            params: Additional parameters
            
        Returns:
            A unique cache key
        """
        # Normalize parameters that may contain enum values
        normalized_params = {}
        for k, v in params.items():
            if k == "stream" or v is None:
                continue
                
            if isinstance(v, Enum):
                normalized_params[k] = v.value
            else:
                normalized_params[k] = v
        
        # Convert to JSON string and hash
        request_str = json.dumps({
            "model": model,
            "messages": messages,
            "params": normalized_params
        }, sort_keys=True, cls=CustomJSONEncoder)
        
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def get(self, model: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Optional[LLMResponse]:
        """
        Get a cached response if available.
        
        Args:
            model: The model name
            messages: The conversation messages
            params: Additional parameters
            
        Returns:
            Cached response or None if not found
        """
        # Don't cache streaming requests
        if params.get("stream", False):
            self.miss_count += 1
            return None
            
        key = self._get_cache_key(model, messages, params)
        response = self.cache.get(key)
        
        if response:
            self.hit_count += 1
            # Mark as cached
            response.cached = True
            return response
            
        self.miss_count += 1
        return None
    
    def store(self, model: str, messages: List[Dict[str, str]], params: Dict[str, Any], response: LLMResponse) -> None:
        """
        Store a response in the cache.
        
        Args:
            model: The model name
            messages: The conversation messages
            params: Additional parameters
            response: The response to cache
        """
        # Don't cache streaming responses
        if params.get("stream", False):
            return
            
        key = self._get_cache_key(model, messages, params)
        self.cache.set(key, response, expire=self.ttl)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get cache statistics.
        
        Returns:
            Cache hit ratio and counts
        """
        total = self.hit_count + self.miss_count
        return {
            "hit_ratio": self.hit_count / total if total > 0 else 0,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class LLMConnector:
    """
    Manages connections to LLM providers and handles API requests.
    
    Features:
    - Connects to multiple LLM providers
    - Manages API keys and authentication
    - Handles request rate limiting
    - Implements caching for efficiency
    - Provides fallback mechanisms for service disruptions
    """
    
    def __init__(self, 
                config_path: str = "config/api_keys.json",
                cache_dir: str = ".cache",
                cache_ttl: int = 86400,
                enable_cache: bool = True):
        """
        Initialize the LLM connector.
        
        Args:
            config_path: Path to API keys configuration file
            cache_dir: Directory for cache storage
            cache_ttl: Time-to-live for cache entries in seconds
            enable_cache: Whether to enable response caching
        """
        self.api_keys = {}
        self.session = None
        self.rate_limiter = RateLimiter()
        
        # Initialize cache if enabled
        self.cache = ResponseCache(cache_dir, cache_ttl) if enable_cache else None
        self.enable_cache = enable_cache
        
        # Load API keys from config file
        self._load_api_keys(config_path)
    
    def _load_api_keys(self, config_path: str) -> None:
        """
        Load API keys from configuration file.
        
        Args:
            config_path: Path to API keys configuration file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Map keys to providers
                key_mapping = {
                    "openai_api_key": LLMProvider.OPENAI,
                    "anthropic_api_key": LLMProvider.ANTHROPIC
                }
                
                for key_name, provider in key_mapping.items():
                    if key_name in config and config[key_name]:
                        self.api_keys[provider] = config[key_name]
                        logger.info(f"Loaded API key for {provider.value}")
                    else:
                        logger.warning(f"API key for {provider.value} not found in config")
                        
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
    
    async def _init_session(self) -> None:
        """Initialize aiohttp session if needed."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    def _prepare_headers(self, provider: LLMProvider) -> Dict[str, str]:
        """
        Prepare headers for API requests.
        
        Args:
            provider: The LLM provider
            
        Returns:
            Dictionary of headers
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization headers
        if provider == LLMProvider.OPENAI:
            if provider in self.api_keys:
                headers["Authorization"] = f"Bearer {self.api_keys[provider]}"
                
        elif provider == LLMProvider.ANTHROPIC:
            if provider in self.api_keys:
                headers["x-api-key"] = self.api_keys[provider]
                headers["anthropic-version"] = "2023-06-01"  # Use appropriate version
        
        return headers
    
    def _prepare_payload(self, 
                         provider: LLMProvider, 
                         messages: List[Dict[str, str]], 
                         config: ModelConfig) -> Dict[str, Any]:
        """
        Prepare request payload based on provider.
        
        Args:
            provider: The LLM provider
            messages: List of conversation messages
            config: Model configuration
            
        Returns:
            Request payload for the API
        """
        # Convert provider to string to ensure JSON serialization
        provider_str = str(provider)
        
        if provider == LLMProvider.OPENAI:
            payload = {
                "model": config.model_name,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stream": config.stream
            }
            
            if config.stop_sequences:
                payload["stop"] = config.stop_sequences
                
        elif provider == LLMProvider.ANTHROPIC:
            # Modern Anthropic API uses the messages format directly
            payload = {
                "model": config.model_name,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stream": config.stream
            }
            
            if config.stop_sequences:
                payload["stop_sequences"] = config.stop_sequences
        
        else:
            raise ValueError(f"Unsupported provider: {provider_str}")
        
        return payload
    
    def _parse_response(self, 
                       provider: LLMProvider, 
                       model: str, 
                       response_data: Dict[str, Any],
                       latency: float) -> LLMResponse:
        """
        Parse API response into a standardized format.
        
        Args:
            provider: The LLM provider
            model: The model name
            response_data: Raw API response
            latency: Request latency in seconds
            
        Returns:
            Standardized LLM response
        """
        # Convert provider to string for error messages
        provider_str = str(provider)
        
        # Log the raw response structure for debugging
        logger.debug(f"Raw response from {provider_str}: {json.dumps(response_data, default=str)}")
        
        if provider == LLMProvider.OPENAI:
            text = response_data["choices"][0]["message"]["content"]
            finish_reason = response_data["choices"][0]["finish_reason"]
            token_usage = response_data["usage"]
            
        elif provider == LLMProvider.ANTHROPIC:
            # Modern Anthropic API structure
            if "content" in response_data and isinstance(response_data["content"], list):
                text_blocks = [item["text"] for item in response_data["content"] if item["type"] == "text"]
                text = "".join(text_blocks)
            else:
                # Fallback for older API format
                text = response_data.get("completion", "")
                
            finish_reason = response_data.get("stop_reason", "unknown")
            
            # Extract token usage from modern Anthropic API
            token_usage = {
                "prompt_tokens": response_data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response_data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": 
                    response_data.get("usage", {}).get("input_tokens", 0) + 
                    response_data.get("usage", {}).get("output_tokens", 0)
            }
            
        else:
            raise ValueError(f"Unsupported provider: {provider_str}")
        
        return LLMResponse(
            text=text,
            model=model,
            provider=provider,
            token_usage=token_usage,
            finish_reason=finish_reason,
            raw_response=response_data,
            latency=latency
        )
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate the number of tokens in a request.
        
        This is a simplified estimator that assumes ~4 chars per token.
        For more accurate estimates, use a tokenizer.
        
        Args:
            messages: The conversation messages
            
        Returns:
            Estimated token count
        """
        total_chars = sum(len(m["content"]) for m in messages)
        # Assume 4 characters per token (very rough estimate)
        return total_chars // 4 + 1
    
    def _get_api_url(self, provider: LLMProvider) -> str:
        """
        Get the API URL for a provider.
        
        Args:
            provider: The LLM provider
            
        Returns:
            API URL
        """
        if provider == LLMProvider.OPENAI:
            return "https://api.openai.com/v1/chat/completions"
        elif provider == LLMProvider.ANTHROPIC:
            return "https://api.anthropic.com/v1/messages"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=5,
        max_time=60
    )
    async def _make_api_request(self, 
                               provider: LLMProvider, 
                               messages: List[Dict[str, str]], 
                               config: ModelConfig) -> Dict[str, Any]:
        """
        Make an API request with backoff retry logic.
        
        Args:
            provider: The LLM provider
            messages: The conversation messages
            config: Model configuration
            
        Returns:
            API response data
        """
        await self._init_session()
        
        headers = self._prepare_headers(provider)
        payload = self._prepare_payload(provider, messages, config)
        url = self._get_api_url(provider)
        
        # Estimate tokens for rate limiting
        estimated_tokens = self._estimate_tokens(messages) + config.max_tokens
        
        # Wait if rate limited
        await self.rate_limiter.wait_if_needed(provider, estimated_tokens)
        
        # Make the request - use json.dumps with custom encoder to handle all cases
        start_time = time.time()
        json_payload = json.dumps(payload, cls=CustomJSONEncoder)
        async with self.session.post(
            url,
            headers=headers,
            data=json_payload,
            timeout=config.timeout
        ) as response:
            latency = time.time() - start_time
            
            if response.status != 200:
                error_info = await response.text()
                logger.error(f"API error ({response.status}): {error_info}")
                response.raise_for_status()
                
            return await response.json(), latency
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      config: ModelConfig,
                      fallback_configs: Optional[List[ModelConfig]] = None) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: The conversation messages
            config: Primary model configuration
            fallback_configs: Optional list of fallback configurations
            
        Returns:
            LLM response
        """
        provider = config.provider
        
        # Check API key
        if provider not in self.api_keys:
            logger.error(f"API key for {provider.value} not found")
            # Try fallback if available
            if fallback_configs:
                logger.info(f"Falling back to alternative model")
                return await self.generate(messages, fallback_configs[0], 
                                          fallback_configs[1:] if len(fallback_configs) > 1 else None)
            else:
                raise ValueError(f"No API key for {provider.value} and no fallback available")
        
        # Convert config to dict for serialization
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        
        # Check cache if enabled and not streaming
        if self.enable_cache and not config.stream:
            cached_response = self.cache.get(config.model_name, messages, config_dict)
            if cached_response:
                logger.info(f"Cache hit for {provider.value}/{config.model_name}")
                return cached_response
        
        try:
            # Make the API request
            response_data, latency = await self._make_api_request(provider, messages, config)
            
            # Parse the response
            response = self._parse_response(provider, config.model_name, response_data, latency)
            
            # Store in cache if enabled and not streaming
            if self.enable_cache and not config.stream:
                self.cache.store(config.model_name, messages, config_dict, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response from {provider.value}/{config.model_name}: {e}")
            
            # Try fallback if available
            if fallback_configs:
                logger.info(f"Falling back to alternative model")
                return await self.generate(messages, fallback_configs[0], 
                                          fallback_configs[1:] if len(fallback_configs) > 1 else None)
            else:
                raise
    
    def update_rate_limits(self, provider: LLMProvider, requests_per_minute: int = None, tokens_per_minute: int = None):
        """
        Update rate limits for a provider.
        
        Args:
            provider: The LLM provider
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
        """
        self.rate_limiter.update_limits(provider, requests_per_minute, tokens_per_minute)
    
    def get_cache_stats(self) -> Dict[str, float]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics or None if cache is disabled
        """
        if self.enable_cache:
            return self.cache.get_stats()
        return None
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self.enable_cache:
            self.cache.clear()
            logger.info("Response cache cleared")
    
    def __del__(self):
        """Clean up resources."""
        # Close aiohttp session if in a running event loop
        if self.session and not self.session.closed:
            loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None
            if loop:
                loop.create_task(self.close())
            else:
                logger.warning("Unable to close aiohttp session properly during cleanup")
