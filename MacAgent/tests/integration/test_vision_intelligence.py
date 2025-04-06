"""
Integration tests for vision and intelligence components.

This module tests the integration between vision and intelligence systems.
"""

import os
import logging
import pytest
import cv2
import numpy as np

from MacAgent.src.vision.screen_capture import ScreenCapture
from MacAgent.src.vision.element_detector import UIElementDetector
from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture
def ui_element_detector():
    """Fixture providing a UI element detector."""
    return UIElementDetector()


@pytest.fixture
def screen_capture():
    """Fixture providing a screen capture instance."""
    capture = ScreenCapture()
    yield capture
    capture.close()


@pytest.fixture
def llm_connector():
    """Fixture providing an LLM connector instance."""
    connector = LLMConnector(config_path="config/api_keys.json")
    yield connector
    import asyncio
    asyncio.run(connector.close())


@pytest.mark.integration
@pytest.mark.vision
@pytest.mark.intelligence
def test_describe_ui_elements(
    ui_element_detector,
    screen_capture,
    llm_connector,
    integration_test_output_dir,
    test_resources_dir
):
    """Test integration of UI element detection and LLM description."""
    try:
        # Try to use a synthetic image if available
        synthetic_image_path = os.path.join(test_resources_dir, "images", "synthetic_ui_light.png")
        if os.path.exists(synthetic_image_path):
            logger.info(f"Using synthetic UI image: {synthetic_image_path}")
            image = cv2.imread(synthetic_image_path)
        else:
            # Capture a screenshot as a fallback
            logger.info("Capturing screenshot for UI element detection")
            image = screen_capture.capture()
    
        # Save the image for reference
        input_image_path = os.path.join(integration_test_output_dir, "ui_screenshot.png")
        cv2.imwrite(input_image_path, image)
        
        # Detect UI elements
        logger.info("Detecting UI elements")
        ui_elements = ui_element_detector.detect_elements(image)
        
        # Skip if no UI elements were detected
        if not ui_elements:
            pytest.skip("No UI elements detected in the image")
            
        # Create a visualization of detected elements
        visualization = image.copy()
        for element in ui_elements:
            x, y, w, h = element.bounding_box
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(visualization, element.element_type.name, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save the visualization
        vis_path = os.path.join(integration_test_output_dir, "detected_elements.png")
        cv2.imwrite(vis_path, visualization)
        
        # Prepare description for LLM
        element_descriptions = [
            f"- {i+1}. Type: {element.element_type.name}, Position: {element.bounding_box}"
            for i, element in enumerate(ui_elements[:10])  # Limit to first 10 elements
        ]
        
        element_description_text = "\n".join(element_descriptions)
        
        # Skip LLM test if running in CI environment or API keys not available
        import json
        try:
            with open("config/api_keys.json", "r") as f:
                api_keys = json.load(f)
            has_api_keys = (
                ("OPENAI_API_KEY" in api_keys and api_keys["OPENAI_API_KEY"]) or
                ("ANTHROPIC_API_KEY" in api_keys and api_keys["ANTHROPIC_API_KEY"])
            )
        except (FileNotFoundError, json.JSONDecodeError):
            has_api_keys = False
            
        if "CI" in os.environ or not has_api_keys:
            logger.info("Skipping LLM integration test (CI environment or no API keys)")
            pytest.skip("Skipping LLM test in CI environment or missing API keys")
            
        # Get LLM to describe the UI
        import asyncio
        
        async def get_ui_description():
            provider = LLMProvider.OPENAI
            model = "gpt-4"
            
            if "OPENAI_API_KEY" not in api_keys or not api_keys["OPENAI_API_KEY"]:
                provider = LLMProvider.ANTHROPIC
                model = "claude-3-opus-20240229"
            
            config = ModelConfig(
                provider=provider,
                model_name=model,
                max_tokens=300
            )
            
            messages = [
                {"role": "system", "content": "You are a UI analysis assistant."},
                {"role": "user", "content": f"I've detected these UI elements:\n\n{element_description_text}\n\nProvide a brief, clear summary of what this UI might be (e.g., login screen, settings page, etc.) and what functionality it likely offers based on the detected elements. Keep your response under 150 words."}
            ]
            
            response = await llm_connector.generate(messages, config)
            return response.text
        
        # Run the async function
        description = asyncio.run(get_ui_description())
        
        # Save the description
        with open(os.path.join(integration_test_output_dir, "ui_description.txt"), "w") as f:
            f.write(description)
            
        logger.info(f"Generated UI description: {description}")
        
        # Simple validation that we got some text back
        assert description, "No description was generated"
        assert len(description) > 50, "Description is too short"
        
    except Exception as e:
        logger.error(f"Error in integration test: {e}")
        pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    # For manual execution outside pytest
    import sys
    import os
    
    # Configure output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "integration")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure resources directory
    resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
    
    # Run the test
    detector = UIElementDetector()
    capture = ScreenCapture()
    connector = LLMConnector(config_path="config/api_keys.json")
    
    try:
        test_describe_ui_elements(detector, capture, connector, output_dir, resources_dir)
        print("Integration test completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        capture.close()
        import asyncio
        asyncio.run(connector.close()) 