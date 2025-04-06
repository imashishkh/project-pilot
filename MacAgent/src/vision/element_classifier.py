"""
UI Element Classifier Module for MacAgent

This module provides classification capabilities for detected UI elements,
assigning types, confidence scores, and filtering out false positives.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Set, TYPE_CHECKING
from collections import defaultdict, Counter
import re

# Local imports
from .element_detector import UIElement, ElementType, ElementState

# Configure logging
logger = logging.getLogger(__name__)


class ElementClassifier:
    """
    Classifies and refines detected UI elements.
    
    This class provides methods for categorizing UI elements, assigning confidence
    scores, filtering out false positives, and prioritizing elements based on context.
    """
    
    def __init__(self, min_confidence: float = 0.6, 
                 text_patterns: Optional[Dict[ElementType, List[str]]] = None):
        """
        Initialize the element classifier.
        
        Args:
            min_confidence: Minimum confidence threshold for classification
            text_patterns: Dictionary mapping element types to regex patterns for their typical text
        """
        self.min_confidence = min_confidence
        
        # Define text patterns for element types if not provided
        self.text_patterns = text_patterns or {
            ElementType.BUTTON: [
                r'submit', r'save', r'cancel', r'ok', r'apply', r'close',
                r'add', r'delete', r'remove', r'edit', r'new', r'open'
            ],
            ElementType.CHECKBOX: [
                r'enable', r'disable', r'show', r'hide', r'remember',
                r'agree', r'accept', r'select all'
            ],
            ElementType.DROPDOWN: [
                r'select', r'choose', r'options?'
            ],
        }
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for faster matching."""
        self.compiled_patterns = {}
        for elem_type, patterns in self.text_patterns.items():
            self.compiled_patterns[elem_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def classify_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """
        Classify UI elements and refine their properties.
        
        Args:
            elements: List of detected UI elements
            
        Returns:
            List of classified elements with updated properties
        """
        logger.info(f"Classifying {len(elements)} UI elements")
        
        # First, refine classification based on element properties
        for element in elements:
            self._refine_element_classification(element)
        
        # Filter out false positives
        filtered_elements = self.filter_false_positives(elements)
        
        # Compute confidence scores
        for element in filtered_elements:
            confidence = self.compute_confidence(element)
            element.confidence = max(element.confidence, confidence)
        
        # Final filtering based on confidence
        classified_elements = [elem for elem in filtered_elements 
                              if elem.confidence >= self.min_confidence]
        
        logger.info(f"Classification complete. {len(classified_elements)} elements retained.")
        return classified_elements
    
    def _refine_element_classification(self, element: UIElement) -> None:
        """
        Refine the classification of a UI element based on its properties.
        
        Args:
            element: The UI element to refine
        """
        # If the element has text, use it to help with classification
        if element.text:
            text = element.text.lower()
            
            # Check against text patterns for different element types
            for elem_type, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(text):
                        # If element type is unknown or the confidence is improved
                        if element.element_type == ElementType.UNKNOWN:
                            element.element_type = elem_type
                            break
        
        # Use size and shape for additional classification cues
        x, y, w, h = element.bounding_box
        
        # Buttons typically have certain width-to-height ratios
        aspect_ratio = w / h if h > 0 else 0
        
        if element.element_type == ElementType.UNKNOWN:
            if 1.5 <= aspect_ratio <= 5.0 and 20 <= h <= 40:
                element.element_type = ElementType.BUTTON
            elif aspect_ratio > 8.0 and h < 25:
                element.element_type = ElementType.TEXT_FIELD
            elif 0.9 <= aspect_ratio <= 1.1 and 10 <= w <= 30 and 10 <= h <= 30:
                # Square elements could be checkboxes or radio buttons
                element.element_type = ElementType.CHECKBOX
    
    def compute_confidence(self, element: UIElement) -> float:
        """
        Compute a confidence score for an element's classification.
        
        Args:
            element: The UI element
            
        Returns:
            Confidence score between 0 and 1
        """
        # Start with base confidence
        confidence = element.confidence
        
        # Adjust based on element properties
        if element.text:
            # Elements with text are more likely to be correctly classified
            text = element.text.lower()
            
            # Check if the text matches expected patterns for the element type
            if element.element_type in self.compiled_patterns:
                for pattern in self.compiled_patterns[element.element_type]:
                    if pattern.search(text):
                        confidence += 0.1
                        break
        
        # Adjust based on bounding box properties
        x, y, w, h = element.bounding_box
        aspect_ratio = w / h if h > 0 else 0
        
        # Element-specific adjustments
        if element.element_type == ElementType.BUTTON:
            # Buttons typically have certain aspect ratios
            if 1.5 <= aspect_ratio <= 5.0 and 20 <= h <= 40:
                confidence += 0.1
            else:
                confidence -= 0.1
        
        elif element.element_type == ElementType.CHECKBOX:
            # Checkboxes are typically square
            if 0.9 <= aspect_ratio <= 1.1 and 10 <= w <= 30:
                confidence += 0.1
            else:
                confidence -= 0.1
        
        elif element.element_type == ElementType.TEXT_FIELD:
            # Text fields are typically wider than tall
            if aspect_ratio > 3.0 and h >= 20:
                confidence += 0.1
            else:
                confidence -= 0.1
        
        # Cap confidence at 1.0
        return min(max(confidence, 0.0), 1.0)
    
    def filter_false_positives(self, elements: List[UIElement]) -> List[UIElement]:
        """
        Filter out likely false positive detections.
        
        Args:
            elements: List of detected UI elements
            
        Returns:
            Filtered list of elements
        """
        logger.info(f"Filtering false positives from {len(elements)} elements")
        
        filtered = []
        overlapping_elements = self._find_overlapping_elements(elements)
        
        for element in elements:
            # Skip elements with very low confidence
            if element.confidence < 0.3:
                continue
            
            # Check if this element overlaps with others
            if element.element_id in overlapping_elements:
                # For overlapping elements, keep only the one with highest confidence
                overlapping_ids = overlapping_elements[element.element_id]
                overlapping_elements_list = [e for e in elements if e.element_id in overlapping_ids]
                
                # Add if this is the highest confidence element in the group
                if element.confidence >= max(e.confidence for e in overlapping_elements_list):
                    filtered.append(element)
            else:
                # Non-overlapping element, keep it
                filtered.append(element)
        
        logger.info(f"Filtered out {len(elements) - len(filtered)} potential false positives")
        return filtered
    
    def _find_overlapping_elements(self, elements: List[UIElement]) -> Dict[str, Set[str]]:
        """
        Find sets of overlapping elements.
        
        Args:
            elements: List of UI elements
            
        Returns:
            Dictionary mapping element IDs to sets of overlapping element IDs
        """
        overlapping = defaultdict(set)
        
        # Check all pairs of elements for overlap
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if self._elements_overlap(elem1, elem2):
                    overlapping[elem1.element_id].add(elem2.element_id)
                    overlapping[elem2.element_id].add(elem1.element_id)
        
        return overlapping
    
    def _elements_overlap(self, elem1: UIElement, elem2: UIElement, threshold: float = 0.5) -> bool:
        """
        Check if two elements overlap significantly.
        
        Args:
            elem1: First UI element
            elem2: Second UI element
            threshold: Minimum overlap ratio to consider elements as overlapping
            
        Returns:
            True if elements overlap significantly, False otherwise
        """
        x1, y1, w1, h1 = elem1.bounding_box
        x2, y2, w2, h2 = elem2.bounding_box
        
        # Calculate intersection
        x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection_area = x_intersection * y_intersection
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate overlap ratio
        if area1 > 0 and area2 > 0:
            overlap_ratio = intersection_area / min(area1, area2)
            return overlap_ratio > threshold
        
        return False
    
    def prioritize_elements(self, elements: List[UIElement], 
                         context: Optional[Dict[str, Any]] = None) -> List[UIElement]:
        """
        Prioritize elements based on context and importance.
        
        Args:
            elements: List of UI elements to prioritize
            context: Optional context information for better prioritization
            
        Returns:
            List of elements sorted by priority (highest first)
        """
        logger.info(f"Prioritizing {len(elements)} elements")
        
        # Default context if none provided
        context = context or {}
        
        # Assign priority scores
        prioritized = [(element, self._calculate_priority(element, context)) 
                      for element in elements]
        
        # Sort by priority score (descending)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the elements
        return [element for element, _ in prioritized]
    
    def _calculate_priority(self, element: UIElement, context: Dict[str, Any]) -> float:
        """
        Calculate a priority score for an element based on context.
        
        Args:
            element: The UI element
            context: Context information
            
        Returns:
            Priority score (higher is more important)
        """
        priority = element.confidence
        
        # Adjust priority based on element type
        type_priority = {
            ElementType.BUTTON: 0.8,
            ElementType.TEXT_FIELD: 0.7,
            ElementType.CHECKBOX: 0.6,
            ElementType.RADIO_BUTTON: 0.6,
            ElementType.DROPDOWN: 0.7,
            ElementType.WINDOW_CONTROL: 0.5,
            ElementType.MENU_ITEM: 0.6,
            ElementType.UNKNOWN: 0.4
        }
        
        priority += type_priority.get(element.element_type, 0.5)
        
        # Adjust priority based on element state
        if element.state == ElementState.DISABLED:
            priority -= 0.3
        elif element.state == ElementState.SELECTED:
            priority += 0.1
        
        # If there's a search term in the context, prioritize elements with matching text
        if 'search_term' in context and element.text:
            search_term = context['search_term'].lower()
            if search_term in element.text.lower():
                priority += 0.5
        
        # Prioritize elements in the center of the screen if screen dimensions are provided
        if 'screen_width' in context and 'screen_height' in context:
            screen_width = context['screen_width']
            screen_height = context['screen_height']
            
            x, y, w, h = element.bounding_box
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Calculate distance from screen center
            screen_center_x = screen_width / 2
            screen_center_y = screen_height / 2
            
            distance_from_center = np.sqrt((center_x - screen_center_x) ** 2 + 
                                       (center_y - screen_center_y) ** 2)
            
            # Normalize by screen diagonal
            screen_diagonal = np.sqrt(screen_width ** 2 + screen_height ** 2)
            normalized_distance = distance_from_center / screen_diagonal
            
            # Closer to center gets higher priority
            priority += 0.2 * (1.0 - normalized_distance)
        
        return priority
    
    def analyze_element_metadata(self, element: UIElement) -> Dict[str, Any]:
        """
        Analyze an element and extract additional metadata.
        
        Args:
            element: The UI element to analyze
            
        Returns:
            Dictionary of metadata about the element
        """
        metadata = {}
        
        # Extract information based on element type
        if element.element_type == ElementType.BUTTON:
            metadata['action'] = self._infer_button_action(element)
            metadata['importance'] = self._infer_button_importance(element)
        
        elif element.element_type == ElementType.TEXT_FIELD:
            metadata['field_type'] = self._infer_text_field_type(element)
        
        elif element.element_type in (ElementType.CHECKBOX, ElementType.RADIO_BUTTON):
            metadata['setting_type'] = self._infer_setting_type(element)
        
        return metadata
    
    def _infer_button_action(self, button: UIElement) -> str:
        """Infer the likely action of a button based on its text."""
        if not button.text:
            return "unknown"
            
        text = button.text.lower()
        
        # Destructive actions
        if any(keyword in text for keyword in ['delete', 'remove', 'clear']):
            return "destructive"
            
        # Constructive actions
        if any(keyword in text for keyword in ['add', 'create', 'new']):
            return "constructive"
            
        # Confirmation actions
        if any(keyword in text for keyword in ['ok', 'yes', 'confirm', 'accept']):
            return "confirmation"
            
        # Cancellation actions
        if any(keyword in text for keyword in ['cancel', 'no', 'abort']):
            return "cancellation"
            
        # Save/apply actions
        if any(keyword in text for keyword in ['save', 'apply', 'update']):
            return "save"
            
        return "other"
    
    def _infer_button_importance(self, button: UIElement) -> str:
        """Infer the importance of a button based on its properties."""
        if not button.text:
            return "medium"
            
        text = button.text.lower()
        
        # Primary actions are usually highlighted in the UI
        if any(keyword in text for keyword in ['ok', 'save', 'submit', 'continue']):
            return "high"
            
        # Secondary actions
        if any(keyword in text for keyword in ['cancel', 'back']):
            return "medium"
            
        # Destructive actions are often de-emphasized
        if any(keyword in text for keyword in ['delete', 'remove']):
            return "low"
            
        return "medium"
    
    def _infer_text_field_type(self, text_field: UIElement) -> str:
        """Infer the type of a text field based on its properties and surroundings."""
        if not text_field.text:
            return "generic"
            
        text = text_field.text.lower()
        
        # Password fields often have masked text or password hints
        if any(keyword in text for keyword in ['password', '••••']):
            return "password"
            
        # Email fields
        if any(keyword in text for keyword in ['email', '@']):
            return "email"
            
        # Search fields
        if any(keyword in text for keyword in ['search', 'find']):
            return "search"
            
        # URL fields
        if any(keyword in text for keyword in ['url', 'http', 'www']):
            return "url"
            
        return "generic"
    
    def _infer_setting_type(self, element: UIElement) -> str:
        """Infer the type of setting represented by a checkbox or radio button."""
        if not element.text:
            return "generic"
            
        text = element.text.lower()
        
        # Privacy-related settings
        if any(keyword in text for keyword in ['privacy', 'track', 'collect']):
            return "privacy"
            
        # Notification settings
        if any(keyword in text for keyword in ['notify', 'notification', 'alert']):
            return "notification"
            
        # Feature toggles
        if any(keyword in text for keyword in ['enable', 'disable', 'activate']):
            return "feature_toggle"
            
        # Preference settings
        if any(keyword in text for keyword in ['prefer', 'option', 'setting']):
            return "preference"
            
        return "generic"
    
    def merge_duplicate_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """
        Merge duplicate elements that likely represent the same UI component.
        
        Args:
            elements: List of UI elements
            
        Returns:
            List with duplicates merged
        """
        # Group elements by their text content
        text_groups = defaultdict(list)
        for element in elements:
            if element.text:
                text_groups[element.text.lower()].append(element)
        
        # Process groups to find and merge duplicates
        merged_elements = []
        processed_ids = set()
        
        # First add elements that don't have duplicates
        for element in elements:
            if element.element_id in processed_ids:
                continue
                
            if not element.text or len(text_groups[element.text.lower()]) <= 1:
                merged_elements.append(element)
                processed_ids.add(element.element_id)
                continue
            
            # For elements with potential duplicates, merge them
            duplicates = text_groups[element.text.lower()]
            if len(duplicates) > 1:
                merged = self._merge_element_group(duplicates)
                merged_elements.append(merged)
                
                # Mark all duplicates as processed
                for dup in duplicates:
                    processed_ids.add(dup.element_id)
        
        return merged_elements
    
    def _merge_element_group(self, elements: List[UIElement]) -> UIElement:
        """
        Merge a group of similar elements into a single element.
        
        Args:
            elements: List of similar UI elements
            
        Returns:
            Merged UI element
        """
        # Sort by confidence and use the highest confidence element as base
        elements.sort(key=lambda x: x.confidence, reverse=True)
        base_element = elements[0]
        
        # Calculate the average bounding box
        total_x, total_y, total_w, total_h = 0, 0, 0, 0
        for element in elements:
            x, y, w, h = element.bounding_box
            total_x += x
            total_y += y
            total_w += w
            total_h += h
        
        avg_x = total_x // len(elements)
        avg_y = total_y // len(elements)
        avg_w = total_w // len(elements)
        avg_h = total_h // len(elements)
        
        # Use the base element but with averaged bounding box
        base_element.bounding_box = (avg_x, avg_y, avg_w, avg_h)
        
        # Boost confidence slightly for merged elements
        base_element.confidence = min(base_element.confidence + 0.1, 1.0)
        
        return base_element


class RelationshipMapper:
    """
    Maps relationships between UI elements to understand UI structure.
    
    This class identifies parent-child relationships, logical groups,
    and navigation paths between elements.
    """
    
    def __init__(self, spatial_threshold: float = 0.8):
        """
        Initialize the relationship mapper.
        
        Args:
            spatial_threshold: Threshold for spatial relationship determination
        """
        self.spatial_threshold = spatial_threshold
    
    def map_relationships(self, elements: List[UIElement]) -> List[UIElement]:
        """
        Map relationships between UI elements.
        
        Args:
            elements: List of UI elements
            
        Returns:
            Elements with relationship information added
        """
        logger.info(f"Mapping relationships for {len(elements)} elements")
        
        # Identify parent-child relationships
        self._identify_parent_child_relationships(elements)
        
        # Identify logical groups
        groups = self._identify_logical_groups(elements)
        
        # Store group information in element metadata
        for group_id, group_elements in groups.items():
            for element in group_elements:
                if 'groups' not in element.metadata:
                    element.metadata['groups'] = []
                element.metadata['groups'].append(group_id)
        
        # Map navigation paths
        self._map_navigation_paths(elements)
        
        return elements
    
    def _identify_parent_child_relationships(self, elements: List[UIElement]) -> None:
        """
        Identify parent-child relationships between elements.
        
        Args:
            elements: List of UI elements
        """
        # Sort elements by area (larger elements first)
        elements_by_area = sorted(elements, 
                                key=lambda e: e.bounding_box[2] * e.bounding_box[3],
                                reverse=True)
        
        # For each potential parent element
        for parent_idx, parent in enumerate(elements_by_area):
            parent_x, parent_y, parent_w, parent_h = parent.bounding_box
            parent_area = parent_w * parent_h
            
            # Check smaller elements as potential children
            for child_idx, child in enumerate(elements_by_area[parent_idx+1:], parent_idx+1):
                # Skip if already has a parent
                if child.parent_id is not None:
                    continue
                    
                child_x, child_y, child_w, child_h = child.bounding_box
                child_area = child_w * child_h
                
                # If the child is much smaller than the parent
                if child_area > parent_area * 0.8:
                    continue
                
                # Check if child is contained within parent
                if (child_x >= parent_x and
                    child_y >= parent_y and
                    child_x + child_w <= parent_x + parent_w and
                    child_y + child_h <= parent_y + parent_h):
                    
                    # Set parent-child relationship
                    child.parent_id = parent.element_id
                    parent.children_ids.append(child.element_id)
    
    def _identify_logical_groups(self, elements: List[UIElement]) -> Dict[str, List[UIElement]]:
        """
        Identify logical groups of related elements.
        
        Args:
            elements: List of UI elements
            
        Returns:
            Dictionary mapping group IDs to lists of elements
        """
        groups = {}
        group_id_counter = 0
        
        # Group elements by type
        elements_by_type = defaultdict(list)
        for element in elements:
            elements_by_type[element.element_type].append(element)
        
        # Find aligned elements of the same type
        for element_type, type_elements in elements_by_type.items():
            # Skip if only one element of this type
            if len(type_elements) < 2:
                continue
            
            # Find horizontally aligned elements
            horizontal_groups = self._find_aligned_elements(type_elements, 'horizontal')
            for h_group in horizontal_groups:
                if len(h_group) >= 2:
                    group_id = f"group_{group_id_counter}"
                    groups[group_id] = h_group
                    group_id_counter += 1
            
            # Find vertically aligned elements
            vertical_groups = self._find_aligned_elements(type_elements, 'vertical')
            for v_group in vertical_groups:
                if len(v_group) >= 2:
                    group_id = f"group_{group_id_counter}"
                    groups[group_id] = v_group
                    group_id_counter += 1
        
        # Group elements by parent
        parent_groups = defaultdict(list)
        for element in elements:
            if element.parent_id:
                parent_groups[element.parent_id].append(element)
        
        # Add parent-based groups
        for parent_id, children in parent_groups.items():
            if len(children) >= 2:
                group_id = f"group_{group_id_counter}"
                groups[group_id] = children
                group_id_counter += 1
        
        return groups
    
    def _find_aligned_elements(self, elements: List[UIElement], 
                           alignment: str) -> List[List[UIElement]]:
        """
        Find groups of aligned elements.
        
        Args:
            elements: List of UI elements to check
            alignment: Alignment direction ('horizontal' or 'vertical')
            
        Returns:
            List of aligned element groups
        """
        aligned_groups = []
        
        if alignment == 'horizontal':
            # Sort by y-coordinate
            elements_by_y = defaultdict(list)
            for element in elements:
                y = element.bounding_box[1]
                elements_by_y[y].append(element)
            
            # Group elements with similar y-coordinates
            for base_y in sorted(elements_by_y.keys()):
                aligned = elements_by_y[base_y]
                
                # Check nearby y-coordinates
                for y in range(base_y - 10, base_y + 11):
                    if y != base_y and y in elements_by_y:
                        aligned.extend(elements_by_y[y])
                
                if len(aligned) >= 2:
                    # Sort horizontally
                    aligned.sort(key=lambda e: e.bounding_box[0])
                    aligned_groups.append(aligned)
        
        elif alignment == 'vertical':
            # Sort by x-coordinate
            elements_by_x = defaultdict(list)
            for element in elements:
                x = element.bounding_box[0]
                elements_by_x[x].append(element)
            
            # Group elements with similar x-coordinates
            for base_x in sorted(elements_by_x.keys()):
                aligned = elements_by_x[base_x]
                
                # Check nearby x-coordinates
                for x in range(base_x - 10, base_x + 11):
                    if x != base_x and x in elements_by_x:
                        aligned.extend(elements_by_x[x])
                
                if len(aligned) >= 2:
                    # Sort vertically
                    aligned.sort(key=lambda e: e.bounding_box[1])
                    aligned_groups.append(aligned)
        
        return aligned_groups
    
    def _map_navigation_paths(self, elements: List[UIElement]) -> None:
        """
        Map navigation paths between elements (tab order, etc.).
        
        Args:
            elements: List of UI elements
        """
        # Clone the elements list for manipulation
        sorted_elements = elements.copy()
        
        # Sort elements by vertical position, then horizontal
        sorted_elements.sort(key=lambda e: (e.bounding_box[1], e.bounding_box[0]))
        
        # For each element, store its likely next element in tab order
        for i, element in enumerate(sorted_elements):
            if i < len(sorted_elements) - 1:
                next_element = sorted_elements[i + 1]
                
                # Store the next element's ID in metadata
                if 'navigation' not in element.metadata:
                    element.metadata['navigation'] = {}
                
                element.metadata['navigation']['next'] = next_element.element_id
                
                # Also store previous element
                if 'navigation' not in next_element.metadata:
                    next_element.metadata['navigation'] = {}
                
                next_element.metadata['navigation']['previous'] = element.element_id
