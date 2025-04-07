from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ServiceRegistry:
    """Registry for agent services."""
    
    def __init__(self):
        """Initialize the service registry."""
        self._services: Dict[str, Any] = {}
        logger.debug("ServiceRegistry initialized")
    
    def register(self, name: str, service: Any) -> None:
        """
        Register a service.
        
        Args:
            name: Unique name for the service
            service: The service object to register
            
        Raises:
            ValueError: If the service name is already registered
        """
        if not name:
            raise ValueError("Service name cannot be empty")
            
        if name in self._services:
            raise ValueError(f"Service with name '{name}' is already registered")
            
        self._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a registered service.
        
        Args:
            name: Name of the service to retrieve
            
        Returns:
            The service object or None if not found
        """
        service = self._services.get(name)
        if service is None:
            logger.debug(f"Service not found: {name}")
        return service
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all registered services.
        
        Returns:
            Dictionary of all registered services
        """
        return self._services.copy()
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a service.
        
        Args:
            name: Name of the service to unregister
            
        Returns:
            True if service was unregistered, False if not found
        """
        if name in self._services:
            del self._services[name]
            logger.debug(f"Unregistered service: {name}")
            return True
        
        logger.debug(f"Could not unregister service (not found): {name}")
        return False
    
    def has_service(self, name: str) -> bool:
        """
        Check if a service is registered.
        
        Args:
            name: Name of the service to check
            
        Returns:
            True if service exists, False otherwise
        """
        return name in self._services
    
    def clear(self) -> None:
        """
        Clear all registered services.
        """
        count = len(self._services)
        self._services.clear()
        logger.debug(f"Cleared {count} registered services")
