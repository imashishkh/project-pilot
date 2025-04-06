import os
import re
import json
import time
import base64
import hashlib
import logging
import datetime
import subprocess
import threading
from typing import Dict, List, Any, Set, Optional, Tuple, Union
from pathlib import Path

# Import the permission manager and safety protocol
from .permission_manager import PermissionManager
from .safety_protocol import SafetyProtocol

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

class SecurityMonitor:
    """
    Monitors and enforces security for AI agent operations.
    Provides protection for sensitive information, secure credential storage,
    security risk detection, audit logging, and privacy-preserving mechanisms.
    """
    
    # Security event types
    AUTH_SUCCESS = "authentication_success"
    AUTH_FAILURE = "authentication_failure"
    CRED_ACCESS = "credential_access"
    SENSITIVE_ACCESS = "sensitive_data_access"
    PERMISSION_VIOLATION = "permission_violation"
    CONFIG_CHANGE = "configuration_change"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_EXPORT = "data_export"
    SYSTEM_SCAN = "system_scan"
    
    # Privacy levels
    PRIVACY_PUBLIC = "public"
    PRIVACY_INTERNAL = "internal"
    PRIVACY_CONFIDENTIAL = "confidential"
    PRIVACY_SENSITIVE = "sensitive"
    PRIVACY_SECRET = "secret"
    
    def __init__(self, permission_manager: PermissionManager, 
                safety_protocol: SafetyProtocol,
                config_path: str = "config/security.json",
                logs_path: str = "logs/security",
                secure_storage_path: str = "data/secure"):
        """
        Initialize the SecurityMonitor.
        
        Args:
            permission_manager: PermissionManager instance to check permissions
            safety_protocol: SafetyProtocol instance for safety checks
            config_path: Path to the security configuration file
            logs_path: Path to store security logs
            secure_storage_path: Path for secure storage
        """
        self.permission_manager = permission_manager
        self.safety_protocol = safety_protocol
        self.config_path = config_path
        self.logs_path = logs_path
        self.secure_storage_path = secure_storage_path
        
        # Ensure directories exist
        os.makedirs(logs_path, exist_ok=True)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        os.makedirs(secure_storage_path, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("security_monitor")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        log_file = os.path.join(logs_path, f"security_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to logger
        self.logger.addHandler(file_handler)
        
        # Set up secure audit log
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Add file handler for audit log
        audit_log_file = os.path.join(logs_path, f"audit_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        audit_file_handler = logging.FileHandler(audit_log_file)
        audit_file_handler.setLevel(logging.INFO)
        
        # Add formatter
        audit_file_handler.setFormatter(formatter)
        
        # Add the handler to audit logger
        self.audit_logger.addHandler(audit_file_handler)
        
        # Load or create security config
        self._load_security_config()
        
        # Initialize encryption if available
        self.encryption_key = None
        if CRYPTO_AVAILABLE:
            self._initialize_encryption()
        
        # Security events
        self.security_events: List[Dict[str, Any]] = []
        
        # Suspicious patterns for detection
        self.suspicious_patterns: Dict[str, Any] = self.config.get("suspicious_patterns", {})
        
        # Create a thread for periodic security scans
        self.scan_interval = self.config.get("security_scan_interval_minutes", 60)
        self._start_security_scan_thread()
        
        # Initialize PII detection patterns
        self._initialize_pii_patterns()
    
    def _load_security_config(self) -> None:
        """Load the security configuration from the config file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Create default security configuration
                self.config = self._create_default_config()
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error loading security config: {str(e)}")
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default security configuration."""
        return {
            "security_level": "standard",  # standard, high, paranoid
            "encryption_enabled": CRYPTO_AVAILABLE,
            "secure_deletion_enabled": True,
            "audit_logging_enabled": True,
            "privacy_enabled": True,
            "credential_storage": "keyring" if KEYRING_AVAILABLE else "encrypted_file",
            "pii_detection_enabled": True,
            "security_scan_interval_minutes": 60,
            "max_login_attempts": 5,
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special_chars": True
            },
            "suspicious_patterns": {
                "file_access": [
                    "/etc/shadow",
                    "/etc/passwd",
                    "~/Library/Keychains",
                    "~/.ssh/id_rsa"
                ],
                "commands": [
                    "rm -rf /",
                    "sudo",
                    "chown",
                    "chmod 777",
                    "dd if=/dev/zero"
                ],
                "network": [
                    "api/token",
                    "credentials",
                    "password",
                    "/login"
                ]
            },
            "sensitive_data_types": [
                "password",
                "api_key",
                "token",
                "credit_card",
                "social_security_number",
                "private_key"
            ],
            "privacy_levels": {
                "user_name": "public",
                "email": "internal",
                "phone_number": "confidential",
                "address": "sensitive",
                "financial_data": "secret"
            }
        }
    
    def save_config(self) -> None:
        """Save the current security configuration to the config file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving security config: {str(e)}")
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption for secure storage."""
        try:
            key_file = os.path.join(self.secure_storage_path, ".key")
            
            if os.path.exists(key_file):
                # Load existing key
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                
                # Save key (in a real system, this would be better protected)
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                
                # Set permissions to prevent other users from reading
                os.chmod(key_file, 0o600)
            
            self.cipher = Fernet(self.encryption_key)
            self.logger.info("Encryption initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing encryption: {str(e)}")
            self.encryption_key = None
    
    def _derive_key(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """
        Derive an encryption key from a password using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation (generated if None)
            
        Returns:
            (key, salt): Derived key and salt
        """
        if not salt:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using the system's encryption key.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            encrypted_data: Encrypted data
        """
        if not CRYPTO_AVAILABLE:
            self.logger.warning("Encryption requested but cryptography library not available")
            return data.encode() if isinstance(data, str) else data
        
        if self.encryption_key is None:
            self.logger.warning("Encryption key not initialized")
            return data.encode() if isinstance(data, str) else data
        
        try:
            if isinstance(data, str):
                data = data.encode()
            
            return self.cipher.encrypt(data)
        
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            return data
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt encrypted data using the system's encryption key.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            
        Returns:
            decrypted_data: Decrypted data
        """
        if not CRYPTO_AVAILABLE:
            self.logger.warning("Decryption requested but cryptography library not available")
            return encrypted_data
        
        if self.encryption_key is None:
            self.logger.warning("Encryption key not initialized")
            return encrypted_data
        
        try:
            return self.cipher.decrypt(encrypted_data)
        
        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            return encrypted_data
    
    def store_credential(self, service_name: str, username: str, 
                       password: str) -> bool:
        """
        Securely store a credential.
        
        Args:
            service_name: Name of the service
            username: Username for the credential
            password: Password or secret to store
            
        Returns:
            success: Whether the credential was stored successfully
        """
        # Log the credential access (without the actual credential)
        self.log_security_event(
            self.CRED_ACCESS,
            f"Storing credential for {service_name}",
            {"service": service_name, "username": username}
        )
        
        storage_method = self.config.get("credential_storage", "encrypted_file")
        
        if storage_method == "keyring" and KEYRING_AVAILABLE:
            try:
                keyring.set_password(service_name, username, password)
                return True
            except Exception as e:
                self.logger.error(f"Error storing credential in keyring: {str(e)}")
                # Fall back to encrypted file
        
        # Use encrypted file storage
        try:
            credentials_file = os.path.join(self.secure_storage_path, "credentials.enc")
            
            # Load existing credentials
            credentials = {}
            if os.path.exists(credentials_file):
                with open(credentials_file, 'rb') as f:
                    encrypted_data = f.read()
                    try:
                        decrypted_data = self.decrypt_data(encrypted_data)
                        credentials = json.loads(decrypted_data)
                    except Exception:
                        self.logger.error("Failed to decrypt credentials file")
            
            # Add new credential
            if service_name not in credentials:
                credentials[service_name] = {}
            
            credentials[service_name][username] = password
            
            # Encrypt and save
            encrypted = self.encrypt_data(json.dumps(credentials))
            with open(credentials_file, 'wb') as f:
                f.write(encrypted)
            
            # Set secure permissions
            os.chmod(credentials_file, 0o600)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing credential in encrypted file: {str(e)}")
            return False
    
    def get_credential(self, service_name: str, username: str) -> Optional[str]:
        """
        Retrieve a securely stored credential.
        
        Args:
            service_name: Name of the service
            username: Username for the credential
            
        Returns:
            password: The stored password or None if not found
        """
        # Log the credential access (without the actual credential)
        self.log_security_event(
            self.CRED_ACCESS,
            f"Retrieving credential for {service_name}",
            {"service": service_name, "username": username}
        )
        
        storage_method = self.config.get("credential_storage", "encrypted_file")
        
        if storage_method == "keyring" and KEYRING_AVAILABLE:
            try:
                password = keyring.get_password(service_name, username)
                if password:
                    return password
            except Exception as e:
                self.logger.error(f"Error retrieving credential from keyring: {str(e)}")
                # Fall back to encrypted file
        
        # Use encrypted file storage
        try:
            credentials_file = os.path.join(self.secure_storage_path, "credentials.enc")
            
            if not os.path.exists(credentials_file):
                return None
            
            # Load and decrypt credentials
            with open(credentials_file, 'rb') as f:
                encrypted_data = f.read()
                
                try:
                    decrypted_data = self.decrypt_data(encrypted_data)
                    credentials = json.loads(decrypted_data)
                    
                    if service_name in credentials and username in credentials[service_name]:
                        return credentials[service_name][username]
                
                except Exception as e:
                    self.logger.error(f"Error decrypting credentials: {str(e)}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving credential from encrypted file: {str(e)}")
            return None
    
    def delete_credential(self, service_name: str, username: str) -> bool:
        """
        Delete a securely stored credential.
        
        Args:
            service_name: Name of the service
            username: Username for the credential
            
        Returns:
            success: Whether the credential was deleted successfully
        """
        # Log the credential deletion
        self.log_security_event(
            self.CRED_ACCESS,
            f"Deleting credential for {service_name}",
            {"service": service_name, "username": username}
        )
        
        storage_method = self.config.get("credential_storage", "encrypted_file")
        
        if storage_method == "keyring" and KEYRING_AVAILABLE:
            try:
                keyring.delete_password(service_name, username)
                return True
            except Exception as e:
                self.logger.error(f"Error deleting credential from keyring: {str(e)}")
                # Continue to also check encrypted file
        
        # Also check encrypted file storage
        try:
            credentials_file = os.path.join(self.secure_storage_path, "credentials.enc")
            
            if not os.path.exists(credentials_file):
                return False
            
            # Load and decrypt credentials
            with open(credentials_file, 'rb') as f:
                encrypted_data = f.read()
                
                try:
                    decrypted_data = self.decrypt_data(encrypted_data)
                    credentials = json.loads(decrypted_data)
                    
                    if service_name in credentials and username in credentials[service_name]:
                        # Remove the credential
                        del credentials[service_name][username]
                        
                        # Remove the service if no more credentials
                        if not credentials[service_name]:
                            del credentials[service_name]
                        
                        # Encrypt and save
                        encrypted = self.encrypt_data(json.dumps(credentials))
                        with open(credentials_file, 'wb') as f:
                            f.write(encrypted)
                        
                        return True
                
                except Exception as e:
                    self.logger.error(f"Error updating credentials file: {str(e)}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting credential from encrypted file: {str(e)}")
            return False
    
    def is_sensitive_data(self, data: str) -> Tuple[bool, Optional[str]]:
        """
        Check if data contains sensitive information like passwords or API keys.
        
        Args:
            data: Data to check
            
        Returns:
            (is_sensitive, data_type): Whether data is sensitive and what type
        """
        # Simple pattern matching for sensitive data
        sensitive_patterns = {
            "password": r"password[=:]\S+|pwd[=:]\S+",
            "api_key": r"api[-_]?key[=:]\S+|api[-_]?secret[=:]\S+",
            "token": r"token[=:]\S+|oauth[=:]\S+|bearer[=:]\S+",
            "credit_card": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
            "social_security_number": r"\d{3}[-\s]?\d{2}[-\s]?\d{4}",
            "private_key": r"-----BEGIN PRIVATE KEY-----"
        }
        
        for data_type, pattern in sensitive_patterns.items():
            if re.search(pattern, data, re.IGNORECASE):
                return True, data_type
        
        return False, None
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect personally identifiable information (PII) in text.
        
        Args:
            text: Text to check for PII
            
        Returns:
            detected_pii: List of PII instances found
        """
        if not self.config.get("pii_detection_enabled", True):
            return []
        
        pii_found = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                pii_found.append({
                    "type": pii_type,
                    "value": match.group(),
                    "position": (match.start(), match.end()),
                    "privacy_level": self.config.get("privacy_levels", {}).get(pii_type, self.PRIVACY_CONFIDENTIAL)
                })
        
        return pii_found
    
    def _initialize_pii_patterns(self) -> None:
        """Initialize regex patterns for PII detection."""
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_number": r'\b(\+\d{1,2}\s?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "social_security_number": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "address": r'\b\d+\s+[A-Za-z0-9\s,]+\b(?:avenue|ave|boulevard|blvd|circle|cir|court|ct|drive|dr|lane|ln|parkway|pkwy|place|pl|road|rd|square|sq|street|st|way)\b',
            "date_of_birth": r'\b(0[1-9]|1[0-2])[/.-](0[1-9]|[12]\d|3[01])[/.-](19|20)\d{2}\b',
            "user_name": r'@[A-Za-z0-9_]+'
        }
    
    def redact_sensitive_data(self, text: str, privacy_level: str = None) -> str:
        """
        Redact sensitive information from text based on privacy level.
        
        Args:
            text: Text potentially containing sensitive data
            privacy_level: Minimum privacy level to redact
            
        Returns:
            redacted_text: Text with sensitive information redacted
        """
        if not self.config.get("privacy_enabled", True):
            return text
        
        # Default privacy threshold
        if privacy_level is None:
            privacy_level = self.PRIVACY_CONFIDENTIAL
        
        # Privacy level hierarchy
        privacy_levels = [
            self.PRIVACY_PUBLIC,
            self.PRIVACY_INTERNAL,
            self.PRIVACY_CONFIDENTIAL,
            self.PRIVACY_SENSITIVE,
            self.PRIVACY_SECRET
        ]
        
        # Find threshold index
        try:
            threshold_index = privacy_levels.index(privacy_level)
        except ValueError:
            threshold_index = 2  # Default to CONFIDENTIAL if level not found
        
        # Detect PII
        detected_pii = self.detect_pii(text)
        
        # Sort PII by position (end position descending)
        # This allows us to replace from end to beginning, maintaining correct positions
        detected_pii.sort(key=lambda p: p["position"][1], reverse=True)
        
        # Redact PII based on privacy level
        result = text
        for pii in detected_pii:
            pii_level = pii["privacy_level"]
            try:
                pii_level_index = privacy_levels.index(pii_level)
            except ValueError:
                pii_level_index = 2  # Default to CONFIDENTIAL
            
            if pii_level_index >= threshold_index:
                # Redact this PII
                start, end = pii["position"]
                replacement = f"[REDACTED:{pii['type']}]"
                result = result[:start] + replacement + result[end:]
        
        return result
    
    def securely_delete_file(self, file_path: str) -> bool:
        """
        Securely delete a file by overwriting its contents before deletion.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            success: Whether the file was securely deleted
        """
        if not self.config.get("secure_deletion_enabled", True):
            # Just perform regular delete
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                self.logger.error(f"Error deleting file: {str(e)}")
                return False
        
        try:
            if not os.path.exists(file_path):
                return False
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Open file for binary writing
            with open(file_path, 'wb') as f:
                # Pass 1: Overwrite with zeros
                f.write(b'\x00' * file_size)
                f.flush()
                os.fsync(f.fileno())
                
                # Pass 2: Overwrite with ones
                f.seek(0)
                f.write(b'\xFF' * file_size)
                f.flush()
                os.fsync(f.fileno())
                
                # Pass 3: Overwrite with random data
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
            
            # Delete the file
            os.remove(file_path)
            
            self.logger.info(f"Securely deleted file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error securely deleting file: {str(e)}")
            return False
    
    def log_security_event(self, event_type: str, description: str, 
                         details: Dict[str, Any] = None) -> None:
        """
        Log a security-related event.
        
        Args:
            event_type: Type of security event
            description: Description of the event
            details: Additional details about the event
        """
        timestamp = datetime.datetime.now().isoformat()
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "description": description,
            "details": details or {}
        }
        
        # Add to in-memory events
        self.security_events.append(event)
        
        # Log to audit log
        if self.config.get("audit_logging_enabled", True):
            redacted_details = None
            if details:
                # Redact sensitive information from details before logging
                redacted_details = {}
                for key, value in details.items():
                    if isinstance(value, str):
                        redacted_details[key] = self.redact_sensitive_data(value)
                    else:
                        redacted_details[key] = value
            
            self.audit_logger.info(
                f"{event_type}: {description} - " +
                (f"{redacted_details}" if redacted_details else "")
            )
    
    def get_security_events(self, event_type: str = None, 
                          start_time: datetime.datetime = None,
                          end_time: datetime.datetime = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get security events filtered by various criteria.
        
        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of events to return
            
        Returns:
            events: List of security events
        """
        filtered_events = self.security_events.copy()
        
        if event_type:
            filtered_events = [e for e in filtered_events if e["event_type"] == event_type]
        
        if start_time:
            filtered_events = [e for e in filtered_events 
                            if datetime.datetime.fromisoformat(e["timestamp"]) >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events 
                            if datetime.datetime.fromisoformat(e["timestamp"]) <= end_time]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e["timestamp"], reverse=True)
        
        return filtered_events[:limit]
    
    def is_suspicious_activity(self, activity_type: str, 
                             resource: str) -> Tuple[bool, str]:
        """
        Check if an activity might be suspicious or malicious.
        
        Args:
            activity_type: Type of activity (file_access, command, network)
            resource: Resource being accessed
            
        Returns:
            (is_suspicious, reason): Whether activity is suspicious and why
        """
        suspicious_patterns = self.suspicious_patterns.get(activity_type, [])
        
        for pattern in suspicious_patterns:
            if pattern in resource:
                reason = f"Matches suspicious pattern: {pattern}"
                
                # Log the suspicious activity
                self.log_security_event(
                    self.SUSPICIOUS_ACTIVITY,
                    f"Suspicious {activity_type} detected",
                    {"resource": resource, "pattern": pattern}
                )
                
                return True, reason
        
        return False, ""
    
    def _start_security_scan_thread(self) -> None:
        """Start a thread for periodic security scans."""
        def security_scan_thread():
            while True:
                try:
                    # Wait for the configured interval
                    time.sleep(self.scan_interval * 60)
                    
                    # Run security scan
                    self.run_security_scan()
                    
                except Exception as e:
                    self.logger.error(f"Error in security scan: {str(e)}")
        
        scan_thread = threading.Thread(target=security_scan_thread, daemon=True)
        scan_thread.start()
    
    def run_security_scan(self) -> Dict[str, Any]:
        """
        Run a security scan of the system.
        
        Returns:
            results: Results of the security scan
        """
        self.logger.info("Starting security scan")
        self.log_security_event(
            self.SYSTEM_SCAN,
            "Starting routine security scan"
        )
        
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "issues_found": [],
            "scan_complete": False
        }
        
        try:
            # Check permissions configuration
            if not hasattr(self.permission_manager, "permissions"):
                results["issues_found"].append({
                    "type": "configuration",
                    "severity": "high",
                    "description": "Permission manager not properly configured"
                })
            
            # Check emergency stop functionality
            if not hasattr(self.safety_protocol, "emergency_stop_flag"):
                results["issues_found"].append({
                    "type": "safety",
                    "severity": "high",
                    "description": "Emergency stop functionality not available"
                })
            
            # Check encryption
            if self.config.get("encryption_enabled", True) and not CRYPTO_AVAILABLE:
                results["issues_found"].append({
                    "type": "security",
                    "severity": "medium",
                    "description": "Encryption enabled but not available"
                })
            
            # Check secure storage permissions
            storage_path = Path(self.secure_storage_path)
            if storage_path.exists():
                storage_permissions = storage_path.stat().st_mode & 0o777
                if storage_permissions > 0o700:
                    results["issues_found"].append({
                        "type": "file_permissions",
                        "severity": "medium",
                        "description": f"Secure storage has weak permissions: {oct(storage_permissions)}"
                    })
            
            # Check log file permissions
            logs_path = Path(self.logs_path)
            if logs_path.exists():
                for log_file in logs_path.glob("*.log"):
                    log_permissions = log_file.stat().st_mode & 0o777
                    if log_permissions > 0o640:
                        results["issues_found"].append({
                            "type": "file_permissions",
                            "severity": "low",
                            "description": f"Log file {log_file.name} has weak permissions: {oct(log_permissions)}"
                        })
            
            results["scan_complete"] = True
            
            # Log scan completion
            self.log_security_event(
                self.SYSTEM_SCAN,
                f"Security scan completed, found {len(results['issues_found'])} issues",
                {"issues_count": len(results["issues_found"])}
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during security scan: {str(e)}")
            results["error"] = str(e)
            return results
    
    def validate_data_export(self, data: Any, destination: str, 
                           privacy_level: str = None) -> Tuple[bool, Any]:
        """
        Validate and sanitize data before export or sharing.
        
        Args:
            data: Data to be exported
            destination: Where the data will be sent
            privacy_level: Privacy level to enforce
            
        Returns:
            (is_valid, sanitized_data): Whether export is valid and sanitized data
        """
        # Log the export attempt
        self.log_security_event(
            self.DATA_EXPORT,
            f"Data export to {destination}",
            {"destination": destination, "privacy_level": privacy_level}
        )
        
        # Default privacy level
        if privacy_level is None:
            privacy_level = self.PRIVACY_INTERNAL
        
        # Check if destination is allowed
        # This could be expanded with a whitelist of destinations
        is_suspicious, reason = self.is_suspicious_activity("network", destination)
        if is_suspicious:
            self.logger.warning(f"Suspicious data export destination: {destination}")
            return False, None
        
        # Handle different data types
        if isinstance(data, str):
            # Redact sensitive information
            sanitized_data = self.redact_sensitive_data(data, privacy_level)
            return True, sanitized_data
            
        elif isinstance(data, dict):
            # Recursively sanitize dict
            sanitized_data = {}
            for key, value in data.items():
                # Skip entirely sensitive keys
                sensitive_keys = self.config.get("sensitive_data_types", [])
                if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                    sanitized_data[key] = "[REDACTED]"
                    continue
                
                # Recursively sanitize
                _, sanitized_value = self.validate_data_export(value, destination, privacy_level)
                sanitized_data[key] = sanitized_value
            
            return True, sanitized_data
            
        elif isinstance(data, list):
            # Recursively sanitize list
            sanitized_data = []
            for item in data:
                _, sanitized_item = self.validate_data_export(item, destination, privacy_level)
                sanitized_data.append(sanitized_item)
            
            return True, sanitized_data
            
        else:
            # Other types are passed through unchanged
            return True, data
