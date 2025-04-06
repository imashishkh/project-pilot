import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

class WorkflowAutomator:
    """
    A class that handles recording, editing, and executing multi-step workflows 
    across different applications on macOS.
    """
    
    def __init__(self, workflows_dir: str = "workflows"):
        """
        Initialize the WorkflowAutomator.
        
        Args:
            workflows_dir: Directory to store workflow files
        """
        self.workflows_dir = workflows_dir
        os.makedirs(workflows_dir, exist_ok=True)
        self.current_workflow = None
        self.recording = False
        self.recording_start_time = None
    
    def start_recording(self, workflow_name: str) -> str:
        """
        Start recording a new workflow.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            workflow_id: Unique ID for the workflow
        """
        if self.recording:
            raise RuntimeError("Already recording a workflow")
        
        workflow_id = str(uuid.uuid4())
        self.current_workflow = {
            "id": workflow_id,
            "name": workflow_name,
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "steps": []
        }
        self.recording = True
        self.recording_start_time = time.time()
        return workflow_id
    
    def record_step(self, 
                   app_name: str, 
                   action: str, 
                   parameters: Dict[str, Any] = None,
                   wait_time: Optional[float] = None) -> int:
        """
        Record a step in the current workflow.
        
        Args:
            app_name: Name of the application
            action: Action to perform
            parameters: Parameters for the action
            wait_time: Time to wait after executing this step (in seconds)
            
        Returns:
            step_index: Index of the recorded step
        """
        if not self.recording:
            raise RuntimeError("Not currently recording a workflow")
        
        step = {
            "app_name": app_name,
            "action": action,
            "parameters": parameters or {},
            "wait_time": wait_time,
            "timestamp": time.time() - self.recording_start_time
        }
        
        self.current_workflow["steps"].append(step)
        return len(self.current_workflow["steps"]) - 1
    
    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording the current workflow and save it.
        
        Returns:
            workflow: The recorded workflow
        """
        if not self.recording:
            raise RuntimeError("Not currently recording a workflow")
        
        self.recording = False
        self.current_workflow["modified_at"] = datetime.now().isoformat()
        self.save_workflow(self.current_workflow)
        
        result = self.current_workflow
        self.current_workflow = None
        self.recording_start_time = None
        
        return result
    
    def save_workflow(self, workflow: Dict[str, Any]) -> None:
        """
        Save a workflow to disk.
        
        Args:
            workflow: Workflow to save
        """
        file_path = os.path.join(self.workflows_dir, f"{workflow['id']}.json")
        with open(file_path, 'w') as f:
            json.dump(workflow, f, indent=2)
    
    def load_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Load a workflow from disk.
        
        Args:
            workflow_id: ID of the workflow to load
            
        Returns:
            workflow: The loaded workflow
        """
        file_path = os.path.join(self.workflows_dir, f"{workflow_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Workflow {workflow_id} not found")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all available workflows.
        
        Returns:
            workflows: List of workflow metadata
        """
        workflows = []
        for filename in os.listdir(self.workflows_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.workflows_dir, filename)
                with open(file_path, 'r') as f:
                    workflow = json.load(f)
                    workflows.append({
                        "id": workflow["id"],
                        "name": workflow["name"],
                        "created_at": workflow["created_at"],
                        "modified_at": workflow["modified_at"],
                        "step_count": len(workflow["steps"])
                    })
        return workflows
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow.
        
        Args:
            workflow_id: ID of the workflow to delete
            
        Returns:
            success: Whether the deletion was successful
        """
        file_path = os.path.join(self.workflows_dir, f"{workflow_id}.json")
        if not os.path.exists(file_path):
            return False
        
        os.remove(file_path)
        return True
    
    def edit_workflow(self, workflow_id: str, edits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Edit a workflow's metadata or steps.
        
        Args:
            workflow_id: ID of the workflow to edit
            edits: Edits to apply to the workflow
            
        Returns:
            workflow: The updated workflow
        """
        workflow = self.load_workflow(workflow_id)
        
        # Apply edits to metadata
        for key, value in edits.items():
            if key != "steps" and key in workflow:
                workflow[key] = value
        
        # Apply edits to steps if provided
        if "steps" in edits:
            workflow["steps"] = edits["steps"]
        
        workflow["modified_at"] = datetime.now().isoformat()
        self.save_workflow(workflow)
        
        return workflow
    
    def execute_workflow(self, workflow_id: str, app_controller=None) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            app_controller: Controller for interacting with applications
            
        Returns:
            result: Execution result
        """
        if app_controller is None:
            raise ValueError("app_controller must be provided")
        
        workflow = self.load_workflow(workflow_id)
        result = {
            "workflow_id": workflow_id,
            "started_at": datetime.now().isoformat(),
            "steps_executed": 0,
            "success": True,
            "steps_results": []
        }
        
        try:
            for i, step in enumerate(workflow["steps"]):
                step_result = {
                    "step_index": i,
                    "app_name": step["app_name"],
                    "action": step["action"],
                    "success": False,
                    "error": None
                }
                
                try:
                    # Execute the step using the app_controller
                    app = app_controller.get_app(step["app_name"])
                    action_method = getattr(app, step["action"])
                    step_output = action_method(**step["parameters"])
                    
                    step_result["success"] = True
                    step_result["output"] = step_output
                    
                    # Wait if specified
                    if step.get("wait_time"):
                        time.sleep(step["wait_time"])
                        
                except Exception as e:
                    step_result["success"] = False
                    step_result["error"] = str(e)
                    result["success"] = False
                    result["error"] = f"Failed at step {i}: {str(e)}"
                    
                result["steps_results"].append(step_result)
                result["steps_executed"] += 1
                
                # Stop execution if a step failed
                if not step_result["success"]:
                    break
                    
        except Exception as e:
            result["success"] = False
            result["error"] = f"Workflow execution failed: {str(e)}"
            
        result["completed_at"] = datetime.now().isoformat()
        return result
