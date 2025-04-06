# MacAgent

MacAgent is an AI-powered agent system for macOS that helps automate tasks, organize information, and assist users with complex workflows.

## Intelligence Module

The intelligence module (`src/intelligence`) provides the core AI capabilities for MacAgent, including:

- **LLM Integration**: Connects to large language models like GPT-4 via OpenAI and other providers
- **Planning System**: Creates, manages, and adapts plans for complex tasks
- **Decision-Making**: Evaluates options and makes optimal decisions based on criteria
- **Execution Monitoring**: Tracks task execution and handles deviations

### Key Components

The planning system consists of three main components:

1. **TaskPlanner**: Creates structured task plans from high-level goals and breaks them down into manageable steps
2. **DecisionMaker**: Evaluates multiple options against specified criteria to select the best approach
3. **ExecutionMonitor**: Tracks task execution, detects deviations, and provides feedback

## Example Usage

Check out the `examples/planning_system_example.py` script to see how these components work together.

### Running the Example

```bash
# From the project root directory
python examples/planning_system_example.py
```

The example demonstrates:
- Creating a task plan for organizing image files
- Making implementation decisions about categorization approaches
- Monitoring task execution and handling failures
- Updating plans based on new context

## Setting Up Development Environment

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/MacAgent.git
   cd MacAgent
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
MacAgent/
├── data/
│   ├── plans/             # Stored task plans
│   ├── decisions/         # Stored decision records
│   ├── execution_results/ # Execution monitoring results
│   └── prompts/           # Prompt templates
├── examples/
│   └── planning_system_example.py
├── src/
│   ├── intelligence/
│   │   ├── __init__.py
│   │   ├── llm_connector.py
│   │   ├── prompt_manager.py
│   │   ├── task_planner.py
│   │   ├── decision_maker.py
│   │   └── execution_monitor.py
│   └── ...
└── tests/
```

## Intelligence Module Details

### TaskPlanner

The `TaskPlanner` creates structured plans for complex tasks:

```python
from MacAgent.src.intelligence.task_planner import TaskPlanner, PlanningStrategy

# Create a plan
plan = await task_planner.create_plan(
    goal="Organize image files across multiple directories",
    context={"user_preferences": {...}},
    strategy=PlanningStrategy.SEQUENTIAL
)

# Update a plan based on new context
updated_plan = await task_planner.update_plan(
    plan_id=plan.id,
    context=new_context
)
```

### DecisionMaker

The `DecisionMaker` helps evaluate options and select the best approach:

```python
from MacAgent.src.intelligence.decision_maker import DecisionMaker, DecisionCriteria

# Analyze options
decision = await decision_maker.analyze_options(
    question="What approach should be used to categorize images?",
    options=["AI-based", "Metadata-based", "Hybrid approach"],
    criteria=[DecisionCriteria.EFFICIENCY, DecisionCriteria.RELIABILITY],
    criteria_weights={"efficiency": 0.8, "reliability": 1.0}
)

# Make a decision
decision = await decision_maker.make_decision(decision.id)
selected_option = decision.options[decision.selected_option_id]
```

### ExecutionMonitor

The `ExecutionMonitor` tracks task execution and handles deviations:

```python
from MacAgent.src.intelligence.execution_monitor import ExecutionMonitor

# Generate expected outcome
expected_outcome = await execution_monitor.generate_expected_outcome(
    task_id=task.id,
    task_description=task.description
)

# Start monitoring
result = execution_monitor.start_execution(
    task_id=task.id,
    expected_outcome_id=expected_outcome.id
)

# Complete execution
result = execution_monitor.complete_execution(
    result_id=result.id,
    success=True,
    output=output_data
)

# Verify outcome
result = await execution_monitor.verify_outcome(result.id)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 