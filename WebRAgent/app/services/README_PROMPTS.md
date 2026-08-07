# Prompt Template System for WebRAgent

This document provides an overview of the prompt template system implemented in WebRAgent, which centralizes and standardizes all LLM prompts used throughout the application.

## Key Benefits

1. **Centralized Prompt Management**
   - Single source of truth for all prompts
   - Easy to update and maintain
   - Consistent style and formatting

2. **Enhanced Source Citation**
   - All prompts now include explicit citation requirements
   - Standardized citation format ([Source X] or URLs in parentheses)
   - Clear distinction between context information and general knowledge

3. **Specialized Agent Roles**
   - Defined persona for each task type (decomposition, synthesis, analysis)
   - Task-specific expertise and instructions
   - More focused and effective prompts

4. **Improved Context Handling**
   - Efficient context truncation based on relevance
   - Inclusion of confidence/relevance scores
   - Better handling of URLs and document titles

5. **Standardized Response Formats**
   - Consistent output formatting instructions
   - Clear bullet point guidelines
   - Enhanced readability

## Core Components

The prompt template system consists of the following key components:

### `PromptTemplateService` Class

The central service that manages all prompt templates and provides methods for retrieving and formatting them. Located in `app/services/prompt_template_service.py`.

### Template Categories

- **FORMAT_INSTRUCTIONS**: Reusable formatting guidelines
- **AGENT_ROLES**: Specialized persona definitions
- **SYSTEM_MESSAGES**: System prompts for different contexts
- **INTENT_ANALYSIS**: Query understanding templates
- **DECOMPOSITION**: Query breakdown templates
- **SYNTHESIS**: Answer combination templates
- **WEB_ANALYSIS**: Web content analysis templates
- **TITLE_GENERATION**: Chat title generation templates
- **SPECIALIZED**: Task-specific prompt templates

### Helper Methods

- `get_role()`: Retrieves agent role definitions
- `get_format_instructions()`: Gets formatting instructions
- `get_system_message()`: Retrieves system messages
- `get_decomposition_prompt()`: Formats decomposition prompts
- `get_synthesis_prompt()`: Formats synthesis prompts
- `format_context()`: Optimizes context presentation

## Implementation Example

```python
# Old approach
prompt = f"""
You are an expert at breaking down complex questions.
Original Query: {query}
Please break this query down into 2-4 specific, focused subqueries...
"""

# New approach using the template service
prompt = PromptTemplateService.get_decomposition_prompt(
    query=query,
    type="document_search",
    role=PromptTemplateService.get_role("decomposer")
)
```

## Best Practices

1. **Always use the template service** for new prompts rather than crafting them inline
2. **Add new template categories** when introducing fundamentally new prompt types
3. **Keep prompt guidelines focused** on the specific task at hand
4. **Include citation requirements** in all prompts that use external information
5. **Use role definitions** to establish the right expertise for each task
6. **Format context efficiently** using the `format_context()` method
7. **Standardize output formatting** with the appropriate format instructions

## Future Improvements

- Implement prompt template versioning
- Add A/B testing capabilities for prompt variations
- Create a web interface for non-technical team members to edit prompts
- Add support for more structured output formats (e.g., JSON)