import os
import re
from textwrap import dedent
from app.services.llm_service import LLMFactory
from app.services.prompt_template_service import PromptTemplateService

class MindmapService:
    """Service for generating mindmaps from user queries and context"""
    
    def __init__(self):
        """Initialize mindmap service"""
        self.llm_service = LLMFactory.create_llm_service()
    
    def generate_mindmap(self, query, context=None):
        """
        Generate a mindmap for the given query and context
        
        Args:
            query (str): User query
            context (str, optional): Contextual information
            
        Returns:
            str: HTML content for rendering the mindmap
        """
        # Generate PlantUML mindmap using LLM
        plantuml_mindmap = self._generate_plantuml_mindmap(query, context)
        
        # Convert to markmap format
        markmap_content = self._convert_to_markmap(plantuml_mindmap)
        
        # Generate HTML with markmap
        html_content = self._create_markmap_html(markmap_content)
        
        return html_content
    
    def _generate_plantuml_mindmap(self, query, context=None):
        """
        Ask the LLM to generate a PlantUML mindmap
        """
        system_prompt = """
From now on you will behave as "MapGPT" and, for every text the user will submit, you are going to create a PlantUML mind map file for the inputted text to best describe main ideas. Format it as a code and remember that the mind map should be in the same language as the inputted context. You don't have to provide a general example for the mind map format before the user inputs the text.
        """
        
        mindmap_prompt = f"""
Question:
{query}

Context:
{context if context else 'No additional context provided.'}

Generate a sample PlantUML mindmap for based on the provided question and context above. Only includes context relevant to the question to produce the mindmap.

Use the template like this:

@startmindmap
* Title
** Item A
*** Item B
**** Item C
*** Item D
@endmindmap
        """
        
        # Get response from LLM
        response = self.llm_service.generate_response(
            prompt=mindmap_prompt,
            context=None,  # Context is already included in the prompt
            max_tokens=1000
        )
        
        # Extract PlantUML code from response
        return self._extract_plantuml_code(response)
    
    def _extract_plantuml_code(self, response):
        """Extract the PlantUML code from the LLM response"""
        # Look for code between @startmindmap and @endmindmap
        pattern = r'@startmindmap(.*?)@endmindmap'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return f"@startmindmap{match.group(1)}@endmindmap"
        
        # Fallback: if no match, return a simple mindmap
        return """@startmindmap
* Query Mindmap
** No mindmap could be generated
*** Please try again with a different query
@endmindmap"""
    
    def _convert_to_markmap(self, plantuml_mindmap):
        """
        Convert PlantUML mindmap to Markmap markdown format
        """
        # Remove @startmindmap and @endmindmap
        content = plantuml_mindmap.replace('@startmindmap', '').replace('@endmindmap', '').strip()
        
        # Convert PlantUML syntax to Markdown
        lines = content.split('\n')
        markdown_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Count the number of * or + at the beginning
            count = 0
            for char in stripped:
                if char in ['*', '+']:
                    count += 1
                else:
                    break
                    
            # Replace the * or + with markdown heading level
            if count > 0:
                heading = '#' * count
                text = stripped[count:].strip()
                markdown_lines.append(f"{heading} {text}")
        
        return '\n'.join(markdown_lines)
    
    def _create_markmap_html(self, markdown_content):
        """
        Create HTML with embedded markmap
        """
        # Format the HTML with the markmap content
        markmap_div = f'<div class="markmap"><script type="text/template">{markdown_content}</script></div>'
        
        html_template = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Mindmap</title>
    <style>
      svg.markmap {{
        width: 100%;
        height: 400px;
      }}
      body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        height: 100%;
      }}
      html {{
        height: 100%;
      }}
      .markmap {{
        height: 100%;
      }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/markmap-autoloader@0.18.1"></script>
  </head>
  <body>
    {markmap_div}
  </body>
</html>"""
        
        return html_template.format(markmap_div=markmap_div)