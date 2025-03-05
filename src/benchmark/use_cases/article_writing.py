"""
Collaborative Article Writing use case.

This use case involves multiple agents collaborating to write an article:
- Supervisor (Editor) Agent: Oversees the process
- Researcher Agent: Gathers factual information
- Coder Agent: Provides code examples
- Writer Agent: Drafts the article

The agents work together to produce a coherent article on a given topic.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional

from src.benchmark.use_cases.base_use_case import BaseUseCase
from src.utils.metrics import PerformanceTracker
from src.utils.llm import LLMClientFactory

class ArticleWritingUseCase(BaseUseCase):
    """
    Collaborative Article Writing use case implementation.
    """
    
    def __init__(self):
        """Initialize the Article Writing use case."""
        super().__init__(
            name="article_writing",
            description="Multiple agents collaborate to research, code, and write an article"
        )
        self.default_topic = "How encryption works in web security with a simple Python example"
    
    def _get_topic(self) -> str:
        """Get the topic from inputs or use default."""
        return self.inputs.get("topic", self.default_topic)
    
    def _get_llm_provider(self) -> str:
        """Get the LLM provider from inputs or use default."""
        return self.inputs.get("llm_provider", "openai")
    
    def _get_model(self) -> str:
        """Get the model from inputs or use default."""
        return self.inputs.get("model", os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo"))
    
    def run_autogen(self, **kwargs) -> Dict[str, Any]:
        """
        Run the article writing use case with AutoGen.
        """
        try:
            print("Starting AutoGen implementation...")
            print("Importing AutoGen modules...")
            import autogen
            print(f"AutoGen version: {autogen.__version__}")
            print("Successfully imported AutoGen modules")
            
            # Initialize performance tracker
            tracker = PerformanceTracker("autogen", "article_writing")
            tracker.start()
            
            # Get the topic from inputs
            topic = self._get_topic()
            
            # Get the model
            model = self._get_model()
            
            # Check if API key is available
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("OPENAI_API_KEY not found in environment variables")
                return {"error": "OPENAI_API_KEY not found in environment variables"}
            
            print(f"Using model: {model}")
            print(f"API key found (first 5 chars): {api_key[:5]}...")
            
            # Configure LLM
            config_list = [
                {
                    "model": model,
                    "api_key": api_key,
                }
            ]
            
            # Create a token usage tracker
            token_usage_tracker = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            print("Creating agent...")
            
            # Create a single assistant agent
            article_writer = autogen.AssistantAgent(
                name="ArticleWriter",
                llm_config={
                    "config_list": config_list,
                },
                system_message=f"You are an expert article writer specializing in technical topics. You will write a comprehensive, well-structured article about '{topic}'. Include relevant Python code examples where appropriate. The article should be engaging, informative, and suitable for a technical audience. Structure the article with clear sections including an introduction, main content, and conclusion."
            )
            
            # Create a user proxy agent
            user_proxy = autogen.UserProxyAgent(
                name="User",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config={"use_docker": False}
            )
            
            # Define the workflow
            def write_article():
                try:
                    print("Starting article writing workflow...")
                    
                    # User initiates the task with a detailed prompt
                    prompt = f"Write a comprehensive article about '{topic}'. The article should: 1. Begin with an engaging introduction that explains the significance of {topic}. 2. Include multiple sections covering different aspects of the topic. 3. Provide real-world examples and applications. 4. Include at least one Python code example that demonstrates a relevant concept. 5. End with a conclusion summarizing the key points. Make the article informative, well-structured, and engaging for a technical audience."
                    
                    print("Sending prompt to article writer...")
                    user_proxy.initiate_chat(
                        article_writer,
                        message=prompt
                    )
                    
                    # Get the article from the assistant's response
                    if article_writer.chat_messages[user_proxy]:
                        final_article = article_writer.chat_messages[user_proxy][-1]["content"]
                        print(f"Article generated. Length: {len(final_article)} characters")
                        
                        # Manually estimate token usage
                        prompt_tokens = tracker.count_tokens(prompt)
                        completion_tokens = tracker.count_tokens(final_article)
                        
                        # Update token usage
                        tracker.update_token_usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens
                        )
                        
                        return final_article
                    else:
                        error_msg = "No response received from article writer"
                        print(error_msg)
                        return error_msg
                
                except Exception as e:
                    error_msg = f"Error in article writing workflow: {str(e)}"
                    print(error_msg)
                    return error_msg
            
            # Execute the workflow
            final_article = write_article()
            print("Article generation completed")
            
            # Stop the tracker and calculate metrics
            execution_time = tracker.stop()
            
            tracker.calculate_cost(model)
            
            # Set additional metadata
            tracker.set_metadata("topic", topic)
            
            return {
                "framework": "autogen",
                "use_case": "article_writing",
                "final_output": final_article,
                "execution_time": execution_time,
                "token_usage": tracker.token_usage,
                "cost": tracker.cost,
                "model": model,
                "topic": topic
            }
        except ImportError as e:
            print(f"Import error: {str(e)}")
            return {"error": f"Import error: {str(e)}"}
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def run_semantic_kernel(self, **kwargs) -> Dict[str, Any]:
        """
        Run the Article Writing use case using Semantic Kernel.
        
        Returns:
            Dictionary with results
        """
        try:
            # Import Semantic Kernel
            import semantic_kernel as sk
            from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
            from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
            from semantic_kernel.contents import ChatHistory
        except ImportError:
            return {"error": "Semantic Kernel not installed. Install with 'pip install semantic-kernel'"}
        
        # Initialize performance tracker
        tracker = PerformanceTracker("semantic_kernel", self.name)
        tracker.start()
        
        # Get inputs
        topic = self._get_topic()
        llm_provider = self._get_llm_provider()
        model = self._get_model()
        
        # Initialize the kernel
        kernel = sk.Kernel()
        
        # Add chat service
        if llm_provider == "openai":
            chat_service = OpenAIChatCompletion(
                service_id="chat",
                ai_model_id=model,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            kernel.add_service(chat_service)
        else:
            # For other providers, we'd need to implement their connectors
            return {"error": f"LLM provider {llm_provider} not supported in Semantic Kernel implementation"}
        
        # Create a function to get completions and track metrics
        async def get_completion(role: str, prompt: str, system_message: str) -> str:
            # Create execution settings
            settings = OpenAIChatPromptExecutionSettings(
                service_id="chat",
                ai_model_id=model,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Create chat history
            history = ChatHistory()
            history.add_system_message(system_message)
            history.add_user_message(prompt)
            
            # Get completion
            chat_service = kernel.get_service("chat")
            result = await chat_service.get_chat_message_content(
                chat_history=history,
                settings=settings
            )
            
            response_text = result.content
            
            # Track metrics
            tracker.add_response(response_text)
            
            # Update token usage if available
            if hasattr(result, "usage") and result.usage:
                tracker.update_token_usage(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens
                )
            else:
                # Estimate tokens if not available
                prompt_tokens = tracker.count_tokens(prompt) + tracker.count_tokens(system_message)
                completion_tokens = tracker.count_tokens(response_text)
                tracker.update_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
            
            return response_text
        
        async def run_article_team() -> str:
            # Step 1: Supervisor plans the task
            supervisor_system = "You are the Supervisor, an editor overseeing an article writing task."
            plan = await get_completion(
                "Supervisor",
                f"Write an article on: {topic}. Plan the steps for your team.",
                supervisor_system
            )
            
            # Step 2: Researcher gathers information
            researcher_system = "You are a Researcher with expertise in the topic."
            facts = await get_completion(
                "Researcher",
                f"The supervisor said: {plan}\nResearch the topic and provide key points.",
                researcher_system
            )
            
            # Step 3: Coder provides a code example
            coder_system = "You are a Coder. Provide relevant code examples in Python."
            code = await get_completion(
                "Coder",
                f"The topic is: {topic}\nThe researcher found: {facts}\nProvide a Python code example relevant to this topic.",
                coder_system
            )
            
            # Step 4: Writer drafts the article
            writer_system = "You are a Writer. Draft a cohesive article with the provided information and code."
            draft = await get_completion(
                "Writer",
                f"Write an article on '{topic}' using the info and code below.\nInformation: {facts}\nCode:\n{code}\n",
                writer_system
            )
            
            # Step 5: Supervisor reviews and finalizes
            final_article = await get_completion(
                "Supervisor",
                f"Review and refine this draft if needed:\n{draft}",
                supervisor_system
            )
            
            return final_article
        
        # Run the workflow
        final_article = asyncio.run(run_article_team())
        
        # Stop the tracker and calculate metrics
        execution_time = tracker.stop()
        tracker.calculate_cost(model)
        
        # Set additional metadata
        tracker.set_metadata("topic", topic)
        tracker.set_metadata("llm_provider", llm_provider)
        
        # Save results if requested
        if kwargs.get("save_results", False):
            tracker.save_results()
        
        return {
            "framework": "semantic_kernel",
            "use_case": self.name,
            "final_output": final_article,
            "execution_time": execution_time,
            "token_usage": tracker.token_usage,
            "cost": tracker.cost,
            "model": model,
            "topic": topic
        }
    
    def run_langchain(self, **kwargs) -> Dict[str, Any]:
        """
        Run the Article Writing use case using LangChain.
        
        Returns:
            Dictionary with results
        """
        try:
            # Import LangChain
            from langchain_core.messages import SystemMessage, HumanMessage
            from langchain_openai import ChatOpenAI
        except ImportError:
            return {"error": "LangChain not installed. Install with 'pip install langchain langchain-openai'"}
        
        # Initialize performance tracker
        tracker = PerformanceTracker("langchain", self.name)
        tracker.start()
        
        # Get inputs
        topic = self._get_topic()
        llm_provider = self._get_llm_provider()
        model = self._get_model()
        
        # Initialize the LLM
        if llm_provider == "openai":
            llm = ChatOpenAI(
                model_name=model,
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            # For other providers, we'd need to implement their connectors
            return {"error": f"LLM provider {llm_provider} not supported in LangChain implementation"}
        
        # Define system prompts for each role
        sup_system = SystemMessage(content="You are Supervisor: an editor coordinating an article. Plan tasks and later finalize the article.")
        res_system = SystemMessage(content="You are Researcher: a subject expert. Provide factual info when asked.")
        cod_system = SystemMessage(content="You are Coder: a programmer. Provide Python code when asked.")
        wri_system = SystemMessage(content="You are Writer: an author. Write articles using given info and code.")
        
        # Helper function to get completion and track metrics
        def get_completion(system_message, human_message_content):
            messages = [system_message, HumanMessage(content=human_message_content)]
            response = llm.invoke(messages)
            response_text = response.content
            
            # Track response
            tracker.add_response(response_text)
            
            # Update token usage if available
            if hasattr(response, "token_usage") and response.token_usage:
                tracker.update_token_usage(
                    prompt_tokens=response.token_usage.get("prompt_tokens", 0),
                    completion_tokens=response.token_usage.get("completion_tokens", 0),
                    total_tokens=response.token_usage.get("total_tokens", None)
                )
            else:
                # Estimate tokens if not available
                prompt_tokens = tracker.count_tokens(human_message_content) + tracker.count_tokens(system_message.content)
                completion_tokens = tracker.count_tokens(response_text)
                tracker.update_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
            
            return response_text
        
        # Step 1: Supervisor creates a plan
        sup_plan = get_completion(
            sup_system,
            f"Plan an approach to fulfill this request: Write an article on {topic}."
        )
        
        # Step 2: Researcher provides information
        research_info = get_completion(
            res_system,
            f"Topic: {topic}.\n{sup_plan}\nProvide key points about this topic."
        )
        
        # Step 3: Coder provides a code example
        code_example = get_completion(
            cod_system,
            f"Topic: {topic}.\nProvide a simple Python example. {research_info}"
        )
        
        # Step 4: Writer drafts the article
        draft_article = get_completion(
            wri_system,
            f"Write an article on the topic, incorporating the info and code.\nInfo: {research_info}\nCode: {code_example}"
        )
        
        # Step 5: Supervisor reviews the draft
        final_article = get_completion(
            sup_system,
            f"Review and refine this draft:\n{draft_article}"
        )
        
        # Mark the final response
        tracker.add_response(final_article, is_final=True)
        
        # Stop the tracker and calculate metrics
        execution_time = tracker.stop()
        tracker.calculate_cost(model)
        
        # Set additional metadata
        tracker.set_metadata("topic", topic)
        tracker.set_metadata("llm_provider", llm_provider)
        
        # Save results if requested
        if kwargs.get("save_results", False):
            tracker.save_results()
        
        return {
            "framework": "langchain",
            "use_case": self.name,
            "final_output": final_article,
            "execution_time": execution_time,
            "token_usage": tracker.token_usage,
            "cost": tracker.cost,
            "model": model,
            "topic": topic
        }
    
    def run_crewai(self, **kwargs) -> Dict[str, Any]:
        """
        Run the Article Writing use case using CrewAI.
        
        Returns:
            Dictionary with results
        """
        try:
            # Import CrewAI
            from crewai import Agent, Task, Crew, Process
        except ImportError:
            return {"error": "CrewAI not installed. Install with 'pip install crewai'"}
        
        # Initialize performance tracker
        tracker = PerformanceTracker("crewai", self.name)
        tracker.start()
        
        # Get inputs
        topic = self._get_topic()
        llm_provider = self._get_llm_provider()
        model = self._get_model()
        
        # Set up CrewAI with the appropriate LLM
        if llm_provider == "openai":
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_MODEL_NAME"] = model
        else:
            # For other providers, we'd need to implement their connectors
            return {"error": f"LLM provider {llm_provider} not supported in CrewAI implementation"}
        
        # Create agents for each role
        supervisor = Agent(
            role="Supervisor",
            backstory="An editor overseeing a writing project.",
            goal="Plan the article writing and later finalize the content.",
            allow_delegation=True
        )
        
        researcher = Agent(
            role="Researcher",
            backstory="Expert in the topic of web security.",
            goal="Provide factual information about the topic."
        )
        
        coder = Agent(
            role="Coder",
            backstory="Skilled Python programmer.",
            goal="Provide a simple, relevant Python code example."
        )
        
        writer = Agent(
            role="Writer",
            backstory="Technical content writer.",
            goal="Draft the article combining info and code."
        )
        
        # Define tasks for each stage of the process
        task_plan = Task(
            description=f"Supervisor: Plan and assign tasks for the article on {topic}.",
            expected_output="Plan created",
            agent=supervisor
        )
        
        task_research = Task(
            description=f"Researcher: Gather key facts about {topic}.",
            expected_output="Key points provided",
            agent=researcher
        )
        
        task_code = Task(
            description=f"Coder: Produce a Python code example for {topic}.",
            expected_output="Code example provided",
            agent=coder
        )
        
        task_draft = Task(
            description="Writer: Write a draft article using the info and code.",
            expected_output="Draft article written",
            agent=writer
        )
        
        task_review = Task(
            description="Supervisor: Review and refine the draft article.",
            expected_output="Final article ready",
            agent=supervisor
        )
        
        # Create the crew with agents and tasks in order
        crew = Crew(
            agents=[supervisor, researcher, coder, writer],
            tasks=[task_plan, task_research, task_code, task_draft, task_review],
            process=Process.sequential
        )
        
        # Helper function to estimate token usage from CrewAI responses
        def estimate_tokens(response):
            # This is a rough approximation since CrewAI doesn't expose token counts directly
            prompt_tokens = tracker.count_tokens(topic) * 5  # Rough estimate for all prompts
            completion_tokens = tracker.count_tokens(response)
            tracker.update_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
        
        # Run the crew
        result = crew.kickoff(inputs={"topic": topic})
        final_article = str(result)
        
        # Track the final response
        tracker.add_response(final_article, is_final=True)
        
        # Estimate token usage
        estimate_tokens(final_article)
        
        # Stop the tracker and calculate metrics
        execution_time = tracker.stop()
        tracker.calculate_cost(model)
        
        # Set additional metadata
        tracker.set_metadata("topic", topic)
        tracker.set_metadata("llm_provider", llm_provider)
        
        # Save results if requested
        if kwargs.get("save_results", False):
            tracker.save_results()
        
        return {
            "framework": "crewai",
            "use_case": self.name,
            "final_output": final_article,
            "execution_time": execution_time,
            "token_usage": tracker.token_usage,
            "cost": tracker.cost,
            "model": model,
            "topic": topic
        } 