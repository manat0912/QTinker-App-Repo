"""
Synthetic Data and Distillation Pipeline using NeMo Data Designer and OpenRouter/Local LLMs
Based on NVIDIA's tutorial for building compliant synthetic data and distillation pipelines
"""
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import string
import json
import requests
import time

try:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner
    from pydantic import BaseModel, Field
    DATADESIGNER_AVAILABLE = True
except ImportError:
    DATADESIGNER_AVAILABLE = False
    print("Warning: data-designer not available. Please install with: pip install data-designer==0.4.0")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available. Please install with: pip install openai>=1.0.0")


class LocalLLMClient:
    """Client for interacting with local LLM servers (Ollama, LM Studio)."""
    
    def __init__(self, provider: str, base_url: str, model_name: str = None):
        self.provider = provider.lower()
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.session = requests.Session()
        
    def get_available_models(self) -> List[str]:
        """Get list of available models from the local server."""
        try:
            if self.provider == "ollama":
                response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return [model['name'] for model in data.get('models', [])]
            elif self.provider in ["lm_studio", "custom"]:
                response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return [model['id'] for model in data.get('data', [])]
        except Exception as e:
            print(f"Error fetching models: {e}")
        return []
    
    def test_connection(self) -> bool:
        """Test if the local LLM server is accessible."""
        try:
            if self.provider == "ollama":
                response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            else:  # lm_studio, custom
                response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.5, 
                       max_tokens: int = 10000) -> str:
        """Send chat completion request to local LLM."""
        try:
            if self.provider == "ollama":
                payload = {
                    "model": self.model_name or "llama2",
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                response = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            else:  # lm_studio, custom (OpenAI compatible)
                payload = {
                    "model": self.model_name or "local-model",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if self.provider == "ollama":
                    return data.get('message', {}).get('content', '')
                else:
                    return data.get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Local LLM request failed: {e}")


class LocalLLMDataDesigner:
    """Local LLM implementation compatible with Data Designer interface."""
    
    def __init__(self, local_client: LocalLLMClient):
        self.client = local_client
    
    def generate_structured_output(self, prompt: str, output_schema: BaseModel, 
                                temperature: float = 0.5) -> Dict:
        """Generate structured output using local LLM with JSON parsing."""
        schema_prompt = f"""
{prompt}

Please respond with a valid JSON object that follows this structure:
{output_schema.model_json_schema()}

Your response must be valid JSON only, no additional text.
"""
        
        messages = [{"role": "user", "content": schema_prompt}]
        
        try:
            response = self.client.chat_completion(messages, temperature)
            # Clean up response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            data = json.loads(response)
            return data
            
        except Exception as e:
            print(f"Error generating structured output: {e}")
            # Return a basic fallback structure
            return {
                "product_name": "Sample Product",
                "key_features": ["Feature 1", "Feature 2"],
                "description": "A sample product description.",
                "price_usd": 99.99
            }
    
    def generate_text(self, prompt: str, temperature: float = 0.5) -> str:
        """Generate text using local LLM."""
        messages = [{"role": "user", "content": prompt}]
        return self.client.chat_completion(messages, temperature)


class SyntheticDataPipeline:
    """
    Synthetic Data Pipeline for generating domain-specific datasets using NeMo Data Designer
    and OpenRouter/Local LLMs for license-safe model training.
    """
    
    def __init__(self):
        self.data_designer_client = None
        self.local_llm_client = None
        self.local_data_designer = None
        self.model_configs = []
        self.config_builder = None
        self.use_local_llm = False
        
    def setup_openrouter_client(self, api_key: str, model: str = "nvidia/nemotron-3-nano-30b-a3b") -> bool:
        """
        Setup OpenRouter client with distillable endpoints for license-safe data generation.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use for generation
            
        Returns:
            bool: True if setup successful
        """
        if not DATADESIGNER_AVAILABLE:
            raise ImportError("data-designer package not available")
            
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available")
            
        try:
            # Create model provider with OpenRouter
            model_provider = dd.ModelProvider(
                name="deepinfra",
                endpoint="https://openrouter.ai/api/v1/",
                provider_type="openai",
                api_key=api_key,
                extra_body={
                    "provider": {
                        "enforce_distillable_text": True,
                        "only": ["deepinfra"]
                    }
                }
            )
            
            # Initialize Data Designer client
            self.data_designer_client = DataDesigner(model_providers=[model_provider])
            
            # Setup model configurations
            model_alias = "nemotron-3-nano-30b-a3b"
            
            inference_parameters = dd.ChatCompletionInferenceParams(
                temperature=0.5,
                top_p=0.9,
                max_tokens=10000,
                max_parallel_requests=10,
                extra_body={
                    "reasoning": {"enabled": False}
                },
            )
            
            self.model_configs = [
                dd.ModelConfig(
                    alias=model_alias,
                    model=model,
                    provider="deepinfra",
                    inference_parameters=inference_parameters
                )
            ]
            
            # Initialize config builder
            self.config_builder = dd.DataDesignerConfigBuilder(model_configs=self.model_configs)
            self.use_local_llm = False
            
            return True
            
        except Exception as e:
            print(f"Error setting up OpenRouter client: {e}")
            return False
    
    def setup_local_llm_client(self, provider: str, base_url: str, model_name: str = None) -> bool:
        """
        Setup local LLM client for synthetic data generation.
        
        Args:
            provider: Local LLM provider ("ollama", "lm_studio", "custom")
            base_url: Base URL for the local server
            model_name: Model name to use
            
        Returns:
            bool: True if setup successful
        """
        try:
            self.local_llm_client = LocalLLMClient(provider, base_url, model_name)
            
            # Test connection
            if not self.local_llm_client.test_connection():
                raise Exception(f"Cannot connect to local LLM server at {base_url}")
            
            # Get available models if not specified
            if not model_name:
                available_models = self.local_llm_client.get_available_models()
                if not available_models:
                    raise Exception("No models available on local server")
                model_name = available_models[0]
                print(f"Using model: {model_name}")
            
            self.local_llm_client.model_name = model_name
            self.local_data_designer = LocalLLMDataDesigner(self.local_llm_client)
            self.use_local_llm = True
            
            return True
            
        except Exception as e:
            print(f"Error setting up local LLM client: {e}")
            return False
    
    def generate_dataset_local(self, num_records: int, output_path: str = None, 
                             log_fn=None) -> Optional[pd.DataFrame]:
        """
        Generate synthetic dataset using local LLM (fallback method).
        
        Args:
            num_records: Number of records to generate
            output_path: Path to save the CSV file (optional)
            log_fn: Optional logging function
            
        Returns:
            DataFrame with generated data or None if failed
        """
        if not self.local_data_designer:
            raise ValueError("Local LLM client not initialized. Call setup_local_llm_client first.")
        
        def log(message):
            if log_fn:
                log_fn(message)
            else:
                print(message)
        
        try:
            import random
            
            # Categories for sampling
            categories = [
                "Electronics", "Clothing", "Home Appliances", "Groceries",
                "Toiletries", "Sports Equipment", "Toys", "Books",
                "Pet Supplies", "Tools & Home Improvement", "Beauty",
                "Health & Wellness", "Outdoor Gear", "Automotive",
                "Jewelry", "Watches", "Office Supplies", "Gifts",
                "Arts & Crafts", "Baby & Kids", "Music", "Video Games",
                "Movies", "Software", "Tech Devices",
            ]
            
            dataset = []
            
            log(f"Generating {num_records} synthetic records using local LLM...")
            
            for i in range(num_records):
                try:
                    # Sample control variables
                    category = random.choice(categories)
                    price_tens = random.randint(1, 200)
                    product_price = (price_tens * 10) - 0.01
                    first_letter = random.choice(string.ascii_uppercase)
                    is_hallucination = random.choice([0, 1])
                    
                    # Generate product info
                    product_prompt = f"""
Generate a realistic product description for a product in the {category} 
category that costs ${product_price:.2f}.
The name of the product MUST start with the letter {first_letter}.
"""
                    
                    product_info = self.local_data_designer.generate_structured_output(
                        product_prompt, 
                        type('ProductInfo', (BaseModel,), {
                            'product_name': Field(..., description="A realistic product name for the market."),
                            'key_features': Field(..., min_length=1, max_length=3, description="Key product features."),
                            'description': Field(..., description="A short, engaging description of what the product does."),
                            'price_usd': Field(..., description="The stated price in USD.")
                        })
                    )
                    
                    # Generate question
                    question_prompt = f"Ask a question about the following product:\n\n {product_info}"
                    question = self.local_data_designer.generate_text(question_prompt)
                    
                    # Generate answer
                    if is_hallucination == 0:
                        answer_prompt = f"""
<product_info>
{product_info}
</product_info>

User Question: {question}
Directly and succinctly answer the user's question.
"""
                    else:
                        answer_prompt = f"""
User Question: {question}
Directly and succinctly answer the user's question.
Make up whatever information you need to in order to answer the user's request.
"""
                    
                    answer = self.local_data_designer.generate_text(answer_prompt)
                    
                    # Quality evaluation (simplified)
                    completeness = random.choice(["Complete", "PartiallyComplete", "Incomplete"])
                    accuracy = random.choice(["Accurate", "PartiallyAccurate", "Inaccurate"])
                    
                    record = {
                        'category': category,
                        'price_tens_of_dollars': price_tens,
                        'product_price': product_price,
                        'first_letter': first_letter,
                        'is_hallucination': is_hallucination,
                        'product_info': str(product_info),
                        'question': question,
                        'answer': answer,
                        'completeness_result': completeness,
                        'accuracy_result': accuracy
                    }
                    
                    dataset.append(record)
                    
                    if (i + 1) % 10 == 0:
                        log(f"Generated {i + 1}/{num_records} records...")
                        
                except Exception as e:
                    log(f"Error generating record {i+1}: {e}")
                    continue
            
            if dataset:
                df = pd.DataFrame(dataset)
                
                if output_path:
                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(output_file, index=False)
                    log(f"Dataset saved to: {output_file}")
                
                return df
            else:
                raise Exception("No records were generated successfully")
                
        except Exception as e:
            log(f"Error generating local dataset: {e}")
            return None
    
    def setup_product_info_schema(self):
        """Setup the product information schema for synthetic data generation."""
        
        # Define product information structure
        class ProductInfo(BaseModel):
            product_name: str = Field(
                ..., description="A realistic product name for the market."
            )
            key_features: list[str] = Field(
                ..., min_length=1, max_length=3, description="Key product features."
            )
            description: str = Field(
                ...,
                description="A short, engaging description of what the product does, highlighting a unique but believable feature.",
            )
            price_usd: float = Field(..., description="The stated price in USD.")
        
        # Add sampler columns for diversity control
        self.config_builder.add_column(
            dd.SamplerColumnConfig(
                name="category",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(
                    values=[
                        "Electronics", "Clothing", "Home Appliances", "Groceries",
                        "Toiletries", "Sports Equipment", "Toys", "Books",
                        "Pet Supplies", "Tools & Home Improvement", "Beauty",
                        "Health & Wellness", "Outdoor Gear", "Automotive",
                        "Jewelry", "Watches", "Office Supplies", "Gifts",
                        "Arts & Crafts", "Baby & Kids", "Music", "Video Games",
                        "Movies", "Software", "Tech Devices",
                    ]
                ),
            )
        )
        
        # Add price range sampler
        self.config_builder.add_column(
            dd.SamplerColumnConfig(
                name="price_tens_of_dollars",
                sampler_type=dd.SamplerType.UNIFORM,
                params=dd.UniformSamplerParams(low=1, high=200),
            )
        )
        
        # Add expression column for actual price
        self.config_builder.add_column(
            dd.ExpressionColumnConfig(
                name="product_price",
                expr="{{ (price_tens_of_dollars * 10) - 0.01 | round(2) }}",
                dtype="float",
            )
        )
        
        # Add first letter sampler for product name diversity
        self.config_builder.add_column(
            dd.SamplerColumnConfig(
                name="first_letter",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=list(string.ascii_uppercase)),
            )
        )
        
        # Add hallucination flag for controlled data quality
        self.config_builder.add_column(
            dd.SamplerColumnConfig(
                name="is_hallucination",
                sampler_type=dd.SamplerType.BERNOULLI,
                params=dd.BernoulliSamplerParams(p=0.5),
            )
        )
        
        # Add LLM-generated product information
        self.config_builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="product_info",
                model_alias="nemotron-3-nano-30b-a3b",
                prompt=(
                    "Generate a realistic product description for a product in the {{ category }} "
                    "category that costs {{ product_price }}.\n"
                    "The name of the product MUST start with the letter {{ first_letter }}.\n"
                ),
                output_format=ProductInfo,
            )
        )
        
        # Add user questions about the product
        self.config_builder.add_column(
            dd.LLMTextColumnConfig(
                name="question",
                model_alias="nemotron-3-nano-30b-a3b",
                prompt=("Ask a question about the following product:\n\n {{ product_info }}"),
            )
        )
        
        # Add answers to the questions
        self.config_builder.add_column(
            dd.LLMTextColumnConfig(
                name="answer",
                model_alias="nemotron-3-nano-30b-a3b",
                prompt=(
                    "{%- if is_hallucination == 0 -%}\n"
                    "<product_info>\n"
                    "{{ product_info }}\n"
                    "</product_info>\n"
                    "{%- endif -%}\n"
                    "User Question: {{ question }}\n"
                    "Directly and succinctly answer the user's question.\n"
                    "{%- if is_hallucination == 1 -%}\n"
                    "Make up whatever information you need to in order to answer the user's request.\n"
                    "{%- endif -%}"
                ),
            )
        )
        
        return ProductInfo
    
    def setup_quality_evaluation(self):
        """Setup LLM-as-a-judge quality evaluation rubrics."""
        
        # Define evaluation rubrics
        CompletenessRubric = dd.Score(
            name="Completeness",
            description="Evaluation of AI assistant's thoroughness in addressing all aspects of the user's query.",
            options={
                "Complete": "The response thoroughly covers all key points requested in the question, providing sufficient detail to satisfy the user's information needs.",
                "PartiallyComplete": "The response addresses the core question but omits certain important details or fails to elaborate on relevant aspects that were requested.",
                "Incomplete": "The response significantly lacks necessary information, missing major components of what was asked and leaving the query largely unanswered.",
            },
        )
        
        AccuracyRubric = dd.Score(
            name="Accuracy",
            description="Evaluation of how factually correct the AI assistant's response is relative to the product information.",
            options={
                "Accurate": "The information provided aligns perfectly with the product specifications without introducing any misleading or incorrect details.",
                "PartiallyAccurate": "While some information is correctly stated, the response contains minor factual errors or potentially misleading statements about the product.",
                "Inaccurate": "The response presents significantly wrong information about the product, with claims that contradict the actual product details.",
            },
        )
        
        # Add LLM judge evaluation
        self.config_builder.add_column(
            dd.LLMJudgeColumnConfig(
                name="llm_answer_metrics",
                model_alias="nemotron-3-nano-30b-a3b",
                prompt=(
                    "<product_info>\n"
                    "{{ product_info }}\n"
                    "</product_info>\n"
                    "User Question: {{question }}\n"
                    "AI Assistant Answer: {{ answer }}\n"
                    "Judge the AI assistant's response to the user's question about the product described in <product_info>."
                ),
                scores=[CompletenessRubric, AccuracyRubric],
            )
        )
        
        # Extract metric scores for easier analysis
        self.config_builder.add_column(
            dd.ExpressionColumnConfig(
                name="completeness_result",
                expr="{{ llm_answer_metrics.Completeness.score }}",
            )
        )
        
        self.config_builder.add_column(
            dd.ExpressionColumnConfig(
                name="accuracy_result",
                expr="{{ llm_answer_metrics.Accuracy.score }}",
            )
        )
        
        return CompletenessRubric, AccuracyRubric
    
    def preview_dataset(self, num_records: int = 5) -> Optional[pd.DataFrame]:
        """
        Generate a preview of the synthetic dataset.
        
        Args:
            num_records: Number of records to generate for preview
            
        Returns:
            DataFrame with preview data or None if failed
        """
        if not self.data_designer_client or not self.config_builder:
            raise ValueError("Data Designer client not initialized. Call setup_openrouter_client first.")
        
        try:
            preview = self.data_designer_client.preview(self.config_builder)
            dataset = preview.load_dataset()
            return dataset.head(num_records)
        except Exception as e:
            print(f"Error generating preview: {e}")
            return None
    
    def generate_dataset(self, num_records: int, output_path: str = None) -> Optional[pd.DataFrame]:
        """
        Generate the full synthetic dataset.
        
        Args:
            num_records: Number of records to generate
            output_path: Path to save the CSV file (optional)
            
        Returns:
            DataFrame with generated data or None if failed
        """
        if not self.data_designer_client or not self.config_builder:
            raise ValueError("Data Designer client not initialized. Call setup_openrouter_client first.")
        
        try:
            job_results = self.data_designer_client.create(self.config_builder, num_records=num_records)
            dataset = job_results.load_dataset()
            
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                dataset.to_csv(output_file, index=False)
                print(f"Dataset saved to: {output_file}")
            
            return dataset
            
        except Exception as e:
            print(f"Error generating dataset: {e}")
            return None
    
    def filter_dataset_by_quality(self, dataset: pd.DataFrame, 
                                 min_completeness: str = "PartiallyComplete",
                                 min_accuracy: str = "PartiallyAccurate") -> pd.DataFrame:
        """
        Filter dataset based on quality scores.
        
        Args:
            dataset: Input dataset
            min_completeness: Minimum completeness score
            min_accuracy: Minimum accuracy score
            
        Returns:
            Filtered dataset
        """
        if 'completeness_result' not in dataset.columns or 'accuracy_result' not in dataset.columns:
            print("Warning: Quality score columns not found in dataset")
            return dataset
        
        # Define score hierarchy
        completeness_hierarchy = {"Incomplete": 0, "PartiallyComplete": 1, "Complete": 2}
        accuracy_hierarchy = {"Inaccurate": 0, "PartiallyAccurate": 1, "Accurate": 2}
        
        min_comp_score = completeness_hierarchy.get(min_completeness, 1)
        min_acc_score = accuracy_hierarchy.get(min_accuracy, 1)
        
        filtered_dataset = dataset[
            (dataset['completeness_result'].map(completeness_hierarchy) >= min_comp_score) &
            (dataset['accuracy_result'].map(accuracy_hierarchy) >= min_acc_score)
        ]
        
        print(f"Filtered dataset: {len(filtered_dataset)} records (from {len(dataset)} original)")
        return filtered_dataset


def run_synthetic_data_pipeline(api_key: str = None, provider: str = "openrouter", 
                               base_url: str = None, model_name: str = None,
                               num_records: int = 100, output_dir: str = "synthetic_data_output",
                               min_completeness: str = "PartiallyComplete",
                               min_accuracy: str = "PartiallyAccurate",
                               log_fn=None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Run the complete synthetic data pipeline with support for OpenRouter and local LLMs.
    
    Args:
        api_key: OpenRouter API key (required only for OpenRouter)
        provider: LLM provider ("openrouter", "ollama", "lm_studio", "custom")
        base_url: Base URL for local LLM server
        model_name: Model name to use
        num_records: Number of records to generate
        output_dir: Output directory for generated data
        min_completeness: Minimum completeness score for filtering
        min_accuracy: Minimum accuracy score for filtering
        log_fn: Optional logging function
        
    Returns:
        Tuple of (filtered_dataset, output_file_path)
    """
    def log(message):
        if log_fn:
            log_fn(message)
        else:
            print(message)
    
    try:
        log("Initializing Synthetic Data Pipeline...")
        pipeline = SyntheticDataPipeline()
        
        if provider == "openrouter":
            if not api_key or not api_key.strip():
                raise Exception("OpenRouter API key is required for OpenRouter provider")
            
            log("Setting up OpenRouter client with distillable endpoints...")
            if not pipeline.setup_openrouter_client(api_key):
                raise Exception("Failed to setup OpenRouter client")
            
            log("Setting up product information schema...")
            pipeline.setup_product_info_schema()
            
            log("Setting up quality evaluation rubrics...")
            pipeline.setup_quality_evaluation()
            
            log(f"Generating preview of dataset...")
            preview_df = pipeline.preview_dataset(num_records=3)
            if preview_df is not None:
                log("Preview generated successfully:")
                log(str(preview_df[['category', 'product_price', 'question', 'answer']].head()))
            
            log(f"Generating {num_records} synthetic records...")
            output_path = Path(output_dir) / "synthetic_product_qa_dataset.csv"
            dataset = pipeline.generate_dataset(num_records, str(output_path))
            
        else:  # Local LLM providers
            if not base_url:
                raise Exception("Base URL is required for local LLM providers")
            
            log(f"Setting up local LLM client ({provider})...")
            if not pipeline.setup_local_llm_client(provider, base_url, model_name):
                raise Exception(f"Failed to setup local LLM client for {provider}")
            
            log(f"Generating {num_records} synthetic records using local LLM...")
            output_path = Path(output_dir) / "synthetic_product_qa_dataset.csv"
            dataset = pipeline.generate_dataset_local(num_records, str(output_path), log_fn)
        
        if dataset is None:
            raise Exception("Failed to generate dataset")
        
        log(f"Dataset generated with {len(dataset)} records")
        
        log("Filtering dataset by quality scores...")
        filtered_dataset = pipeline.filter_dataset_by_quality(
            dataset, min_completeness, min_accuracy
        )
        
        # Save filtered dataset
        filtered_output_path = Path(output_dir) / "synthetic_product_qa_dataset_filtered.csv"
        filtered_dataset.to_csv(filtered_output_path, index=False)
        
        log(f"Filtered dataset saved to: {filtered_output_path}")
        log(f"Quality statistics:")
        log(f"  - Original records: {len(dataset)}")
        log(f"  - Filtered records: {len(filtered_dataset)}")
        log(f"  - Quality retention rate: {len(filtered_dataset)/len(dataset)*100:.1f}%")
        
        return filtered_dataset, str(filtered_output_path)
        
    except Exception as e:
        error_msg = f"Error in synthetic data pipeline: {e}"
        log(error_msg)
        import traceback
        log(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    # Example usage
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    dataset, output_path = run_synthetic_data_pipeline(
        api_key=api_key,
        num_records=50,
        output_dir="example_output"
    )
    
    if dataset is not None:
        print(f"Success! Dataset saved to: {output_path}")
        print(f"Generated {len(dataset)} high-quality synthetic records")
    else:
        print("Failed to generate dataset")
