# Hallucination Detection Environment Implementation
import re
import json
import logging
import asyncio
from typing import Any, Optional, TypedDict, Tuple, Dict, List
import numpy as np
import ray
import torch
from dataclasses import dataclass

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

# Import the Google Search Client
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ========================= STAGE 1: FACTUAL CLAIM EXTRACTION =========================

FACTUAL_CLAIM_EXTRACTION_PROMPT = """You are a meticulous fact-checker. Extract ONLY specific, verifiable factual claims from the given response.

Focus on these ERROR TYPES:
1. **Dates and Years**: Specific dates, years, or time periods (e.g., "1969", "January 2020")
2. **Numbers and Quantities**: Numerical values, counts, measurements (e.g., "206 bones", "3.14", "100 million")
3. **Names and Proper Nouns**: People's names, organization names, product names (e.g., "Guido van Rossum", "NASA", "Python 3.0")
4. **Places and Locations**: Countries, cities, geographical locations (e.g., "Paris", "Mount Everest")
5. **Scientific Facts**: Physical constants, chemical formulas, biological facts (e.g., "speed of light", "H2O", "DNA structure")
6. **Historical Events**: Specific historical occurrences with verifiable details (e.g., "first moon landing", "World War II ended")

**Context:**
{context}

**Response to Check:**
{response}

Instructions:
1. Extract ONLY claims that contain the error types listed above
2. Focus on ATOMIC facts - one specific verifiable detail per claim
3. Each claim should be narrow and specific enough to verify with a single search
4. Format each claim as a clear, standalone statement

Respond in JSON format:
{{
    "factual_claims": [
        {{
            "claim": "The exact atomic factual statement from the text",
            "claim_type": "date|number|name|place|scientific_fact|historical_event",
            "key_element": "The specific element to verify (e.g., '1968', 'Linus Torvalds', '208')",
            "search_terms": "specific search terms for verification"
        }}
    ]
}}"""

# ========================= STAGE 2: ERROR DETECTION =========================

ERROR_DETECTION_PROMPT = """Based on the search results as below, you should determine if factual claims are correct or incorrect.

**Search Results:**
{search_results}

Instructions:
1. For each claim you extracted previously, verify the KEY ELEMENT (the specific date, number, name, place, etc.)
2. Compare the key elements in the claim with information from search results
3. Mark as error ONLY if any key element is definitively wrong according to search results
4. If search results don't provide clear information, mark as "uncertain" rather than error

Respond in JSON format:
{{
    "error_detection": [
        {{
            "claim": "The original claim being analyzed",
            "key_element": "The specific element that was checked",
            "is_error": true/false,
            "is_uncertain": true/false,
            "correct_info": "The correct value if this is an error",
            "confidence": "high|medium|low",
            "explanation": "Brief explanation based on search results"
        }}
    ]
}}"""

# ========================= DATA STRUCTURES =========================

class HallucinationMetadata(TypedDict):
    context: str
    response: str
    ground_truth_errors: Optional[List[Dict[str, str]]]  # For training
    claim_extraction_complete: Optional[bool]
    extracted_claims: Optional[List[Dict[str, str]]]
    search_results: Optional[List[Dict[str, str]]]

# ========================= PROMPT FORMATTING =========================

def format_claim_extraction_prompt(context: str, response: str) -> str:
    """Format the prompt for extracting factual claims."""
    return FACTUAL_CLAIM_EXTRACTION_PROMPT.format(
        context=context,
        response=response
    )

def format_error_detection_prompt(claims_with_results: List[Dict[str, Any]]) -> str:
    """Format the prompt for error detection using search results."""
    # Format the claims and search results for analysis
    formatted_data = json.dumps(claims_with_results, indent=2)
    return ERROR_DETECTION_PROMPT.format(search_results=formatted_data)

# ========================= PARSING UTILITIES =========================

def parse_claim_extraction_response(response: str) -> Tuple[bool, List[Dict[str, str]], str]:
    """
    Parse claim extraction response.
    Returns (is_valid, extracted_claims, error_msg)
    """
    if '</think>' in response:
        response = response.split('</think>')[-1]
    response = response.strip()

    try:
        # Try to parse JSON response
        result = json.loads(response)
        claims = result.get("factual_claims", [])
        
        # Validate that each claim has required fields
        validated_claims = []
        for claim in claims:
            if all(key in claim for key in ["claim", "claim_type", "key_element", "search_terms"]):
                validated_claims.append(claim)
        
        return True, validated_claims, ""
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                claims = result.get("factual_claims", [])
                return True, claims, ""
            except:
                pass
        return False, [], f"Failed to parse JSON: {e}"
    except Exception as e:
        return False, [], f"Error parsing response: {e}"

def parse_error_detection_response(response: str) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Parse error detection response.
    Returns (is_valid, detected_errors, error_msg)
    """
    if '</think>' in response:
        response = response.split('</think>')[-1]
    response = response.strip()

    try:
        result = json.loads(response)
        detections = result.get("error_detection", [])
        
        # Extract only the errors (not uncertain or correct claims)
        errors = []
        for detection in detections:
            if detection.get("is_error", False):
                errors.append({
                    "claim": detection.get("claim", ""),
                    "key_element": detection.get("key_element", ""),
                    "correct_info": detection.get("correct_info", ""),
                    "explanation": detection.get("explanation", "")
                })
        
        return True, errors, ""
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                detections = result.get("error_detection", [])
                errors = [d for d in detections if d.get("is_error", False)]
                return True, errors, ""
            except:
                pass
        return False, [], f"Failed to parse JSON: {e}"
    except Exception as e:
        return False, [], f"Error parsing response: {e}"

# ========================= GOOGLE SEARCH INTEGRATION =========================

class GoogleSearchClient:
    """Robust Google Search client using REST API directly."""
    
    def __init__(self, api_key: str, cse_id: str, fetch_full_content: bool = False):
        self.api_key = api_key
        self.cse_id = cse_id
        self.fetch_full_content = fetch_full_content
        self.session = self._create_session()
        self.cache = {}
        print("✓ Google Search Client initialized (REST API)")
    
    def _create_session(self) -> requests.Session:
        """Create a robust session with retry strategy."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504, 429],
            respect_retry_after_header=True 
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; SearchBot/1.0)'
        })
        return session
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search using Google Custom Search REST API directly."""

        # Input validation
        if not query or not isinstance(query, str):
            return []
        
        # Clean query
        clean_query = query.strip()
        if not clean_query:
            return []
        
        cache_key = f"{query}_{num_results}"
        if cache_key in self.cache:
            print(f"Cache hit for: '{query}'")
            return self.cache[cache_key]
            
        print(f"Searching for: '{clean_query}'")
        
        try:
            # Use REST API directly instead of googleapiclient to avoid memory issues
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': clean_query,
                'num': min(max(num_results, 1), 10)
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            print(f"✓ API call successful, processing {len(items)} items")
            results = self._process_results_safe(items)

            if results:
                self.cache[cache_key] = results
            
            # Process results safely
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Search API error: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected search error: {e}")
            return []
    
    def _process_results_safe(self, items: list) -> List[Dict[str, str]]:
        """Process search results with safety checks."""
        results = []
        
        for i, item in enumerate(items[:10]):  # Hard limit to 10 results
            try:
                # Skip non-HTML files
                if 'fileFormat' in item or 'mime' in item:
                    continue
                
                link = item.get('link', '')
                if not link:
                    continue
                
                # Skip problematic file types
                if any(link.lower().endswith(ext) for ext in 
                       ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.zip']):
                    continue
                
                # Extract and limit text lengths
                title = str(item.get('title', ''))
                snippet = str(item.get('snippet', ''))[:5000]
                
                # Optional: Fetch extended snippet (disabled by default for safety)
                if self.fetch_full_content and len(results) < 3:
                    print(f"  Fetching extended content for result {i+1}")
                    extended = self._get_extended_snippet_safe(link, snippet)
                    if extended and len(extended) > len(snippet):
                        snippet = extended[:5000]
                        print(f"  ✓ Extended snippet obtained")
                
                results.append({
                    'title': title,
                    'snippet': snippet,
                    'link': link
                })
                
                print(f"  ✓ Processed result {i+1}: {title[:50]}...")
                
            except Exception as e:
                print(f"Error processing result {i}: {e}")
                continue
        
        return results
    
    def _get_extended_snippet_safe(self, url: str, original_snippet: str) -> Optional[str]:
        """Safely fetch extended content from URL."""
        try:
            # Set strict limits
            max_size = 100 * 1024  # 100KB max
            timeout = 3
            
            response = self.session.get(
                url,
                timeout=timeout,
                stream=True,
                verify=False,  # Skip SSL verification for problematic sites
                allow_redirects=True
            )
            
            # Check status and content type
            if response.status_code != 200:
                return original_snippet
            
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return original_snippet
            
            # Read content with size limit
            content = b''
            for chunk in response.iter_content(chunk_size=4096):
                content += chunk
                if len(content) > max_size:
                    break
            
            # Simple text extraction without BeautifulSoup to avoid memory issues
            text = content.decode('utf-8', errors='ignore')
            
            # Remove HTML tags with regex (safer than BeautifulSoup for memory)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Return extended snippet if longer than original
            if len(text) > len(original_snippet):
                return text[:5000]
            
            return original_snippet
            
        except Exception as e:
            print(f"    Extended fetch error: {e}")
            return original_snippet


# ========================= F1 SCORE CALCULATION =========================

def calculate_f1_score(predicted_errors: List[Dict], ground_truth_errors: List[Dict]) -> float:
    """Calculate F1 score for error detection."""
    if not ground_truth_errors and not predicted_errors:
        return 1.0  # Perfect score if both are empty
    
    if not ground_truth_errors or not predicted_errors:
        return 0.0  # Zero score if one is empty but not the other
    
    # Extract key elements for comparison
    pred_keys = set()
    for err in predicted_errors:
        key = err.get("key_element", "").lower().strip()
        if key:
            pred_keys.add(key)
    
    gt_keys = set()
    for err in ground_truth_errors:
        key = err.lower().strip()
        gt_keys.add(key)
    
    if not gt_keys:
        return 0.0
    
    # Calculate precision, recall, and F1
    true_positives = len(pred_keys & gt_keys)
    false_positives = len(pred_keys - gt_keys)
    false_negatives = len(gt_keys - pred_keys)
    
    if true_positives == 0:
        return 0.0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# ========================= HALLUCINATION DETECTION ENVIRONMENT =========================

@ray.remote
class HallucinationEnvironment(EnvironmentInterface):
    """Two-stage hallucination detection environment: Stage 1 claim extraction, Stage 2 error detection."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.format_penalty = cfg.get("format_penalty", -1.0)
        
        # Initialize Google Search Client if API keys are provided
        self.google_client = None
        google_api_key = cfg.get("google_api_key")
        google_cse_id = cfg.get("google_cse_id")
        
        if google_api_key and google_cse_id:
            try:
                # Import the actual GoogleSearchClient
                self.google_client = GoogleSearchClient(
                    api_key=google_api_key, 
                    cse_id=google_cse_id, 
                    fetch_full_content=cfg.get("fetch_full_content", False)
                )
                logging.info("Google Search Client initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize Google Search Client: {e}")
                self.google_client = None
        
        logging.basicConfig(level=logging.INFO)
    
    def _perform_google_search(self, claims: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Perform Google search for extracted claims."""
        if not self.google_client:
            # Return mock results for testing
            return [
                {
                    #"claim": claim["claim"],
                    #"key_element": claim["key_element"],
                    "search_terms": claim["search_terms"],
                    "search_results": "No search results available (Google API not configured)"
                }
                for claim in claims
            ]
        
        results = []
        for claim in claims[:5]:  # Limit to 5 claims to avoid too many API calls
            search_terms = claim.get("search_terms", "")
            if search_terms:
                try:
                    search_result = self.google_client.search(search_terms, num_results=2)
                    formatted_results = self._format_google_results(search_result, search_terms)
                    results.append({
                        #"claim": claim["claim"],
                        #"claim_type": claim.get("claim_type", ""),
                        #"key_element": claim["key_element"],
                        "search_terms": search_terms,
                        "search_results": formatted_results
                    })
                except Exception as e:
                    logging.error(f"Search failed for '{search_terms}': {e}")
                    results.append({
                        #"claim": claim["claim"],
                        #"key_element": claim["key_element"],
                        "search_terms": search_terms,
                        "search_results": f"Search error: {e}"
                    })
        
        return results
    
    def _format_google_results(self, results: List[Dict[str, str]], query: str) -> str:
        """Format Google search results for the model."""
        if not results:
            return f"No results found for: {query}"
        
        formatted = [f"Search results for: {query}\n"]
        for i, result in enumerate(results):
            title = result.get('title', 'No title').strip()[:100]
            snippet = result.get('snippet', 'No snippet').strip()[:500]
            link = result.get('link', 'No link')
            formatted.append(f"Result {i+1}: {title}\n{snippet}\nSource: {link}\n")
        
        return '\n'.join(formatted)
    
    def step(self, message_log_batch: List[List[Dict[str, str]]], metadata: List[HallucinationMetadata]) -> EnvironmentReturn:
        """Process two-stage hallucination detection."""
        
        print(f"\n[HALLUCINATION ENV] Processing batch of {len(message_log_batch)} samples")
        
        rewards = []
        observations = []
        next_metadata = []
        answers = []
        
        for i, (conversation, meta) in enumerate(zip(message_log_batch, metadata)):
            # Extract assistant's response
            assistant_response = ""
            for msg in reversed(conversation):
                if msg["role"] == "assistant":
                    assistant_response = msg["content"]
                    break
            
            # Check which stage we're in
            if not meta.get("claim_extraction_complete"):
                # STAGE 1: Claim extraction and search
                reward, obs, updated_meta, answer = self._process_claim_extraction_stage(
                    assistant_response, meta
                )
            else:
                # STAGE 2: Error detection
                reward, obs, updated_meta, answer = self._process_error_detection_stage(
                    assistant_response, meta
                )
            
            rewards.append(reward)
            observations.append(obs)
            next_metadata.append(updated_meta)
            answers.append(answer)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminateds = torch.tensor([
            meta is None for meta in next_metadata
        ], dtype=torch.bool)
        
        # Add stop strings for each stage
        next_stop_strings = []
        for meta in next_metadata:
            if meta and not meta.get("claim_extraction_complete"):
                next_stop_strings.append(None)  # No specific stop string for claim extraction
            else:
                next_stop_strings.append(None)  # No specific stop string for error detection
        
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds,
            answers=answers,
        )
    
    def _process_claim_extraction_stage(self, response: str, metadata: HallucinationMetadata) -> Tuple[float, dict, HallucinationMetadata]:
        """Process claim extraction stage and perform searches."""
        
        # Parse extracted claims
        is_parsed, extracted_claims, error_msg = parse_claim_extraction_response(response)
        
        if not is_parsed:
            print(f"[CLAIM EXTRACTION] Parse error: {error_msg}")
            obs = {
                "role": "environment",
                "content": f"<environment>Claim extraction format error: {error_msg}</environment>"
            }
            return float(self.format_penalty), obs, None, None
        
        # Perform Google searches for the extracted claims
        search_results = self._perform_google_search(extracted_claims)
        
        # Store results and prepare for error detection stage
        updated_metadata = metadata.copy()
        updated_metadata["claim_extraction_complete"] = True
        updated_metadata["extracted_claims"] = extracted_claims
        updated_metadata["search_results"] = search_results
        
        # Create the observation for the error detection stage
        error_detection_prompt = format_error_detection_prompt(search_results)
        
        # No reward for the first stage
        reward = 0.0
        
        # Return observation that becomes the next user message
        obs = {
            "role": "user",
            "content": "<|im_start|>user\n" + error_detection_prompt + "<|im_end|>\n<|im_start|>assistant\n"
        }
        
        answer = None
        
        return reward, obs, updated_metadata, answer
    
    def _process_error_detection_stage(self, response: str, metadata: HallucinationMetadata) -> Tuple[float, dict, HallucinationMetadata]:
        """Process error detection stage and calculate F1 score reward."""
        
        # Parse detected errors
        is_valid, detected_errors, error_msg = parse_error_detection_response(response)
        
        if not is_valid:
            print(f"[ERROR DETECTION] Parse error: {error_msg}")
            obs = {
                "role": "environment",
                "content": f"<environment>Error detection format error: {error_msg}</environment>"
            }
            return float(self.format_penalty), obs, None, None
        
        # Calculate F1 score if ground truth is available
        reward = 0.0
        if metadata.get("ground_truth_errors") is not None:
            ground_truth_errors = metadata["ground_truth_errors"]
            f1_score = calculate_f1_score(detected_errors, ground_truth_errors)
            # Scale F1 score to a reasonable reward range
            reward = f1_score * 10.0  # Scale to 0-10 range
        else:
            # If no ground truth, give a small positive reward for valid format
            reward = 0.1
        
        obs = {
            "role": "environment",
            "content": f"<environment>Hallucination detection completed. Detected {len(detected_errors)} errors. Reward: {reward:.2f}</environment>",
        }
        
        return float(reward), obs, None, detected_errors  # Terminate episode
    
    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Calculate metrics for hallucination detection."""
        rewards = batch.get("rewards", torch.tensor([]))
        num_samples = len(batch.get("idx", []))
        
        if len(rewards) == 0:
            return batch, {}
        
        rewards_np = rewards.numpy() if hasattr(rewards, 'numpy') else np.array(rewards)
        
        # Calculate metrics
        mean_reward = float(np.mean(rewards_np))
        format_violation_rate = float(np.mean(rewards_np == self.format_penalty))
        
        # For valid rewards, calculate performance metrics
        valid_rewards = rewards_np[rewards_np != self.format_penalty]
        if len(valid_rewards) > 0:
            mean_f1_score = float(np.mean(valid_rewards) / 10.0)  # Convert back to 0-1 range
            high_f1_rate = float(np.mean(valid_rewards > 5.0))  # F1 > 0.5
            perfect_f1_rate = float(np.mean(valid_rewards > 9.0))  # F1 > 0.9
        else:
            mean_f1_score = 0.0
            high_f1_rate = 0.0
            perfect_f1_rate = 0.0
        
        metrics = {
            "mean_reward": mean_reward,
            "format_violation_rate": format_violation_rate,
            "mean_f1_score": mean_f1_score,
            "high_f1_rate": high_f1_rate,
            "perfect_f1_rate": perfect_f1_rate,
            "num_samples": num_samples,
            "valid_samples": len(valid_rewards),
            "approach": "hallucination_detection",
        }
        
        return batch, metrics
    