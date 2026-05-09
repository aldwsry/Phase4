"""
Data loading and preprocessing utilities for evaluation benchmarks
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and manage evaluation datasets"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_mmlu(self, num_samples: Optional[int] = None, 
                  split: str = "test") -> Dict[str, List[str]]:
        """
        Load MMLU benchmark dataset
        
        MMLU: Massive Multitask Language Understanding
        - 14,042 multiple-choice questions
        - 57 subjects from STEM to humanities
        - 4 answer options (A, B, C, D)
        
        Args:
            num_samples: Number of samples to load (None = all)
            split: 'test', 'dev', 'val'
        
        Returns:
            Dictionary with questions, answers, and metadata
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("datasets library not installed. Install with: pip install datasets")
        
        logger.info("Loading MMLU benchmark...")
        
        try:
            # Load from Hugging Face
            dataset = load_dataset("cais/mmlu", "all")
            
            split_data = dataset[split] if split in dataset else dataset['test']
            
            questions = split_data['question']
            choices = split_data['choices']
            answers = split_data['answer']
            
            # Limit to num_samples if specified
            if num_samples:
                questions = questions[:num_samples]
                choices = choices[:num_samples]
                answers = answers[:num_samples]
            
            result = {
                'questions': questions,
                'choices': choices,
                'answers': answers,
                'split': split,
                'num_samples': len(questions),
                'metadata': {
                    'name': 'MMLU',
                    'total_questions': 14042,
                    'subjects': 57,
                    'format': 'multiple-choice'
                }
            }
            
            logger.info(f"✓ Loaded {len(questions)} MMLU samples")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MMLU from Hugging Face: {e}") from e
    
    def load_hellaswag(self, num_samples: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Load HellaSwag benchmark dataset
        
        HellaSwag: Commonsense understanding benchmark
        - 70,000 multiple-choice questions
        - Predict next sentence in narrative
        - 4 answer options
        
        Args:
            num_samples: Number of samples to load (None = all)
        
        Returns:
            Dictionary with contexts, questions, and answers
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("datasets library not installed")
        
        logger.info("Loading HellaSwag benchmark...")
        
        try:
            dataset = load_dataset("hellaswag")
            split_data = dataset['validation']
            
            contexts = split_data['ctx']
            questions = split_data['activity_label']
            choices = split_data['endings']
            answers = split_data['label']
            
            if num_samples:
                contexts = contexts[:num_samples]
                questions = questions[:num_samples]
                choices = choices[:num_samples]
                answers = answers[:num_samples]
            
            result = {
                'contexts': contexts,
                'questions': questions,
                'choices': choices,
                'answers': answers,
                'num_samples': len(contexts),
                'metadata': {
                    'name': 'HellaSwag',
                    'total_questions': 70000,
                    'format': 'commonsense reasoning',
                    'answer_type': 'next sentence prediction'
                }
            }
            
            logger.info(f"✓ Loaded {len(contexts)} HellaSwag samples")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HellaSwag: {e}") from e
    
    def load_truthfulqa(self, num_samples: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Load TruthfulQA benchmark dataset
        
        TruthfulQA: Truthfulness evaluation
        - 817 questions testing factual knowledge
        - Tests if model produces truthful vs. deceptive answers
        - Questions cover diverse topics
        
        Args:
            num_samples: Number of samples to load (None = all)
        
        Returns:
            Dictionary with questions and reference answers
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("datasets library not installed")
        
        logger.info("Loading TruthfulQA benchmark...")
        
        try:
            dataset = load_dataset("truthful_qa", "generation")
            split_data = dataset['validation']
            
            questions = split_data['question']
            correct_answers = split_data['correct_answers']
            incorrect_answers = split_data['incorrect_answers']
            
            if num_samples:
                questions = questions[:num_samples]
                correct_answers = correct_answers[:num_samples]
                incorrect_answers = incorrect_answers[:num_samples]
            
            result = {
                'questions': questions,
                'correct_answers': correct_answers,
                'incorrect_answers': incorrect_answers,
                'num_samples': len(questions),
                'metadata': {
                    'name': 'TruthfulQA',
                    'total_questions': 817,
                    'format': 'open-ended QA',
                    'evaluation_type': 'truthfulness'
                }
            }
            
            logger.info(f"✓ Loaded {len(questions)} TruthfulQA samples")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TruthfulQA: {e}") from e
    
    def load_wikitext(self, num_samples: Optional[int] = None) -> List[str]:
        """
        Load WikiText dataset for perplexity evaluation
        
        WikiText: Language modeling dataset
        - Extracted from Wikipedia and novels
        - High-quality text for evaluation
        - Commonly used for perplexity calculations
        
        Args:
            num_samples: Number of samples (documents) to load
        
        Returns:
            List of text samples
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("datasets library not installed")
        
        logger.info("Loading WikiText dataset...")
        
        try:
            dataset = load_dataset("wikitext", "wikitext-103-v1")
            split_data = dataset['test']
            
            texts = split_data['text']
            
            # Filter empty texts
            texts = [t for t in texts if len(t.strip()) > 0]
            
            if num_samples:
                texts = texts[:num_samples]
            
            logger.info(f"✓ Loaded {len(texts)} WikiText samples")
            return texts
            
        except Exception as e:
            raise RuntimeError(f"Failed to load WikiText: {e}") from e
    
    def load_all_benchmarks(self, num_samples_per_benchmark: int = 100) -> Dict:
        """
        Load all benchmarks at once
        
        Args:
            num_samples_per_benchmark: Samples per benchmark for testing
        
        Returns:
            Dictionary with all benchmark datasets
        """
        logger.info("Loading all benchmarks...")
        
        benchmarks = {
            'mmlu': self.load_mmlu(num_samples_per_benchmark),
            'hellaswag': self.load_hellaswag(num_samples_per_benchmark),
            'truthfulqa': self.load_truthfulqa(num_samples_per_benchmark),
            'wikitext': self.load_wikitext(num_samples_per_benchmark * 2),
        }
        
        logger.info("✓ All benchmarks loaded successfully")
        return benchmarks
    
    # Legacy offline helpers retained for reference only.
    
    @staticmethod
    def _get_mock_mmlu(num_samples: Optional[int] = None) -> Dict:
        """Mock MMLU data for testing"""
        questions = [
            "What is the capital of France?",
            "Which planet is closest to the sun?",
            "What is the chemical symbol for gold?",
            "Who wrote 'To Kill a Mockingbird'?",
            "What is the largest ocean on Earth?",
        ] * 20  # Repeat to get enough samples
        
        choices = [
            ["Paris", "Lyon", "Marseille", "Nice"],
            ["Mercury", "Venus", "Earth", "Mars"],
            ["Au", "Ag", "Al", "Ar"],
            ["Harper Lee", "Mark Twain", "Jane Austen", "Emily Dickinson"],
            ["Pacific", "Atlantic", "Indian", "Arctic"],
        ] * 20
        
        answers = [0, 0, 0, 0, 0] * 20
        
        if num_samples:
            questions = questions[:num_samples]
            choices = choices[:num_samples]
            answers = answers[:num_samples]
        
        return {
            'questions': questions,
            'choices': choices,
            'answers': answers,
            'split': 'test',
            'num_samples': len(questions),
            'metadata': {
                'name': 'MMLU (Mock)',
                'note': 'For demonstration only',
                'total_questions': 14042,
                'subjects': 57,
            }
        }
    
    @staticmethod
    def _get_mock_hellaswag(num_samples: Optional[int] = None) -> Dict:
        """Mock HellaSwag data for testing"""
        contexts = [
            "A woman is sitting at a table with a man.",
            "A person is preparing food in a kitchen.",
            "Two people are walking in the park.",
            "A child is playing with toys.",
            "A man is working on a computer.",
        ] * 20
        
        questions = [
            "interactions", "cooking", "walking", "playing", "working"
        ] * 20
        
        choices = [
            ["They talk", "They eat", "They leave", "They laugh"],
            ["Add sauce", "Set timer", "Stir pot", "Taste food"],
            ["Stop walking", "Sit down", "Turn back", "Speed up"],
            ["Drop toy", "Pick up", "Throw toy", "Ignore"],
            ["Type fast", "Stand up", "Move mouse", "Close laptop"],
        ] * 20
        
        answers = [0, 1, 2, 1, 0] * 20
        
        if num_samples:
            contexts = contexts[:num_samples]
            questions = questions[:num_samples]
            choices = choices[:num_samples]
            answers = answers[:num_samples]
        
        return {
            'contexts': contexts,
            'questions': questions,
            'choices': choices,
            'answers': answers,
            'num_samples': len(contexts),
            'metadata': {
                'name': 'HellaSwag (Mock)',
                'note': 'For demonstration only'
            }
        }
    
    @staticmethod
    def _get_mock_truthfulqa(num_samples: Optional[int] = None) -> Dict:
        """Mock TruthfulQA data for testing"""
        questions = [
            "Is it safe to eat raw chicken?",
            "What year did World War 2 end?",
            "Can humans see ultraviolet light?",
            "What is the speed of light?",
            "Do vaccines cause autism?",
        ] * 20
        
        correct_answers = [
            ["No, raw chicken can cause foodborne illness"],
            ["1945"],
            ["No, humans cannot see UV light"],
            ["299,792,458 meters per second"],
            ["No, vaccines do not cause autism"],
        ] * 20
        
        incorrect_answers = [
            ["Yes, chicken is safe raw"],
            ["1944"],
            ["Yes, humans can see UV"],
            ["300,000 km/s"],
            ["Yes, vaccines cause autism"],
        ] * 20
        
        if num_samples:
            questions = questions[:num_samples]
            correct_answers = correct_answers[:num_samples]
            incorrect_answers = incorrect_answers[:num_samples]
        
        return {
            'questions': questions,
            'correct_answers': correct_answers,
            'incorrect_answers': incorrect_answers,
            'num_samples': len(questions),
            'metadata': {
                'name': 'TruthfulQA (Mock)',
                'note': 'For demonstration only'
            }
        }
    
    @staticmethod
    def _get_mock_wikitext(num_samples: Optional[int] = None) -> List[str]:
        """Mock WikiText data for testing"""
        texts = [
            "The Roman Empire was one of the greatest civilizations in history.",
            "Machine learning has revolutionized many fields of study.",
            "Climate change is one of the most pressing challenges of our time.",
            "The Internet has transformed how people communicate globally.",
            "Artificial intelligence continues to advance rapidly.",
        ] * 20
        
        if num_samples:
            texts = texts[:num_samples]
        
        return texts


class DatasetEvaluator:
    """Evaluate model performance on benchmark datasets"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.loader = DataLoader()

    def _score_candidate(self, prompt: str, candidate: str) -> float:
        """Score a candidate continuation by average log-likelihood."""
        self.model.eval()

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        ).to(self.device)

        candidate_inputs = self.tokenizer(
            prompt + candidate,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        ).to(self.device)

        labels = candidate_inputs['input_ids'].clone()
        prompt_length = prompt_inputs['input_ids'].shape[1]
        labels[:, :prompt_length] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=candidate_inputs['input_ids'],
                attention_mask=candidate_inputs['attention_mask'],
                labels=labels,
            )

        return -outputs.loss.item()

    @staticmethod
    def _normalize_answer(text: str) -> str:
        return ' '.join(text.strip().lower().split())
    
    def evaluate_mmlu(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate on MMLU benchmark
        
        Returns:
            Dictionary with accuracy and other metrics
        """
        logger.info(f"Evaluating on MMLU ({num_samples} samples)...")
        
        data = self.loader.load_mmlu(num_samples=num_samples)

        correct = 0
        total = len(data['questions'])

        for question, choices, answer in zip(data['questions'], data['choices'], data['answers']):
            prompt = (
                f"Question: {question}\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
                "Answer with the best option only."
            )
            scores = [self._score_candidate(prompt, f" {choice}") for choice in choices]
            predicted_answer = max(range(len(scores)), key=scores.__getitem__)
            correct += int(predicted_answer == int(answer))

        accuracy = correct / total if total else 0.0
        
        return {
            'benchmark': 'MMLU',
            'accuracy': accuracy,
            'num_samples': total,
            'num_questions': data['metadata']['total_questions'],
        }
    
    def evaluate_hellaswag(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate on HellaSwag benchmark"""
        logger.info(f"Evaluating on HellaSwag ({num_samples} samples)...")
        
        data = self.loader.load_hellaswag(num_samples=num_samples)

        correct = 0
        total = len(data['contexts'])

        for context, activity, choices, answer in zip(
            data['contexts'], data['questions'], data['choices'], data['answers']
        ):
            prompt = (
                f"Context: {context}\n"
                f"Activity: {activity}\n"
                "Choose the most likely ending."
            )
            scores = [self._score_candidate(prompt, f" {choice}") for choice in choices]
            predicted_answer = max(range(len(scores)), key=scores.__getitem__)
            correct += int(predicted_answer == int(answer))

        accuracy = correct / total if total else 0.0
        
        return {
            'benchmark': 'HellaSwag',
            'accuracy': accuracy,
            'num_samples': len(data['contexts']),
        }
    
    def evaluate_truthfulqa(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate on TruthfulQA benchmark"""
        logger.info(f"Evaluating on TruthfulQA ({num_samples} samples)...")
        
        data = self.loader.load_truthfulqa(num_samples=num_samples)

        correct = 0
        total = len(data['questions'])

        for question, correct_answers, incorrect_answers in zip(
            data['questions'], data['correct_answers'], data['incorrect_answers']
        ):
            prompt = f"Question: {question}\nAnswer truthfully and concisely."
            best_correct_score = max(
                self._score_candidate(prompt, f" {answer}") for answer in correct_answers
            )
            best_incorrect_score = max(
                self._score_candidate(prompt, f" {answer}") for answer in incorrect_answers
            )
            correct += int(best_correct_score >= best_incorrect_score)

        accuracy = correct / total if total else 0.0
        
        return {
            'benchmark': 'TruthfulQA',
            'accuracy': accuracy,
            'num_samples': len(data['questions']),
        }
    
    def evaluate_all(self, num_samples_per_benchmark: int = 100) -> Dict:
        """
        Evaluate on all benchmarks
        
        Returns:
            Dictionary with results for all benchmarks
        """
        logger.info("Starting comprehensive evaluation...")
        
        results = {
            'mmlu': self.evaluate_mmlu(num_samples_per_benchmark),
            'hellaswag': self.evaluate_hellaswag(num_samples_per_benchmark),
            'truthfulqa': self.evaluate_truthfulqa(num_samples_per_benchmark),
        }
        
        # Calculate average accuracy
        accuracies = [r['accuracy'] for r in results.values()]
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        results['average'] = {
            'accuracy': avg_accuracy,
            'num_benchmarks': len(results),
        }
        
        logger.info(f"✓ Evaluation complete. Average accuracy: {avg_accuracy:.2%}")
        
        return results
