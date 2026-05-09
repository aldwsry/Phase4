"""
Stage 1: Hardware-Aware Neural Architecture Search (NAS)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import copy
from .config import StageNASConfig

@dataclass
class ArchitectureCandidate:
    """Represents a candidate architecture"""
    depth_ratio: float  # Layer count reduction
    width_ratio: float  # Hidden dimension reduction
    head_reduction: float  # Attention head reduction
    fitness: float = 0.0
    
    def to_config(self, original_config) -> Dict:
        """Convert to model config"""
        config = copy.deepcopy(original_config)
        
        # Reduce depth
        if hasattr(config, 'num_hidden_layers'):
            config.num_hidden_layers = int(config.num_hidden_layers * self.depth_ratio)
        
        # Reduce width
        if hasattr(config, 'hidden_size'):
            config.hidden_size = int(config.hidden_size * self.width_ratio)
        
        # Reduce attention heads
        if hasattr(config, 'num_attention_heads'):
            config.num_attention_heads = max(1, int(config.num_attention_heads * self.head_reduction))
        
        return config

class HardwareAwareNAS:
    """Hardware-Aware Neural Architecture Search"""
    
    def __init__(self, config: StageNASConfig, model, device='cuda'):
        self.config = config
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.population: List[ArchitectureCandidate] = []
        self.search_history: List[Dict] = []
        
    def initialize_population(self) -> List[ArchitectureCandidate]:
        """Initialize population with random architectures"""
        population = []
        
        for _ in range(self.config.population_size):
            candidate = ArchitectureCandidate(
                depth_ratio=np.random.choice(self.config.search_depth_ratios),
                width_ratio=np.random.choice(self.config.search_width_ratios),
                head_reduction=np.random.uniform(0.5, 1.0),
            )
            population.append(candidate)
        
        self.population = population
        return population
    
    def evaluate_candidate(self, candidate: ArchitectureCandidate, 
                          test_inputs: torch.Tensor) -> float:
        """Evaluate architecture candidate fitness"""
        # Simulate architecture change without rebuilding model
        # In practice, would create modified model and measure latency
        
        # Fitness = combination of:
        # 1. Speedup (how much faster)
        # 2. Accuracy preservation (estimated)
        # 3. Memory reduction
        
        # Estimated speedup based on parameter reduction
        params_reduction = (1 - candidate.depth_ratio * candidate.width_ratio)
        estimated_speedup = 1 + (params_reduction * self.config.target_speedup)
        
        # Accuracy penalty for extreme reductions
        depth_penalty = max(0, 0.3 * (1 - candidate.depth_ratio))
        width_penalty = max(0, 0.2 * (1 - candidate.width_ratio))
        accuracy_estimate = 1.0 - depth_penalty - width_penalty
        
        # Combined fitness
        fitness = (estimated_speedup * 0.5) + (accuracy_estimate * 0.5)
        
        return fitness
    
    def selection(self, tournament_size: int = 3) -> List[ArchitectureCandidate]:
        """Tournament selection"""
        selected = []
        
        for _ in range(len(self.population)):
            tournament = np.random.choice(
                len(self.population),
                size=tournament_size,
                replace=False
            )
            best_idx = max(tournament, key=lambda i: self.population[i].fitness)
            selected.append(copy.deepcopy(self.population[best_idx]))
        
        return selected
    
    def crossover(self, parent1: ArchitectureCandidate, 
                  parent2: ArchitectureCandidate) -> Tuple[ArchitectureCandidate, ArchitectureCandidate]:
        """Single-point crossover"""
        # Randomly mix parent characteristics
        child1 = ArchitectureCandidate(
            depth_ratio=parent1.depth_ratio,
            width_ratio=parent2.width_ratio,
            head_reduction=np.random.choice([parent1.head_reduction, parent2.head_reduction])
        )
        
        child2 = ArchitectureCandidate(
            depth_ratio=parent2.depth_ratio,
            width_ratio=parent1.width_ratio,
            head_reduction=np.random.choice([parent2.head_reduction, parent1.head_reduction])
        )
        
        return child1, child2
    
    def mutate(self, candidate: ArchitectureCandidate) -> ArchitectureCandidate:
        """Mutation operation"""
        mutated = copy.deepcopy(candidate)
        
        # Mutate depth
        if np.random.random() < 0.3:
            mutated.depth_ratio = np.random.choice(self.config.search_depth_ratios)
        
        # Mutate width
        if np.random.random() < 0.3:
            mutated.width_ratio = np.random.choice(self.config.search_width_ratios)
        
        # Mutate head reduction
        if np.random.random() < 0.3:
            mutated.head_reduction = np.clip(
                mutated.head_reduction + np.random.normal(0, 0.05),
                0.5, 1.0
            )
        
        return mutated
    
    def search(self, num_generations: Optional[int] = None) -> ArchitectureCandidate:
        """Genetic algorithm search"""
        num_generations = num_generations or self.config.generations
        
        # Initialize population
        self.initialize_population()
        
        print(f"Starting NAS search: {self.config.population_size} population, {num_generations} generations")
        
        for gen in range(num_generations):
            # Evaluate population
            for candidate in self.population:
                candidate.fitness = self.evaluate_candidate(candidate, None)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Log progress
            best_fitness = self.population[0].fitness
            avg_fitness = np.mean([c.fitness for c in self.population])
            
            self.search_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_candidate': self.population[0]
            })
            
            if (gen + 1) % 10 == 0:
                print(f"Gen {gen+1}/{num_generations}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Early stopping
            if best_fitness > 0.95:  # Near-optimal solution
                print(f"Converged at generation {gen+1}")
                break
            
            # Selection, crossover, mutation
            selected = self.selection()
            new_population = []
            
            # Keep top candidates (elitism)
            new_population.extend([copy.deepcopy(c) for c in self.population[:2]])
            
            # Generate offspring
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < len(selected) else selected[i]
                
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                if np.random.random() < self.config.mutation_rate:
                    child1 = self.mutate(child1)
                if np.random.random() < self.config.mutation_rate:
                    child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure population size
            self.population = new_population[:self.config.population_size]
        
        # Return best architecture
        best_candidate = max(self.population, key=lambda x: x.fitness)
        print(f"\nBest architecture found: {best_candidate}")
        return best_candidate
    
    def get_optimal_config(self) -> Dict:
        """Get optimal model configuration"""
        best_candidate = max(self.population, key=lambda x: x.fitness)
        return best_candidate.to_config(self.model.config)
