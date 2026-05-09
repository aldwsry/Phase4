"""
Stage 3: Knowledge Distillation (FIXED VERSION)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Union, Dict, Any
from tqdm import tqdm


# =========================================================
# LOSS FUNCTION
# =========================================================

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels=None):

        # soft targets
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # optional hard loss
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None:
            ce_loss = self.ce_loss(student_logits, labels)

        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss


# =========================================================
# DISTILLER
# =========================================================

class KnowledgeDistiller:
    def __init__(self, teacher_model, student_model, config, device="cuda"):

        self.config = config

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # models
        self.teacher_model = teacher_model.to(self.device).eval()
        self.student_model = student_model.to(self.device).train()

        # loss
        self.kd_loss = KnowledgeDistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha
        )

        # optimizer (FIXED BUG HERE)
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate
        )

        self.history = {"loss": []}

    # -----------------------------------------------------
    # SINGLE STEP
    # -----------------------------------------------------
    def distill_step(self, batch):

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # teacher
        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_out.logits

        # student
        student_out = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_out.logits

        # reshape
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

        loss = self.kd_loss(student_logits, teacher_logits)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    # -----------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------
    def train(self, train_loader, num_epochs=1):

        print("\n🚀 Starting Knowledge Distillation")

        for epoch in range(num_epochs):
            total_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for batch in pbar:
                loss = self.distill_step(batch)
                total_loss += loss
                pbar.set_postfix({"loss": loss})

            avg_loss = total_loss / len(train_loader)
            self.history["loss"].append(avg_loss)

            print(f"\n📊 Epoch {epoch+1} Loss: {avg_loss:.4f}")

        print("\n✅ Distillation Complete!")

        return self.history

    # -----------------------------------------------------
    def _benchmark_dict_to_texts(self, benchmark_data):
        texts = []

        for name, dataset in benchmark_data.items():
            if isinstance(dataset, list):
                if name == "wikitext":
                    texts.extend([text for text in dataset if isinstance(text, str) and text.strip()])
                    continue

                for sample in dataset:
                    if isinstance(sample, str) and sample.strip():
                        texts.append(sample)
                    elif isinstance(sample, dict):
                        texts.append(" ".join(str(v) for v in sample.values() if v is not None))
                continue

            if not isinstance(dataset, dict):
                continue

            if name == "mmlu":
                questions = dataset.get("questions", [])
                choices = dataset.get("choices", [])
                answers = dataset.get("answers", [])
                for index, question in enumerate(questions):
                    choice_text = ""
                    if index < len(choices):
                        choice_text = " | ".join(str(choice) for choice in choices[index])
                    answer_text = ""
                    if index < len(answers):
                        answer_value = answers[index]
                        if isinstance(answer_value, int) and index < len(choices) and 0 <= answer_value < len(choices[index]):
                            answer_text = str(choices[index][answer_value])
                        else:
                            answer_text = str(answer_value)
                    prompt = f"Question: {question}\nChoices: {choice_text}\nAnswer: {answer_text}".strip()
                    texts.append(prompt)

            elif name == "hellaswag":
                contexts = dataset.get("contexts", [])
                questions = dataset.get("questions", [])
                choices = dataset.get("choices", [])
                answers = dataset.get("answers", [])
                for index, context in enumerate(contexts):
                    question_text = questions[index] if index < len(questions) else ""
                    choice_text = ""
                    if index < len(choices):
                        choice_text = " | ".join(str(choice) for choice in choices[index])
                    answer_text = answers[index] if index < len(answers) else ""
                    prompt = (
                        f"Context: {context}\n"
                        f"Activity: {question_text}\n"
                        f"Choices: {choice_text}\n"
                        f"Answer: {answer_text}"
                    ).strip()
                    texts.append(prompt)

            elif name == "truthfulqa":
                questions = dataset.get("questions", [])
                correct_answers = dataset.get("correct_answers", [])
                for index, question in enumerate(questions):
                    answer_text = ""
                    if index < len(correct_answers):
                        candidate = correct_answers[index]
                        if isinstance(candidate, list):
                            answer_text = " | ".join(str(item) for item in candidate)
                        else:
                            answer_text = str(candidate)
                    prompt = f"Question: {question}\nCorrect Answers: {answer_text}".strip()
                    texts.append(prompt)

        return [text for text in texts if isinstance(text, str) and text.strip()]

    # -----------------------------------------------------
    # FLEXIBLE ENTRY POINT (IMPORTANT FIX)
    # -----------------------------------------------------
    def distill(self, benchmark_data, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        # If already DataLoader
        if isinstance(benchmark_data, DataLoader):
            return self.train(benchmark_data, num_epochs)

        # If dict → build a DataLoader from benchmark prompts
        if isinstance(benchmark_data, dict):
            texts = self._benchmark_dict_to_texts(benchmark_data) 
            texts = texts[:self.config.num_distill_samples]  # Limit to configured number of samples
            if not texts:
                raise ValueError("No usable text samples were found in benchmark_data")

            if not hasattr(self, "tokenizer") or self.tokenizer is None:
                raise RuntimeError(
                    "KnowledgeDistiller requires a tokenizer when passing benchmark dictionaries.\n"
                    "Assign a tokenizer to distiller.tokenizer before calling distill()."
                )

            class TextDataset:
                def __init__(self, samples):
                    self.samples = samples

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return self.samples[idx]

            def collate_fn(batch):
                return self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            loader = DataLoader(
                TextDataset(texts),
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            return self.train(loader, num_epochs)

        raise TypeError("benchmark_data must be DataLoader or dict")

    # -----------------------------------------------------
    def get_distillation_metrics(self):
        return {
            "final_loss": self.history["loss"][-1] if self.history["loss"] else None,
            "loss_curve": self.history["loss"]
        }

    # -----------------------------------------------------
    def save_student(self, path):
        self.student_model.save_pretrained(path)
        print(f"💾 Saved student model at {path}")