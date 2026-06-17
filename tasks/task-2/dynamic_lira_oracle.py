import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rdkit import Chem

from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters

# Import Task 2 model logic
sys.path.append("/home/krict_lukas/hackathon_copenhagen/tasks/task-2")
from model.model import load_model, MLP, smiles_to_ecfp4

class DynamicLiRAOracle(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        
        # 1. Load the Target Model (The model we want to attack/extract data from)
        model_path = self.specific_parameters.get(
            "model_path", 
            "/home/krict_lukas/hackathon_copenhagen/tasks/task-2/model/model.safetensors"
        )
        self.target_model = load_model(model_path)
        self.target_model.eval()
        
        # 2. Initialize the Shadow Model (Same architecture, random weights)
        self.shadow_model = MLP()
        # Using the hyperparameters specified in task-2 instructions
        self.optimizer = optim.AdamW(self.shadow_model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # 3. Replay Buffer to store generated molecules
        self.buffer_ecfp = []
        self.buffer_target_logits = []
        
        # 4. Hyperparameters for Dynamic Training
        self.train_interval = self.specific_parameters.get("train_interval", 10)  # Train every 10 calls
        self.epochs_per_train = self.specific_parameters.get("epochs_per_train", 5) # Epochs per training
        self.call_counter = 0

    def _train_shadow_model(self):
        """
        Train the shadow model to mimic the Target Model on the newly generated data.
        This is called Knowledge Distillation.
        """
        if len(self.buffer_ecfp) == 0:
            return
            
        # Convert buffer to tensors
        X_train = torch.from_numpy(np.vstack(self.buffer_ecfp).astype(np.float32))
        Y_target = torch.from_numpy(np.vstack(self.buffer_target_logits).astype(np.float32))
        
        # Convert target logits to soft probabilities (what the target model thinks)
        target_probs = torch.softmax(Y_target, dim=-1)
        
        self.shadow_model.train()
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs_per_train):
            self.optimizer.zero_grad()
            shadow_logits = self.shadow_model(X_train)
            
            # Train shadow model to predict the same probabilities as the target model
            loss = criterion(shadow_logits, target_probs)
            
            loss.backward()
            self.optimizer.step()
            
        self.shadow_model.eval()
        
        # Optional: Keep the buffer size manageable to prevent memory overflow
        max_buffer_size = 50000
        if len(self.buffer_ecfp) > max_buffer_size:
            self.buffer_ecfp = self.buffer_ecfp[-max_buffer_size:]
            self.buffer_target_logits = self.buffer_target_logits[-max_buffer_size:]

    def __call__(self, mols: np.ndarray[Chem.Mol]) -> np.ndarray[float]:
        self.call_counter += 1
        
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        ecfp_list = [smiles_to_ecfp4(s) for s in smiles_list]
        
        scores = np.zeros(len(smiles_list), dtype=np.float32)
        valid_indices = [i for i, fp in enumerate(ecfp_list) if fp is not None]
        valid_ecfp = [ecfp_list[i] for i in valid_indices]

        if not valid_ecfp:
            return scores
            
        X = torch.from_numpy(np.asarray(valid_ecfp, dtype=np.float32))

        # --- Dynamic LiRA Workflow ---
        with torch.no_grad():
            # A. Get Target Model Predictions
            target_logits = self.target_model(X) # (N, 4)
            
        # B. Store into buffer and trigger Shadow Model training periodically
        self.buffer_ecfp.extend(valid_ecfp)
        self.buffer_target_logits.extend(target_logits.cpu().numpy())
        
        if self.call_counter % self.train_interval == 0:
            self._train_shadow_model()
            
        with torch.no_grad():
            # C. Get Shadow Model Predictions
            self.shadow_model.eval()
            shadow_logits = self.shadow_model(X)
            
            # D. Calculate LiRA Score
            # We check the confidence of the class that the Target model chose
            target_max_logits, target_preds = torch.max(target_logits, dim=-1)
            
            # Extract what the shadow model thinks about that specific class
            shadow_class_logits = shadow_logits.gather(1, target_preds.unsqueeze(1)).squeeze(1)
            
            # LiRA Score: If Target is much more confident than Shadow, it's memorized!
            lira_score = (target_max_logits - shadow_class_logits).cpu().numpy()
            
            # E. Normalization (Min-Max to 0~1)
            min_score = -5.0
            max_score = 5.0
            lira_score = (lira_score - min_score) / (max_score - min_score)
            lira_score = np.clip(lira_score, 0.0, 1.0)
            
        scores[valid_indices] = lira_score
        return scores
