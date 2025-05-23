import torch
import copy
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from ..utils.helper_functions import noop, exists, num_to_groups, cycle
from ..data.utils import video_tensor_to_gif
from ..data.dataset import Dataset
from ..architecture.common import EMA
import torch.nn as nn
from einops import rearrange
from typing import Callable, Optional, Union
import os

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

class Trainer:
    """
    Trainer class for training a diffusion model on two T4 GPUs in a Kaggle environment.
    
    Attributes:
        model: The model to train, wrapped with DataParallel for multi-GPU.
        ema: Exponential moving average of the model.
        ema_model: EMA model, also wrapped with DataParallel.
        update_ema_every: How frequently to update the EMA model.
        step_start_ema: Step at which to start EMA.
        save_model_every: Frequency to save the model.
        batch_size: Batch size used during training.
        image_size: Size of images in the dataset.
        gradient_accumulate_every: Number of steps to accumulate gradients.
        train_num_steps: Total number of training steps.
        ds: Dataset object.
        dl: DataLoader for the dataset.
        opt: Optimizer for training the model.
        step: Current training step.
        amp: Flag to enable automatic mixed precision (AMP).
        scaler: GradScaler used in AMP.
        max_grad_norm: Maximum gradient norm for clipping.
        num_sample_rows: Number of rows for sampling during training.
        results_folder: Folder to save results.
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,  # Diffusion model to train
        folder: str,  # Path to the folder containing training data
        device: str = 'cuda',  # Default to cuda for multi-GPU
        *,
        ema_decay: float = 0.995,  # Exponential moving average decay rate
        num_frames: int = 16,  # Number of frames per video in the dataset
        train_batch_size: int = 1,  # Batch size for training (very small)
        train_lr: float = 1e-4,  # Learning rate for the optimizer
        train_num_steps: int = 100000,  # Number of training steps
        gradient_accumulate_every: int = 1,  # Number of steps to accumulate gradients
        amp: bool = True,  # Use automatic mixed precision for T4 GPUs
        step_start_ema: int = 2000,  # Step to start EMA
        update_ema_every: int = 10,  # Frequency to update EMA
        save_model_every: int = 1000,  # Frequency to save the model
        results_folder: str = './results',  # Folder to save results
        num_sample_rows: int = 4,  # Number of rows for video sampling
        max_grad_norm: Optional[float] = None  # Max gradient norm for clipping
    ):
        """
        Initializes the trainer with parameters optimized for T4 x 2 GPUs.
        
        Args:
            diffusion_model: The model that will be trained.
            folder: Path to the dataset folder (e.g., '/kaggle/input/your-dataset').
            device: Device for training (default: 'cuda' for multi-GPU).
            ema_decay: Decay factor for EMA.
            num_frames: Number of frames in the video data.
            train_batch_size: Batch size for training.
            train_lr: Learning rate.
            train_num_steps: Number of training steps.
            gradient_accumulate_every: Gradient accumulation steps.
            amp: Whether to use automatic mixed precision.
            step_start_ema: Step to start EMA.
            update_ema_every: Frequency to update EMA.
            save_model_every: Frequency to save and sample.
            results_folder: Path to store results and model checkpoints.
            num_sample_rows: Number of rows for video sampling.
            max_grad_norm: Gradient clipping norm.
        """
        super().__init__()
        
        # Validate diffusion model attributes
        required_attrs = ['image_size', 'channels', 'num_frames']
        for attr in required_attrs:
            if not hasattr(diffusion_model, attr) or getattr(diffusion_model, attr) is None:
                raise AttributeError(f"diffusion_model must have a valid '{attr}' attribute")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Ensure Kaggle GPU is enabled.")
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        if num_gpus < 2:
            print("Warning: Expected 2 T4 GPUs, but fewer detected. Using available GPUs.")
        
        self.device = device
        self.model = diffusion_model.to(self.device)
        # Wrap model with DataParallel if multiple GPUs are available
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            print(f"Using {num_gpus} GPUs with DataParallel")
        
        self.ema = EMA(ema_decay)
        # Copy model for EMA, handling DataParallel
        try:
            self.ema_model = copy.deepcopy(self.model.module if isinstance(self.model, nn.DataParallel) else self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to copy model for EMA: {str(e)}")
        
        # Wrap EMA model with DataParallel if multiple GPUs
        if num_gpus > 1:
            self.ema_model = nn.DataParallel(self.ema_model)
        
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_model_every = save_model_every
        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        # Validate dataset folder
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset folder '{folder}' does not exist")
        print(f"Checking dataset folder: {folder}")
        print(f"Files found: {os.listdir(folder)}")
        
        # Initialize dataset
        try:
            self.ds = Dataset(
                folder,
                image_size=diffusion_model.image_size,
                channels=diffusion_model.channels,
                num_frames=diffusion_model.num_frames
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Dataset: {str(e)}")
        
        print(f"Found {len(self.ds)} videos as GIF files at {folder}")
        if len(self.ds) == 0:
            raise ValueError(f"No valid GIF files found in {folder}")
        
        # Initialize DataLoader with optimized num_workers
        self.dl = cycle(torch.utils.data.DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=min(4, os.cpu_count() or 1)  # Adjust for Kaggle
        ))
        
        # Initialize optimizer
        self.opt = Adam(self.model.parameters(), lr=train_lr)
        self.step = 0
        
        # Mixed precision settings for T4 GPUs
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm
        
        # Results folder setup
        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        
        # Initialize EMA model
        try:
            self.reset_parameters()
        except Exception as e:
            raise RuntimeError(f"Failed to reset parameters: {str(e)}")

    def reset_parameters(self):
        """
        Resets the EMA model by copying the current model's state_dict.
        """
        try:
            self.ema_model.load_state_dict(
                self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
            )
        except Exception as e:
            raise RuntimeError(f"Error in reset_parameters: {str(e)}")

    def step_ema(self):
        """
        Updates the EMA model by a weighted average of the model parameters.
        If the training step is before the specified start EMA step, resets the EMA model.
        """
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(
            self.ema_model.module if isinstance(self.ema_model, nn.DataParallel) else self.ema_model,
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

    def save(self, milestone: int):
        """
        Saves the model, EMA model, and other relevant information to a file.

        Args:
            milestone: The current training step (used for checkpoint naming).
        """
        data = {
            'step': self.step,
            'model': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'ema': self.ema_model.module.state_dict() if isinstance(self.ema_model, nn.DataParallel) else self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        print(f"Saved checkpoint to {self.results_folder / f'model-{milestone}.pt'}")

    def load(self, milestone: int, **kwargs):
        """
        Loads a checkpoint from a specific milestone.

        Args:
            milestone: The checkpoint file to load.
            **kwargs: Additional arguments for `state_dict` loading.
        """
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            if not all_milestones:
                raise FileNotFoundError('No checkpoint found to load')
            milestone = max(all_milestones)
        
        checkpoint_path = str(self.results_folder / f'model-{milestone}.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        
        data = torch.load(checkpoint_path)
        
        self.step = data['step']
        model_state = data['model']
        ema_state = data['ema']
        
        try:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(model_state, **kwargs)
            else:
                self.model.load_state_dict(model_state, **kwargs)
            
            if isinstance(self.ema_model, nn.DataParallel):
                self.ema_model.module.load_state_dict(ema_state, **kwargs)
            else:
                self.ema_model.load_state_dict(ema_state, **kwargs)
            
            self.scaler.load_state_dict(data['scaler'])
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {str(e)}")

    def train(
        self,
        prob_focus_present: float = 0.0,
        focus_present_mask: Optional[torch.Tensor] = None,
        log_fn: Callable[[dict], None] = noop
    ):
        """
        Main training loop. Trains the model for a number of steps, updating the model, EMA, and sampling periodically.
        
        Args:
            prob_focus_present: Probability of focusing on the present.
            focus_present_mask: Mask for focusing on specific frames.
            log_fn: Logging function that takes a dictionary as input.
        """
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                # Get the next batch of data
                try:
                    data = next(self.dl)
                except Exception as e:
                    raise RuntimeError(f"Error loading data batch: {str(e)}")

                if len(data) == 2:
                    video_data, text_data = data
                    if text_data is not None:
                        text_data = text_data.to(self.device)
                else:
                    video_data = data
                    text_data = None

                video_data = video_data.to(self.device)

                with autocast(enabled=self.amp):  # Automatic mixed precision
                    try:
                        if text_data is not None:
                            loss = self.model(
                                video_data,
                                cond=text_data,
                                prob_focus_present=prob_focus_present,
                                focus_present_mask=focus_present_mask
                            )
                        else:
                            loss = self.model(
                                video_data,
                                prob_focus_present=prob_focus_present,
                                focus_present_mask=focus_present_mask
                            )
                    except Exception as e:
                        raise RuntimeError(f"Error during model forward pass: {str(e)}")
                    
                    # Handle loss reduction for DataParallel
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()  # Ensure scalar loss

                    # Backpropagate loss
                    try:
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                    except Exception as e:
                        raise RuntimeError(f"Error during backpropagation: {str(e)}")

                print(f'Step {self.step}: Loss = {loss.item():.6f}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            try:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
            except Exception as e:
                raise RuntimeError(f"Error during optimizer step: {str(e)}")

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_model_every == 0:
                milestone = self.step // self.save_model_every
                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('Training completed.')
